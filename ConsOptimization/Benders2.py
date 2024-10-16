import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Load Data
T = range(48)  # Time periods
T1 = range(24)  # First-stage time periods
T2 = range(24, 48)  # Second-stage time periods
S = [1, 2, 3]  # Scenarios

# Parameters
P_EV_max = 7.0       # Maximum EV charging power (kW)
E_EV_required = 50.0 # Total energy required by the EV (kWh)

# House consumption data for 48 hours
P_house_all = pd.read_csv('house9_data.csv', index_col=0)
P_house = P_house_all['total_consumption']['2018-06-01':'2018-06-03']
P_house.reset_index(drop=True, inplace=True)
P_house = P_house.to_dict()

# Scenario-specific electricity prices for the first 24 hours ($/kWh)
p1 = pd.read_csv('Prices_June.csv')['Price'][:24] / 1000  # Assuming prices are in $/MWh
p1 = p1.to_dict()

# Scenario-specific electricity prices for the last 24 hours ($/kWh)
p2s = {
    1: pd.read_csv('Prices_June.csv')['Price'][24:48] / 1000,   # Scenario 1
    2: pd.read_csv('Prices_July.csv')['Price'][24:48] / 1000,   # Scenario 2
    3: pd.read_csv('Prices_August.csv')['Price'][24:48] / 1000  # Scenario 3
}

# Convert to dictionaries
for s in S:
    p2s[s] = p2s[s].to_dict()

# Probability of each scenario
pi = {
    1: 0.4,
    2: 0.3,
    3: 0.3
}

# Initialize Benders Cuts
Cuts = {s: [] for s in S}  # Cuts for each scenario

# Initialize Master Problem
def MasterProblem(P_house, p1, pi, P_EV_max, Cuts, E_EV_required):
    model = pyo.ConcreteModel()
    model.T1 = pyo.Set(initialize=T1)
    model.S = pyo.Set(initialize=S)
    
    # Variables
    model.P_EV_1 = pyo.Var(model.T1, domain=pyo.NonNegativeReals)
    model.eta = pyo.Var(model.S, domain=pyo.Reals)
    
    # Objective
    model.obj = pyo.Objective(
        expr=sum(p1[t] * (model.P_EV_1[t] + P_house[t]) for t in model.T1) + sum(pi[s] * model.eta[s] for s in model.S),
        sense=pyo.minimize
    )

    # Energy Constraint
    def EnergyRequirement_rule(model):
        return sum(model.P_EV_1[t] for t in model.T1) <= E_EV_required
    model.EnergyRequirement = pyo.Constraint(rule=EnergyRequirement_rule)
    
    # Power Limits
    def Max_EV_Power_rule(model, t):
        return model.P_EV_1[t] <= P_EV_max
    model.Max_EV_Power = pyo.Constraint(model.T1, rule=Max_EV_Power_rule)
    
    # Benders Cuts
    def BendersCut_rule(model, s, k):
        cut = Cuts[s][k]
        return model.eta[s] >= cut['theta'] + sum(cut['beta'][t] * (model.P_EV[t] - cut['P_EV_hat'][t]) for t in model.T1)
    
    model.BendersCuts = pyo.ConstraintList()
    for s in model.S:
        for k in range(len(Cuts[s])):
            model.BendersCuts.add(BendersCut_rule(model, s, k))
    
    for t in model.T1:
        print(f"Hour {t}: P_EV = {pyo.value(model.P_EV_1[t])} kW")

    return model

# Subproblem for each scenario
def SubProblem(s, P_EV_hat, P_house, p2s, P_EV_max, E_EV_required):
    model = pyo.ConcreteModel()
    model.T2 = pyo.Set(initialize=T2)
    
    # Variables
    model.P_EV_2 = pyo.Var(model.T2, domain=pyo.NonNegativeReals)
    
    # Objective
    model.obj = pyo.Objective(
        expr=sum(p2s[s][t] * (model.P_EV_2[t] + P_house[t]) for t in model.T2),
        sense=pyo.minimize
    )
    
    # Energy Requirement
    first_stage_energy = sum(P_EV_hat[t] for t in T1)
    def EnergyRequirement_rule(model):
        return sum(model.P_EV_2[t] for t in model.T2) == E_EV_required - first_stage_energy
    model.EnergyRequirement = pyo.Constraint(rule=EnergyRequirement_rule)
    
    def Max_EV_Power_rule(model, t):
        return model.P_EV_2[t] <= P_EV_max
    model.Max_EV_Power = pyo.Constraint(model.T2, rule=Max_EV_Power_rule)

    return model

# Main Benders Decomposition Loop
max_iterations = 20
tolerance = 1e-2
iteration = 0
Gap = float('inf')
LB = -float('inf')
UB = float('inf')

while iteration < max_iterations and Gap > tolerance:
    # Solve Master Problem
    master = MasterProblem(P_house, p1, pi, P_EV_max, Cuts, E_EV_required)
    solver = SolverFactory('gurobi')
    result = solver.solve(master, tee=False)
    for t in master.T1:
        print(f"Hour {t}: P_EV = {pyo.value(master.P_EV_1[t])} kW")

    P_EV_hat = {t: pyo.value(master.P_EV[t]) for t in master.T1}
    eta_hat = {s: pyo.value(master.eta[s]) for s in S}
    
    # Update Lower Bound
    LB = pyo.value(master.obj)
    
    # Initialize total expected second-stage cost
    expected_Q = 0
    
    # Solve Subproblems
    for s in S:
        sub = SubProblem(s, P_EV_hat, P_house, p2s, P_EV_max, E_EV_required)
        sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        result = solver.solve(sub, tee=False)
        
        # Check feasibility
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            # Generate Feasibility Cut
            # Not expected in this problem as the subproblem should always be feasible
            print(f"Subproblem for scenario {s} is infeasible.")
            continue
        
        # Calculate second-stage cost Q_s
        Q_s = pyo.value(sub.obj)
        expected_Q += pi[s] * Q_s
        
        # Get dual variables
        duals = sub.dual
        
        # Retrieve dual multipliers for energy constraint
        mu = duals[sub.EnergyRequirement]
        
        # Generate Benders Optimality Cut
        theta = Q_s - mu * (E_EV_required - sum(P_EV_hat[t] for t in T1))
        beta = {t: -mu for t in T1}
        
        # Store the cut
        Cuts[s].append({
            'theta': theta,
            'beta': beta,
            'P_EV_hat': P_EV_hat
        })
    
    # Update Upper Bound
    UB = sum(p1[t] * (P_EV_hat[t] + P_house[t]) for t in T1) + expected_Q
    
    # Calculate Gap
    Gap = abs(UB - LB)
    
    iteration += 1
    print(f"Iteration {iteration}: LB = {LB}, UB = {UB}, Gap = {Gap}")
    print(f"First-stage decisions P_EV_t:")
    for t in T1:
        print(f"  Hour {t}: P_EV = {P_EV_hat[t]:.2f} kW")
    
    print(f"Expected second-stage cost: {expected_Q:.2f}")
    print("-" * 50)

# Final Solution
print("\nOptimal First-stage Decisions:")
for t in T1:
    print(f"Hour {t}: P_EV = {P_EV_hat[t]:.2f} kW")

print("\nOptimal Expected Cost:", UB)
