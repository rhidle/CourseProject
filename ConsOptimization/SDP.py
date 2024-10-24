import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


T = range(48) # Time periods
T1 = range(24) # Time periods for stage 1
T2 = range(24, 48) # Time periods for stage 2
S = range(1,4) # Scenarios

# Parameters
P_EV_max = 7.0       # Maximum EV charging power (kW)
E_EV_required = 50.0 # Total energy required by the EV (kWh)

# House consumption data for 48 hours
P_house_all = pd.read_csv('house9_data.csv', index_col=0)
P_house = P_house_all['total_consumption']['2018-06-01':'2018-06-03']
P_house.reset_index(drop=True, inplace=True)
P_house1 = P_house[:24].to_dict()
P_house2 = P_house[24:48].to_dict()

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

# Mathematical formulation 1st stage
def Obj_1st(m):
    return sum(m.p1[t]*(m.P_house1[t]+m.P_EV[t]) for t in m.T1) + m.alpha
def ChargeLimit(m,t):
    return m.P_EV[t] <= m.P_EV_max
def ChargeRequirement1(m):
    return sum(m.P_EV[t] for t in m.T1) <= m.E_EV_req
def CreateCuts(m,c):
    return(m.alpha >= m.Phi[c] + m.Lambda[c]*(sum(m.P_EV[t] for t in m.T1) - m.x_hat[c]))


#Mathematical formulation 2nd stage
def Obj_2nd(m):
    return sum(m.pi[s]*sum(m.p2s[s,t]*(m.P_house2[t]+m.P_EV[s,t]) for t in m.T2) for s in S)
def ChargeRequirement(m,s):
    return sum(m.P_EV[s,t] for t in m.T2) + m.X_hat == m.E_EV_req
def ChargeLimit2(m,s,t):
    return m.P_EV[s,t] <= m.P_EV_max


# Set up model 1st stage
def ModelSetUp_1st(T1, p1, P_house1, Cuts):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.T1 = pyo.Set(initialize = T1)  # First-stage time periods
    #Parameters
    m.P_EV_max = pyo.Param(initialize = 7)
    m.E_EV_req = pyo.Param(initialize = 50)
    m.p1 = pyo.Param(m.T1, initialize = p1)
    m.P_house1 = pyo.Param(m.T1, initialize = P_house1)

    #Variables
    m.P_EV = pyo.Var(m.T1, domain=pyo.NonNegativeReals)
    
    #m.C.display()
    """Cuts_information"""
    #Set for cuts
    m.Cut = pyo.Set(initialize = Cuts["Set"])
    #Parameter for cuts
    m.Phi = pyo.Param(m.Cut, initialize = Cuts["Phi"])
    m.Lambda = pyo.Param(m.Cut, initialize = Cuts["Lambda"])
    m.x_hat = pyo.Param(m.Cut, initialize = Cuts["x_hat"])
    #Variable for alpha
    m.alpha = pyo.Var(bounds = (-1000000,1000000))
    
    """Constraint cut"""
    m.CreateCuts = pyo.Constraint(m.Cut,rule = CreateCuts)
    
    
    """Constraints"""
    m.ChargeLimit = pyo.Constraint(m.T1,rule=ChargeLimit)
    m.ChargeRequirement1 = pyo.Constraint(rule=ChargeRequirement1)
    
    # Define objective function
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)
    
    #m.display()
    
    return m

# Set up model 2nd stage
def ModelSetUp_2nd(p2s,S,T1,T2,pi,P_house2,X_hat):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.T2 = pyo.Set(initialize = T2)  # Second-stage time periods
    m.T1 = pyo.Set(initialize = T1)  # First-stage time periods
    m.S = pyo.Set(initialize = S)    # Scenarios
    # Flatten p2s into a dictionary with keys (s, t)
    p2s_2d = {}
    for s in S:
        for t in p2s[s]:
            p2s_2d[(s, t)] = p2s[s][t]

    # Define parameters
    m.P_EV_max = pyo.Param(initialize = 7)
    m.E_EV_req = pyo.Param(initialize = 50)
    m.X_hat = pyo.Param(initialize = X_hat)
    m.p2s = pyo.Param(m.S, m.T2, initialize = p2s_2d)
    m.P_house2 = pyo.Param(m.T2, initialize = P_house2)
    m.pi = pyo.Param(m.S, initialize = pi)
    # Define variables
    m.P_EV = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals)
    
    # Define constraints
    m.ChargeLimit2 = pyo.Constraint(m.S, m.T2,rule=ChargeLimit2)
    m.ChargeRequirement = pyo.Constraint(m.S, rule=ChargeRequirement)
    
    # Define objective function
    m.obj = pyo.Objective(rule=Obj_2nd, sense=pyo.minimize)
    return m

def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m
def DisplayResults(m):
    return print(m.display(), m.dual.display())


def Cut_manage(Cuts, dual_value, second_stage_cost, X_hat):
    cut = len(Cuts["Set"])
    Cuts["Set"].append(cut)
    # Compute expected Phi
    Cuts["Phi"][cut] = second_stage_cost
    # Compute expected Lambda
    Cuts["Lambda"][cut] = dual_value
    Cuts["x_hat"][cut] = X_hat
    return Cuts

def SDP():
    """Setup for Stochastic dynamic programming - Data curation"""
    
    #Pre-step: determine the discretization we want to explore in the second-stage
    Min = 0
    Max = int(E_EV_required)
    #How large each discrete jump is in value
    states_jump = 1
    List_charged = [i for i in range(Min,Max+states_jump,states_jump)]
    
    
    """Start the SDP process"""
    # Store second-stage results
    SecondStageResults = {}
    for X_hat in List_charged:
        m_2nd = ModelSetUp_2nd(p2s, S, T1, T2, pi, P_house2, X_hat)
        Solve(m_2nd)

        # Obtain second-stage cost and dual values
        second_stage_cost = pyo.value(m_2nd.obj)
        dual_values = {}
        for s in S:
            dual_values[s] = m_2nd.dual[m_2nd.ChargeRequirement[s]]
        dual_value = -sum(dual_values[s] for s in S)

        # Store results
        SecondStageResults[X_hat] = {
            'cost': second_stage_cost,
            'dual': dual_value
        }

    # Formulate cut input data
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["Lambda"] = {}
    Cuts["x_hat"] = {}
    #For each combination we acquired
    for idx, X_hat in enumerate(List_charged):
        Cuts['Set'].append(idx)
        Cuts['Phi'][idx] = SecondStageResults[X_hat]['cost']
        Cuts['Lambda'][idx] = SecondStageResults[X_hat]['dual']
        Cuts['x_hat'][idx] = X_hat

    
    #Solve the 1st stage problem with the acquired cuts
    m_1st = ModelSetUp_1st(T1,p1,P_house1,Cuts)
    Solve(m_1st)

    
    # Display first-stage results
    FirstStageCharge = {t: pyo.value(m_1st.P_EV[t]) for t in T1}
    TotalFirstStageEnergy = sum(FirstStageCharge[t] for t in T1)
    
    print("First-Stage Decisions:")
    for t in T1:
        print(f"Hour {t}: P_EV = {FirstStageCharge[t]:.2f} kW")
    
    print(f"\nTotal Energy Charged in First Stage: {TotalFirstStageEnergy:.2f} kWh")
    # print("Hour ; Scenario 1 ; Scenario 2 ; Scenario 3")
    # for t in T2:
    #   print(f"Hour {t}; {pyo.value(m_2nd.P_EV[1,t])} ; {pyo.value(m_2nd.P_EV[2,t])} ; {pyo.value(m_2nd.P_EV[3,t])}")
    print(f"Objective Function Value: {pyo.value(m_1st.obj):.2f}")

    return()
SDP()

# m_1st, m_2nd = Benders()
# print("End of iterations \n Final results")
# for t in T1:
#     print(f"Hour {t}: P_EV = {pyo.value(m_1st.P_EV[t])} kW")
# print("Objective function:",pyo.value(m_1st.obj))

# print("Hour ; Scenario 1 ; Scenario 2 ; Scenario 3")
# for t in T2:
#     print(f"Hour {t}; {pyo.value(m_2nd.P_EV[1,t])} ; {pyo.value(m_2nd.P_EV[2,t])} ; {pyo.value(m_2nd.P_EV[3,t])}")

# print("Objective function:",pyo.value(m_2nd.obj)+pyo.value(m_1st.obj)-m_1st.alpha.value)