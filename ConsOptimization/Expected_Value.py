import pandas as pd
import numpy as np
import pyomo.environ as pyo

T = range(48) # Time periods
T1 = range(24) # Time periods for stage 1
T2 = range(24, 48) # Time periods for stage 2
S = range(1,4) # Scenarios


# Parameters
P_EV_max = 7.0  # Maximum EV charging power (kW)
E_EV_required = 50.0  # Total energy required by the EV (kWh)

P_house = pd.DataFrame() # Consumption data for 48 hours
P_house_all = pd.read_csv('house9_data.csv', index_col=0)
P_house['Consumption'] = P_house_all['total_consumption']['2018-06-01':'2018-06-03']
# P_house.index = pd.to_datetime(P_house_all['2018-06-01':'2018-06-03'].index)
P_house.reset_index(drop=True, inplace=True)
P_house = P_house['Consumption'].to_dict()

# Scenario-specific electricity prices for the first 24 hours ($/kWh)
p1 = pd.read_csv('Prices_June.csv')['Price'][:24] / 1000  # Assuming prices are in $/MWh


# Scenario-specific electricity prices for the last 24 hours ($/kWh)
p2s ={
1 : pd.read_csv('Prices_June.csv')['Price'][24:48] / 1000, # Scenario 1: June prices for 48 hours $/kWh
2 : pd.read_csv('Prices_July.csv')['Price'][24:48] / 1000, # Scenario 2: July prices for 48 hours $/kWh
3 : pd.read_csv('Prices_August.csv')['Price'][24:48] / 1000 # Scenario 3: August prices for 48 hours $/kWh
}

# Probability of each scenario
pi = {
1 : 0.4, 
2 : 0.3, 
3 : 0.3 
}

#### Using the expected values for each scenario ####
p1 = p1.to_dict()
p2 = sum(pi[s]*p2s[s] for s in S).round(4).to_dict()

# Create a Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.T = pyo.Set(initialize=T)

# Variables
model.P_EV = pyo.Var(T, domain=pyo.NonNegativeReals) # EV charging power (kW)

# Objective function
def Objective_rule(model):
    return sum(p1[t]*(model.P_EV[t] + P_house[t]) for t in T1) + sum(p2[t]*(model.P_EV[t] + P_house[t]) for t in T2)
model.obj = pyo.Objective(rule = Objective_rule, sense = pyo.minimize)

# Constraints

# EV Requirement Constraint
def EV_Requirement_rule(model):
    return sum(model.P_EV[t] for t in T) == E_EV_required
model.EV_Requirement = pyo.Constraint(rule=EV_Requirement_rule)

# Maximum EV Charging Power Constraint
def Max_EV_Power_rule(model, t):
    return model.P_EV[t] <= P_EV_max
model.Max_EV_Power = pyo.Constraint(T, rule=Max_EV_Power_rule)

# Solve the model
opt = pyo.SolverFactory('gurobi')
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
results = opt.solve(model, load_solutions=True)
print(model.display(), model.dual.display())
