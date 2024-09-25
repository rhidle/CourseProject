import pandas as pd
import numpy as np
import pyomo.environ as pyo


S = range(1,4) # Scenarios
def Problem(s):
    T = range(48) # Time periods
    # Parameters
    P_EV_max = 7.0  # Maximum EV charging power (kW)
    E_EV_required = 50.0  # Total energy required by the EV (kWh)

    P_house = pd.DataFrame() # Consumption data for 48 hours
    P_house_all = pd.read_csv('house9_data.csv', index_col=0)
    P_house['Consumption'] = P_house_all['total_consumption']['2018-06-01':'2018-06-03']
    # P_house.index = pd.to_datetime(P_house_all['2018-06-01':'2018-06-03'].index)
    P_house.reset_index(drop=True, inplace=True)
    P_house = P_house['Consumption'].to_dict()


    p ={
    1 : pd.read_csv('Prices_June.csv')['Price'] / 1000, # Scenario 1: June prices for 48 hours $/kWh
    2 : pd.read_csv('Prices_July.csv')['Price'] / 1000, # Scenario 2: July prices for 48 hours $/kWh
    3 : pd.read_csv('Prices_August.csv')['Price'] / 1000 # Scenario 3: August prices for 48 hours $/kWh
    }

    # Probability of each scenario
    pi = {
    1 : 0.4, 
    2 : 0.3, 
    3 : 0.3 
    }

    # Create a Pyomo model
    model = pyo.ConcreteModel()

    # Define sets
    model.T = pyo.Set(initialize=T)

    # Variables
    model.P_EV = pyo.Var(T, domain=pyo.NonNegativeReals) # EV charging power (kW)

    # Objective function
    model.obj = pyo.Objective(expr=sum(p[s][t]*(model.P_EV[t] + P_house[t]) for t in T), sense=pyo.minimize)

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
    print(f'Scenario {s}')
    print(model.display(), model.dual.display())

for s in S:
    Problem(s)