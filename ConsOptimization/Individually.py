import pandas as pd
import numpy as np
import pyomo.environ as pyo


S = range(1,4) # Scenarios
def Problem(s):
    T = range(48) # Time periods
    T1 = range(24) # Time periods for stage 1
    T2 = range(24, 48) # Time periods for stage 2
    # Parameters
    P_EV_max = 7.0  # Maximum EV charging power (kW)
    E_EV_required = 50.0  # Total energy required by the EV (kWh)

    P_house = pd.DataFrame() # Consumption data for 48 hours
    P_house_all = pd.read_csv('house9_data.csv', index_col=0)
    P_house['Consumption'] = P_house_all['total_consumption']['2018-06-01':'2018-06-03']
    # P_house.index = pd.to_datetime(P_house_all['2018-06-01':'2018-06-03'].index)
    P_house.reset_index(drop=True, inplace=True)
    P_house = P_house['Consumption'].to_dict()

    p1 = pd.read_csv('Prices_June.csv')['Price'][:24] / 1000  # Prices in $/kWh
    p1 = p1.to_dict()

    p2s ={
    1 : pd.read_csv('Prices_June.csv')['Price'][24:48] / 1000, # Scenario 1: June prices for 48 hours $/kWh
    2 : pd.read_csv('Prices_July.csv')['Price'][24:48] / 1000, # Scenario 2: July prices for 48 hours $/kWh
    3 : pd.read_csv('Prices_August.csv')['Price'][24:48] / 1000 # Scenario 3: August prices for 48 hours $/kWh
    }
    p2 = p2s[s].to_dict()
    # Create a Pyomo model
    model = pyo.ConcreteModel()

    # Define sets
    model.T = pyo.Set(initialize=T)
    model.T1 = pyo.Set(initialize=T1)
    model.T2 = pyo.Set(initialize=T2)

    # Variables
    model.P_EV_1 = pyo.Var(T1, domain=pyo.NonNegativeReals) # EV charging power stage 1 (kW)
    model.P_EV_2 = pyo.Var(T2, domain=pyo.NonNegativeReals) # EV charging power stage 2 (kW)

    # Objective function
    def Objective_rule(model):
        # First-stage cost
        first_stage_cost = sum(
            p1[t] * (model.P_EV_1[t] + P_house[t])
            for t in model.T1
        )
        # Expected second-stage cost
        second_stage_cost = sum(
            p2[t] * (model.P_EV_2[t] + P_house[t])
            for t in model.T2
        )
        return first_stage_cost + second_stage_cost

    model.obj = pyo.Objective(rule = Objective_rule, sense=pyo.minimize)

    # Constraints

    # EV Requirement Constraint
    def EV_Requirement_rule(model):
        first_stage_energy = sum(model.P_EV_1[t] for t in model.T1)
        second_stage_energy = sum(model.P_EV_2[t] for t in model.T2)
        return first_stage_energy + second_stage_energy == E_EV_required
    
    model.EV_Requirement = pyo.Constraint(rule=EV_Requirement_rule)

    # Maximum EV Charging Power Constraint
    def Max_EV_Power_FirstStage_rule(model, t):
        return model.P_EV_1[t] <= P_EV_max

    model.Max_EV_Power_FirstStage = pyo.Constraint(model.T1, rule=Max_EV_Power_FirstStage_rule)

    def Max_EV_Power_SecondStage_rule(model, t):
     return model.P_EV_2[t] <= P_EV_max

    model.Max_EV_Power_SecondStage = pyo.Constraint(model.T2, rule=Max_EV_Power_SecondStage_rule)
    
    # Solve the model
    opt = pyo.SolverFactory('gurobi')
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions=True)
    print(f'Scenario {s}')
    print(model.display(), model.dual.display())

for s in S:
    Problem(s)
