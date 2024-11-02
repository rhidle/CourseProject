import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

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

# Create a Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.T1 = pyo.Set(initialize=T1)  # First-stage time periods
model.T2 = pyo.Set(initialize=T2)  # Second-stage time periods
model.T = pyo.Set(initialize=T)    # All time periods
model.S = pyo.Set(initialize=S)    # Scenarios

# First-stage variables: EV charging power for t in T1 and scenarios s
model.P_EV_1 = pyo.Var(model.T1, domain=pyo.NonNegativeReals)

# Second-stage variables: EV charging power for t in T2 and scenarios s
model.P_EV_2 = pyo.Var(model.S, model.T2, domain=pyo.NonNegativeReals)

def Objective_rule(model):
    # First-stage cost
    first_stage_cost = sum(
            p1[t] * (model.P_EV_1[t] + P_house[t])
            for t in model.T1
    )
    # Expected second-stage cost
    second_stage_cost = sum(
        pi[s] * sum(
            p2s[s][t] * (model.P_EV_2[s, t] + P_house[t])
            for t in model.T2
        )
        for s in model.S
    )
    
    return first_stage_cost + second_stage_cost

model.obj = pyo.Objective(rule=Objective_rule, sense=pyo.minimize)

def EV_Requirement_rule(model,s):
    first_stage_energy = sum(model.P_EV_1[t] for t in model.T1)
    second_stage_energy = sum(model.P_EV_2[s,t] for t in model.T2)
    
    return first_stage_energy + second_stage_energy == E_EV_required

model.EV_Requirement = pyo.Constraint(model.S,rule=EV_Requirement_rule)

def Max_EV_Power_FirstStage_rule(model, t):
    return model.P_EV_1[t] <= P_EV_max

model.Max_EV_Power_FirstStage = pyo.Constraint(model.T1, rule=Max_EV_Power_FirstStage_rule)

def Max_EV_Power_SecondStage_rule(model, s, t):
    return model.P_EV_2[s, t] <= P_EV_max

model.Max_EV_Power_SecondStage = pyo.Constraint(model.S, model.T2, rule=Max_EV_Power_SecondStage_rule)

# Solve the model
opt = pyo.SolverFactory('gurobi')
results = opt.solve(model, tee=True)

# Display results
for t in model.T1:
    print(f"Hour {t}: P_EV = {pyo.value(model.P_EV_1[t])} kW")


for s in model.S:
    print(f"\nScenario {s}:")
    for t in model.T2:
        print(f"  Hour {t}: P_EV = {pyo.value(model.P_EV_2[s, t])} kW")

hour = pd.read_csv('Prices_June.csv')['Date'][:48]
hour = pd.to_datetime(hour)
hour = hour.dt.hour.to_list()

Charge = [pyo.value(model.P_EV_1[t]) for t in model.T1]

# Create figure and plot on the first y-axis
fig, ax1 = plt.subplots()
ax1.bar(range(24), Charge[:24], color='g', label='Y1 Data')  # 'g-' sets the line color to green
ax1.set_xlabel('Hour')
ax1.set_ylabel('EV Charging Power (KWh)', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()
p1_cent = [value * 100 for value in p1.values()]
ax2.plot(range(24), p1_cent, 'b-', label='Y2 Data')  # 'b-' sets the line color to blue
ax2.set_ylabel('¢/KWh', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()  # Ensures the layout fits well
plt.show()

for s in model.S:
    Charge = [pyo.value(model.P_EV_2[s, t]) for t in model.T2]
    fig, ax1 = plt.subplots()
    ax1.bar(range(24), Charge, color='g', label='Y1 Data')  # 'g-' sets the line color to green
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('EV Charging Power (KWh)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Create the second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    p2_cent = [value * 100 for value in p2s[s].values()]
    ax2.plot(range(24), p2_cent, 'b-', label='Y2 Data')  # 'b-' sets the line color to blue
    ax2.set_ylabel('¢/KWh', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.tight_layout()  # Ensures the layout fits well
    plt.show()