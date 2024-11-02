import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

T = range(48) # Time periods
T1 = range(24) # Time periods for stage 1
T2 = range(24, 48) # Time periods for stage 2
S = range(1,4) # Scenarios


# Parameters
P_EV_max = 7.0  # Maximum EV charging power (kW)
E_EV_required = 50.0  # Total energy required by the EV (kWh)

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
    return sum(p1[t]*(model.P_EV[t]) for t in T1) + sum(p2[t]*(model.P_EV[t]) for t in T2)
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

def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m
def DisplayResults(m):
    return print(m.display(), m.dual.display())

# Solve the model
Solve(model)
DisplayResults(model)

hour = pd.read_csv('Prices_June.csv')['Date'][:48]
hour = pd.to_datetime(hour)
hour = hour.dt.hour.to_list()

Charge = [model.P_EV[t]() for t in T]
plt.bar(range(24), Charge[:24], color='b', edgecolor='black', linewidth=1)
plt.xticks(range(24), hour[:24])
plt.xlabel('Hour')
plt.ylabel('EV Charging Power (KWh)')
plt.title('Hourly EV Charging Power for Day 1')
plt.show()

plt.bar(range(24), Charge[24:48], color='b', edgecolor='black', linewidth=1)
plt.xticks(range(24), hour[:24])
plt.xlabel('Hour')
plt.ylabel('EV Charging Power (KWh)')
plt.title('Hourly EV Charging Power for Day 2')
plt.show()


fig, ax1 = plt.subplots()
ax1.bar(range(24), Charge[24:48], color='g', label='Y1 Data')  # 'g-' sets the line color to green
ax1.set_xlabel('Hour')
ax1.set_ylabel('EV Charging Power (KWh)', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()
p2_cent = [value * 100 for value in p2.values()]
ax2.plot(range(24), p2_cent, 'b-', label='Y2 Data')  # 'b-' sets the line color to blue
ax2.set_ylabel('¢/KWh', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()  # Ensures the layout fits well
plt.show()