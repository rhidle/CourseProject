import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


T = range(48) # Time periods
T1 = range(24) # Time periods for stage 1
T2 = range(24, 48) # Time periods for stage 2

# Parameters
P_EV_max = 7.0       # Maximum EV charging power (kW)
E_EV_required = 50.0 # Total energy required by the EV (kWh)

# Scenario-specific electricity prices for the first 24 hours ($/kWh)
p1 = pd.read_csv('Prices_June.csv')['Price'][:24] / 1000  # Assuming prices are in $/MWh
p1 = p1.to_dict()

# Scenario-specific electricity prices for the last 24 hours ($/kWh)
p2 = pd.read_csv('Prices_June.csv')['Price'][24:48] / 1000
p2 = p2.to_dict()

# Mathematical formulation 1st stage
def Obj_1st(m):
    return sum(m.p1[t]*m.P_EV[t] for t in m.T1) + m.alpha
def ChargeLimit(m,t):
    return m.P_EV[t] <= m.P_EV_max
def ChargeRequirement1(m):
    return sum(m.P_EV[t] for t in m.T1) <= m.E_EV_req
def CreateCuts(m,c):
    return(m.alpha >= m.Phi[c] + m.Lambda[c]*(sum(m.P_EV[t] for t in m.T1) - m.x_hat[c]))


#Mathematical formulation 2nd stage
def Obj_2nd(m):
    return sum(m.p2[t]*m.P_EV[t] for t in m.T2)
def ChargeRequirement(m):
    return sum(m.P_EV[t] for t in m.T2) + m.X_hat == m.E_EV_req
def ChargeLimit2(m,t):
    return m.P_EV[t] <= m.P_EV_max


# Set up model 1st stage
def ModelSetUp_1st(T1, p1, Cuts):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.T1 = pyo.Set(initialize = T1)  # First-stage time periods
    #Parameters
    m.P_EV_max = pyo.Param(initialize = 7)
    m.E_EV_req = pyo.Param(initialize = 50)
    m.p1 = pyo.Param(m.T1, initialize = p1)

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
def ModelSetUp_2nd(p2,T1,T2,X_hat):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.T2 = pyo.Set(initialize = T2)  # Second-stage time periods
    m.T1 = pyo.Set(initialize = T1)  # First-stage time periods

    # Define parameters
    m.P_EV_max = pyo.Param(initialize = 7)
    m.E_EV_req = pyo.Param(initialize = 50)
    m.X_hat = pyo.Param(initialize = X_hat)
    m.p2 = pyo.Param(m.T2, initialize = p2)
    # Define variables
    m.P_EV = pyo.Var(m.T2, within=pyo.NonNegativeReals)
    
    # Define constraints
    m.ChargeLimit2 = pyo.Constraint(m.T2,rule=ChargeLimit2)
    m.ChargeRequirement = pyo.Constraint(rule=ChargeRequirement)
    
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
    states_jump = 25 # 5 or 25 depending on wether we want 11 or 3 cuts
    List_charged = [i for i in range(Min,Max+states_jump,states_jump)]
    print(f'List_charged: ',List_charged)
    
    """Start the SDP process"""

    # Formulate cut input data
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["Lambda"] = {}
    Cuts["x_hat"] = {}
    for X_hat in List_charged:
        m_2nd = ModelSetUp_2nd(p2, T1, T2, X_hat)
        Solve(m_2nd)

        # Obtain second-stage cost and dual values
        second_stage_cost = pyo.value(m_2nd.obj)
        dual_value = -m_2nd.dual[m_2nd.ChargeRequirement]

        Cuts = Cut_manage(Cuts, dual_value, second_stage_cost, X_hat)

    
    #Solve the 1st stage problem with the acquired cuts
    m_1st = ModelSetUp_1st(T1,p1,Cuts)
    Solve(m_1st)

    
    # Display first-stage results
    FirstStageCharge = {t: pyo.value(m_1st.P_EV[t]) for t in T1}
    FirstStageEnergy = sum(FirstStageCharge[t] for t in T1)
    m_2nd = ModelSetUp_2nd(p2, T1, T2, FirstStageEnergy)
    Solve(m_2nd)

    print("First-Stage Decisions:")
    for t in T1:
        print(f"Hour {t}: P_EV = {pyo.value(m_1st.P_EV[t]):.2f} kW")
    
    print(f"\nTotal Energy Charged in First Stage: {FirstStageEnergy:.2f} kWh")
    print(pyo.value(m_1st.obj))
    print(pyo.value(m_2nd.obj))
    print(pyo.value(m_1st.alpha))
    # print("Hour ; Scenario 1 ; Scenario 2 ; Scenario 3")
    # for t in T2:
    #   print(f"Hour {t}; {pyo.value(m_2nd.P_EV[1,t])} ; {pyo.value(m_2nd.P_EV[2,t])} ; {pyo.value(m_2nd.P_EV[3,t])}")
    print(f"Objective Function Value: {pyo.value(m_1st.obj)-pyo.value(m_1st.alpha.value)+pyo.value(m_2nd.obj)}")

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