import numpy as np
import pandas as pd
import math, time, random
from gurobipy import Model, GRB, quicksum, LinExpr

n = 10            
p = 3             
alpha_km = 0.2    
T = 2             # Total number of hub lines (i.e., sequential phases)
Junctions = 1     
M = T - 1         

SCRIPT_START = time.time()
MAX_TIME = 2*3600  # 4 hours

data_file = 'AP200.xlsx'

df_w = pd.read_excel(data_file, sheet_name='A')
w_full = {(int(a), int(b)): float(val) for a, b, val in df_w.values}
df_c = pd.read_excel(data_file, sheet_name='B')
c_param = {(int(a), int(b)): float(val) for a, b, val in df_c.values}

N = list(range(1, n+1))
D_set = [(i, j) for i in N for j in N if i != j]

iteration_count = 0
optimal_count = 0
unbounded_count = 0
CURRENT_PHASE = None

def getSets(phase):
    L_phase = list(range(1, phase+1))
    A_set = [(i, j, k, m, l)
             for i in N for j in N for k in N for m in N for l in L_phase]
    D_alloc = [(i, k) for i in N for k in N]
    C_set = [(k, l) for k in N for l in L_phase]
    C1_set = [(k, m, l) for k in N for m in N for l in L_phase]
    return L_phase, A_set, D_alloc, C_set, C1_set

fixed_z = {}       
fixed_y = {}       
fixed_a_diag = {}

def subProblem_ij(i, j, fixed_a, fixed_y):
    if time.time() - SCRIPT_START > MAX_TIME:
        return None, None, GRB.INTERRUPTED

    sub = Model(f"dual_subproblem_{i}_{j}")
    sub.setParam('OutputFlag', 0)
    sub.setParam('DualReductions', 0)
    sub.setParam('Method', 0)

    a_dual = {
        k: sub.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"a_{i}_{j}_{k}")
        for k in N
    }
    delta_vars = {}
    for k in N:
        for m in N:
            if k < m:
                for l in range(1, T+1):
                    delta_vars[(k, m, l)] = sub.addVar(
                        lb=0, vtype=GRB.CONTINUOUS, name=f"delta_{i}_{j}_{k}_{m}_{l}"
                    )
    sub.update()

    obj = quicksum((fixed_a[(i, k)] - fixed_a[(j, k)]) * a_dual[k] for k in N)
    obj -= quicksum(
        fixed_y.get((k, m, l), 0) * delta_vars[(k, m, l)]
        for k in N for m in N if k < m for l in range(1, T+1)
    )
    sub.setObjective(obj, GRB.MAXIMIZE)

    for k in N:
        for m in N:
            if k != m:
                for l in range(1, T+1):
                    rhs = alpha_km * w_full[(i, j)] * c_param[(k, m)]
                    if k < m:
                        sub.addConstr(
                            a_dual[m] - a_dual[k] - delta_vars[(k, m, l)] <= rhs,
                            name=f"dualconstr_{i}_{j}_{k}_{m}_{l}"
                        )
                    else:
                        sub.addConstr(
                            a_dual[m] - a_dual[k] - delta_vars[(m, k, l)] <= rhs,
                            name=f"dualconstr_{i}_{j}_{k}_{m}_{l}"
                        )
    sub.update()
    sub.optimize()

    status = sub.status
    if status == GRB.OPTIMAL:
        mu = {k: a_dual[k].X for k in N}
        nu = {key: delta_vars[key].X for key in delta_vars}
    elif status in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
        ray_a = sub.getAttr("UnbdRay", list(a_dual.values()))
        ray_delta = sub.getAttr("UnbdRay", list(delta_vars.values()))
        mu = dict(zip(a_dual.keys(), ray_a))
        nu = dict(zip(delta_vars.keys(), ray_delta))
    else:
        mu, nu = None, None

    sub.dispose()
    return mu, nu, status

def callBackFunction(model, where):
    global iteration_count, optimal_count, unbounded_count, CURRENT_PHASE
    
    if time.time() - SCRIPT_START > MAX_TIME:
        model.terminate()
        return

    if where == GRB.Callback.MIPSOL:
        iteration_count += 1
        print(f"\n--- Callback Iteration {iteration_count} ---")

        aHat = {
            (i, k): model.cbGetSolution(model.getVarByName(f"a_{i}_{k}"))
            for i in N for k in N
        }
        yHat = {
            (k, m, l): model.cbGetSolution(model.getVarByName(f"y_{k}_{m}_{l}"))
            for k in N for m in N for l in range(1, CURRENT_PHASE+1)
        }

        for (i, j) in D_set:
            mu, nu, status = subProblem_ij(i, j, aHat, yHat)
            if mu is None:
                continue

            expr = LinExpr()
            for k in N:
                expr += mu[k] * (
                    model.getVarByName(f"a_{i}_{k}") -
                    model.getVarByName(f"a_{j}_{k}")
                )
            for k in N:
                for m in N:
                    if k < m:
                        for l in range(1, CURRENT_PHASE+1):
                            expr -= nu.get((k, m, l), 0) * model.getVarByName(f"y_{k}_{m}_{l}")

            if status == GRB.OPTIMAL:
                optimal_count += 1
                model.cbLazy(model.getVarByName(f"eta_{i}_{j}") >= expr)
            else:
                unbounded_count += 1
                model.cbLazy(expr <= 0)

def setupMasterProblemModelSequential(phase, fixed_z, fixed_y, fixed_a_diag):
    L_phase, _, _, _, _ = getSets(phase)
    master = Model(f"master_phase_{phase}")
    master.setParam('LazyConstraints', 1)
    
    elapsed = time.time() - SCRIPT_START
    master.setParam('TimeLimit', max(0, MAX_TIME - elapsed))

    a_vars = {
        (i, k): master.addVar(vtype=GRB.BINARY, name=f"a_{i}_{k}")
        for i in N for k in N
    }
    z_vars = {
        (k, l): master.addVar(vtype=GRB.BINARY, name=f"z_{k}_{l}")
        for k in N for l in L_phase
    }
    y_vars = {
        (k, m, l): master.addVar(vtype=GRB.BINARY, name=f"y_{k}_{m}_{l}")
        for k in N for m in N for l in L_phase
    }
    eta_vars = {
        (i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
        for (i, j) in D_set
    }
    I_vars = master.addVars(N, vtype=GRB.BINARY, name="I") if phase >= 2 else None

    master.update()

    obj = quicksum(
        w_full[(i, j)] * c_param[(i, k)] * a_vars[(i, k)]
        for i in N for j in N if i != j for k in N
    )
    obj += quicksum(
        w_full[(i, j)] * c_param[(k, i)] * a_vars[(i, k)]
        for i in N for j in N if i != j for k in N
    )
    obj += quicksum(eta_vars[(i, j)] for (i, j) in D_set)
    master.setObjective(obj, GRB.MINIMIZE)

    for i in N:
        master.addConstr(quicksum(a_vars[(i, k)] for k in N) == 1, name=f"assign_{i}")
    for i in N:
        for k in N:
            master.addConstr(a_vars[(i, k)] <= a_vars[(k, k)], name=f"alloc_{i}_{k}")
    for k in N:
        for l in L_phase:
            master.addConstr(z_vars[(k, l)] <= a_vars[(k, k)], name=f"z_link_{k}_{l}")
        master.addConstr(
            quicksum(z_vars[(k, l)] for l in L_phase) >= a_vars[(k, k)],
            name=f"z_open_{k}"
        )
    for l in L_phase:
        master.addConstr(
            quicksum(z_vars[(k, l)] for k in N) == p,
            name=f"hub_count_{l}"
        )
        master.addConstr(
            quicksum(y_vars[(k, m, l)] for k in N for m in N if k < m) == p - 1,
            name=f"y_count_{l}"
        )
    for k in N:
        for l in L_phase:
            expr = (
                quicksum(y_vars[(k, m, l)] for m in N if k < m) +
                quicksum(y_vars[(m, k, l)] for m in N if m < k)
            )
            master.addConstr(expr <= 2 * z_vars[(k, l)], name=f"linking_{k}_{l}")
    for k in N:
        for m in N:
            master.addConstr(
                quicksum(y_vars[(k, m, l)] for l in L_phase) <= 1,
                name=f"uniqueLink_{k}_{m}"
            )
    if phase >= 2:
        for k in N:
            master.addConstr(
                quicksum(z_vars[(k, l)] for l in L_phase) >= 2 * I_vars[k],
                name=f"Inter_lb_{k}"
            )
            master.addConstr(
                quicksum(z_vars[(k, l)] for l in L_phase) <= 1 + (T - 1) * I_vars[k],
                name=f"Inter_ub_{k}"
            )
        master.addConstr(quicksum(I_vars[k] for k in N) == Junctions, name="Junctions")

    # Fix decisions from earlier phases
    for (k, l0), val in fixed_z.items():
        if l0 < phase:
            master.addConstr(z_vars[(k, l0)] == val, name=f"fix_z_{k}_{l0}")
    for (k, m, l0), val in fixed_y.items():
        if l0 < phase:
            master.addConstr(y_vars[(k, m, l0)] == val, name=f"fix_y_{k}_{m}_{l0}")
    for k in fixed_a_diag:
        master.addConstr(a_vars[(k, k)] == 1, name=f"fix_a_{k}_{k}")

    master.update()
    return master, a_vars, y_vars, z_vars, eta_vars, I_vars

seq_total_time = 0
seq_obj = None

for phase in range(1, T+1):
    if time.time() - SCRIPT_START > MAX_TIME:
        print(">>> GLOBAL TIME LIMIT EXCEEDED: terminating early.")
        break

    print(f"\n*** Solving Sequential Phase {phase} Using Benders Branch-and-Cut ***")
    CURRENT_PHASE = phase
    master, a_vars, y_vars, z_vars, eta_vars, I_vars = setupMasterProblemModelSequential(
        phase, fixed_z, fixed_y, fixed_a_diag
    )
    start_phase = time.time()
    master.optimize(callback=callBackFunction)
    end_phase = time.time()

    phase_time = end_phase - start_phase
    seq_total_time += phase_time

    if master.status == GRB.OPTIMAL:
        seq_obj = master.ObjVal
        print(f"  Phase {phase} solved in {phase_time:.2f} seconds.")
        for (k, l), val in z_vars.items():
            if l == phase and val.X > 0.5:
                fixed_z[(k, l)] = 1
        for (k, m, l), val in y_vars.items():
            if l == phase and val.X > 0.5:
                fixed_y[(k, m, l)] = 1
        for (i, k), val in a_vars.items():
            if i == k and val.X > 0.5:
                fixed_a_diag[k] = 1

        print("\na[i,k] > 0:")
        for (i, k), var in a_vars.items():
            if var.X > 0.5:
                print(f"  a[{i},{k}] = {var.X}")
        print("z[k,l] > 0:")
        for (k, l), var in z_vars.items():
            if var.X > 0.5:
                print(f"  z[{k},{l}] = {var.X}")
        print("y[k,m,l] > 0:")
        for (k, m, l), var in y_vars.items():
            if var.X > 0.5:
                print(f"  y[{k},{m},{l}] = {var.X}")
    else:
        print(f"Phase {phase} did not solve optimally; skipping fixed updates.")
        seq_obj = None

print("\nFinal Sequential Benders Branch-and-Cut Objective:", seq_obj)
print("Total Sequential Benders Time:", seq_total_time)