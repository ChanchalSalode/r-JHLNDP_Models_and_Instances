import numpy as np
import pandas as pd
import math, time, random
from gurobipy import Model, GRB, quicksum, LinExpr
from contextlib import redirect_stdout
import os

n = 10
p = 4
alpha_km = 0.2
l1 = 2
Junctions = 1
M = l1 - 1

N = list(range(1, n+1))
L = list(range(1, l1+1))
D_set = [(i, j) for i in N for j in N if i != j]

df_w = pd.read_excel('AP200.xlsx', sheet_name='A')
w = {(int(a), int(b)): float(w_val) for a, b, w_val in df_w.values}
df_c = pd.read_excel('AP200.xlsx', sheet_name='B')
c_param = {(int(a), int(b)): float(c_val) for a, b, c_val in df_c.values}

storedOptimalCuts = []     # for optimality cuts
storedFeasibilityCuts = [] # for feasibility cuts

iteration_count = 0
optimal_cut_count = 0
feasibility_cut_count = 0

a_vars = {}
y_vars = {}
z_vars = {}
I = {}

global_instance_time_limit = 86400  

def set_model_time_limit(model):
    remaining = global_instance_time_limit - (time.time() - instance_start_time)
    if remaining < 1:
        remaining = 1
    model.setParam('TimeLimit', remaining)

# Section 1: Phase 1 – Warm-Start LP Relaxation
def setupMasterProblemModelPhase1():
    master = Model("master_phase1")
    master.setParam('OutputFlag', 0)
    
    global a_vars, y_vars, z_vars, I
    a_vars = {(i, k): master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"a_{i}_{k}")
              for i in N for k in N}
    y_vars = {(k, m, l): master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{k}_{m}_{l}")
              for k in N for m in N for l in L}
    z_vars = {(k, l): master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"z_{k}_{l}")
              for k in N for l in L}
    eta_vars = {(i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
                for (i, j) in D_set}
    I = master.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="I")
    master.update()

    # constraints
    for i in N:
        master.addConstr(quicksum(a_vars[(i, k)] for k in N) == 1, name=f"assign_{i}")
    for i in N:
        for k in N:
            master.addConstr(a_vars[(i, k)] <= a_vars[(k, k)], name=f"allocation_{i}_{k}")
    for k in N:
        for l in L:
            master.addConstr(z_vars[(k, l)] <= a_vars[(k, k)], name=f"assi_{k}_{l}")
    for k in N:
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) >= a_vars[(k, k)], name=f"as_{k}")
    for l in L:
        master.addConstr(quicksum(z_vars[(k, l)] for k in N) == p, name=f"hub_count_{l}")
    for l in L:
        master.addConstr(quicksum(y_vars[(k, m, l)] for k in N for m in N if k < m) == p - 1, name=f"y_count_{l}")
    for k in N:
        for l in L:
            expr = quicksum(y_vars[(k, m, l)] for m in N if k < m) + \
                   quicksum(y_vars[(m, k, l)] for m in N if k > m)
            master.addConstr(expr <= 2 * z_vars[(k, l)], name=f"linking_{k}_{l}")
    for k in N:
        for m in N:
            expr1 = quicksum(y_vars[(k, m, l)] for l in L)
            master.addConstr(expr1 <= 1, name=f"linking1_{k}_{m}")
    for k in N:
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) >= 2 * I[k], name=f"Inter_lb_{k}")
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) <= 1 + M * I[k], name=f"Inter_ub_{k}")
    master.addConstr(quicksum(I[k] for k in N) == Junctions, name="Junctions")
    master.update()

    # Objective
    obj = quicksum(w[(i, j)] * c_param[(i, k)] * a_vars[(i, k)]
                   for i in N for j in N if i != j for k in N)
    obj += quicksum(w[(i, j)] * c_param[(k, i)] * a_vars[(i, k)]
                    for i in N for j in N if i != j for k in N)
    obj += quicksum(eta_vars[(i, j)] for (i, j) in D_set)
    master.setObjective(obj, GRB.MINIMIZE)
    master.update()
    return master, eta_vars

def setMagnantiWongInitialPoint():
    for k in N:
        a_vars[(k, k)].start = 0.5
    for i in N:
        for k in N:
            if i != k:
                a_vars[(i, k)].start = 1.0 / (2*n - 2)
    val_y = (n - 2) / (n**2 - n)
    for (k, m, l) in y_vars:
        y_vars[(k, m, l)].start = val_y

def subProblem_ij(i, j, fixed_a, fixed_y):
    sub = Model(f"dual_subproblem_{i}_{j}")
    sub.setParam('OutputFlag', 0)
    sub.setParam('DualReductions', 0)
    sub.setParam('Method', 0)
    
    a_dual = {k: sub.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"a_{i}_{k}") for k in N}
    delta_vars = {}
    for k in N:
        for m in N:
            if k < m:
                for l in L:
                    delta_vars[(k, m, l)] = sub.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"delta_{k}_{m}_{l}")
    sub.update()

    obj1 = quicksum((fixed_a[(i, k)] - fixed_a[(j, k)]) * a_dual[k] for k in N)
    obj2 = -quicksum(fixed_y[(k, m, l)] * delta_vars[(k, m, l)]
                     for k in N for m in N if k < m for l in L)
    sub.setObjective(obj1 + obj2, GRB.MAXIMIZE)

    for k in N:
        for m in N:
            if k != m:
                for l in L:
                    rhs = alpha_km * w[(i, j)] * c_param[(k, m)]
                    if k < m:
                        sub.addConstr(a_dual[m] - a_dual[k] - delta_vars[(k, m, l)] <= rhs,
                                      name=f"constr_{k}_{m}_{l}_direct")
                    else:
                        if (m, k, l) in delta_vars:
                            sub.addConstr(a_dual[m] - a_dual[k] - delta_vars[(m, k, l)] <= rhs,
                                          name=f"constr_{k}_{m}_{l}_swapped")
    sub.update()
    set_model_time_limit(sub)
    sub.optimize()
    status = sub.status

    if status == GRB.OPTIMAL:
        mu = {k: a_dual[k].X for k in N}
        nu = {key: delta_vars[key].X for key in delta_vars}
        return mu, nu, status, a_dual, delta_vars
    elif status in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
        ray_a = sub.getAttr("UnbdRay", list(a_dual.values()))
        ray_delta = sub.getAttr("UnbdRay", list(delta_vars.values()))
        mu = dict(zip(a_dual.keys(), ray_a))
        nu = dict(zip(delta_vars.keys(), ray_delta))
        return mu, nu, status, a_dual, delta_vars
    else:
        return None, None, status, None, None

def addOptimalityCut_ij(master, i, j, mu, nu, a_vars, y_vars, z_vars, I, eta_vars):
    expr = LinExpr()
    for k in N:
        expr += mu[k] * (a_vars[(i, k)] - a_vars[(j, k)])
    for k in N:
        for m in N:
            if k < m:
                for l in L:
                    expr -= nu.get((k, m, l), 0) * y_vars[(k, m, l)]
    master.addConstr(eta_vars[(i, j)] >= expr, name=f"OptCut_{i}_{j}")
    storedOptimalCuts.append(('optimal', i, j, mu.copy(), nu.copy(), 0))
    master.update()

def addFeasibilityCut_ij(master, i, j, mu, nu, a_vars, y_vars, z_vars, I):
    expr = LinExpr()
    for k in N:
        expr += mu[k] * (a_vars[(i, k)] - a_vars[(j, k)])
    for k in N:
        for m in N:
            if k < m:
                for l in L:
                    expr -= nu.get((k, m, l), 0) * y_vars[(k, m, l)]
    master.addConstr(expr <= 0, name=f"FeasCut_{i}_{j}")
    storedFeasibilityCuts.append(('feasibility', i, j, mu.copy(), nu.copy(), 0))
    master.update()

def solveOptimalLambda_for_pair(i, j, a0, y0, ah, yh):
    mdl = Model(f"OptimalLambda_{i}_{j}")
    mdl.setParam('OutputFlag', 0)
    mdl.setParam('Method', 0)

    lam = mdl.addVar(lb=0, ub=0.5, vtype=GRB.CONTINUOUS, name=f"lambda_{i}_{j}")

    x_vars = {}
    for k in N:
        for m in N:
            if k != m:
                for l in L:
                    x_vars[(k, m, l)] = mdl.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                   name=f"x_{i}_{j}_{k}_{m}_{l}")
    mdl.update()

    for k in N:
        lhs_expr = LinExpr()
        for m in N:
            if k != m:
                for l in L:
                    lhs_expr += x_vars[(k, m, l)]
                    lhs_expr -= x_vars[(m, k, l)]
        rhs_val = (1 - lam) * (a0[(i, k)] - a0[(j, k)]) + lam * (ah[(i, k)] - ah[(j, k)])
        mdl.addConstr(lhs_expr == rhs_val, name=f"bal_{i}_{j}_{k}")
    for k in N:
        for m in N:
            if k < m:
                for l in L:
                    lhs_expr = x_vars[(k, m, l)] + x_vars[(m, k, l)]
                    rhs_expr = (1 - lam) * y0[(k, m, l)] + lam * yh[(k, m, l)]
                    mdl.addConstr(lhs_expr <= rhs_expr, name=f"link_{i}_{j}_{k}_{m}_{l}")
    mdl.setObjective(lam, GRB.MAXIMIZE)
    mdl.update()
    set_model_time_limit(mdl)
    mdl.optimize()
    if mdl.status == GRB.OPTIMAL:
        return lam.X
    else:
        return 0.0

def computeOptimalLambdaSystem(a0, y0, ah, yh):
    lambda_values = []
    for (i, j) in D_set:
        lam_val = solveOptimalLambda_for_pair(i, j, a0, y0, ah, yh)
        lambda_values.append(lam_val)
    overall_lambda = min(lambda_values) if lambda_values else 0.0
    print(f"Overall optimal lambda: {overall_lambda:.4f}")
    return overall_lambda

def runWarmStartPhase():
    global storedOptimalCuts, storedFeasibilityCuts, a_vars, y_vars, z_vars, I
    storedOptimalCuts.clear()
    storedFeasibilityCuts.clear()

    master, eta_vars = setupMasterProblemModelPhase1()
    setMagnantiWongInitialPoint()

    set_model_time_limit(master)
    master.optimize()
    if master.status == GRB.TIME_LIMIT:
        print("Time limit reached in Phase 1 initial solve.")
    if master.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("Master not optimal at start. Status =", master.status)
        return master, eta_vars, 0
    LB = master.ObjVal
    UB = float('inf')
    tolerance = 1e-4
    iteration = 0
    max_iter = 5000

    a0 = {(i, k): a_vars[(i, k)].X for (i, k) in a_vars}
    y0 = {(k, m, l): y_vars[(k, m, l)].X for (k, m, l) in y_vars}

    start_phase = time.time()
    while (UB - LB > tolerance) and (iteration < max_iter):
        iteration += 1
        print(f"\n=== WarmStart Iteration {iteration} ===")
        for (i, j) in D_set:
            mu1, nu1, status1, _, _ = subProblem_ij(i, j, a0, y0)
            if status1 == GRB.OPTIMAL:
                addOptimalityCut_ij(master, i, j, mu1, nu1, a_vars, y_vars, z_vars, I, eta_vars)
            elif status1 in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                addFeasibilityCut_ij(master, i, j, mu1, nu1, a_vars, y_vars, z_vars, I)
            else:
                print(f"SP#1 for ({i},{j}) status: {status1} - no cut added.")
        set_model_time_limit(master)
        master.optimize()
        if master.status == GRB.TIME_LIMIT:
            print("Time limit reached during Phase 1 second solve.")
            break
        LB = master.ObjVal
        aHat = {(i, k): a_vars[(i, k)].X for (i, k) in a_vars}
        yHat = {(k, m, l): y_vars[(k, m, l)].X for (k, m, l) in y_vars}
        unbounded_flag = False
        for (i, j) in D_set:
            mu2, nu2, status2, _, _ = subProblem_ij(i, j, aHat, yHat)
            if status2 == GRB.OPTIMAL:
                addOptimalityCut_ij(master, i, j, mu2, nu2, a_vars, y_vars, z_vars, I, eta_vars)
            elif status2 in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                addFeasibilityCut_ij(master, i, j, mu2, nu2, a_vars, y_vars, z_vars, I)
                unbounded_flag = True
            else:
                print(f"SP#2 for ({i},{j}) status: {status2} - no cut added.")
        if unbounded_flag:
            lam_star = computeOptimalLambdaSystem(a0, y0, aHat, yHat)
            print("Optimal lambda from system:", lam_star)
            a0 = {(i, k): (1 - lam_star)*a0[(i, k)] + lam_star*aHat[(i, k)] for (i, k) in a0}
            y0 = {(k, m, l): (1 - lam_star)*y0[(k, m, l)] + lam_star*yHat[(k, m, l)] for (k, m, l) in y0}
        else:
            UB = min(UB, LB)
            a0 = aHat
            y0 = yHat
        gap = UB - LB
        print(f"  LB = {LB:.4f}, UB = {UB:.4f}, Gap = {gap:.4f}")
        if gap < tolerance:
            print("Warm-start converged by gap.")
            break
    elapsed = time.time() - start_phase
    print("\nWarm-Start Phase complete.")
    print("Iterations =", iteration, f"Time = {elapsed:.2f} sec")
    print(f"LB = {LB:.4f}, UB = {UB:.4f}, Gap = {UB - LB:.4f}")
    print("Total optimality cuts stored in Phase 1:", len(storedOptimalCuts))
    print("Total feasibility cuts stored in Phase 1:", len(storedFeasibilityCuts))
    master.write("phase1_warmstart.lp")
    with open("phase1_warmstart.lp", "r") as f:
        content = f.read()
    with open("phase1_warmstart.txt", "w") as f:
        f.write(content)
    return master, eta_vars, iteration

# Section 2: Phase 2 – MILP Master Problem and Callback
def setupMasterProblemModelPhase2():
    master = Model("master_phase2")
    master.setParam('LazyConstraints', 1)
    global a_vars, y_vars, z_vars, I
    a_vars = {(i, k): master.addVar(vtype=GRB.BINARY, name=f"a_{i}_{k}") for i in N for k in N}
    y_vars = {(k, m, l): master.addVar(vtype=GRB.BINARY, name=f"y_{k}_{m}_{l}") for k in N for m in N for l in L}
    z_vars = {(k, l): master.addVar(vtype=GRB.BINARY, name=f"z_{k}_{l}") for k in N for l in L}
    eta_vars = {(i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}") for (i, j) in D_set}
    I = master.addVars(N, vtype=GRB.BINARY, name="I")
    master.update()
    for i in N:
        master.addConstr(quicksum(a_vars[(i, k)] for k in N) == 1, name=f"assign_{i}")
    for i in N:
        for k in N:
            master.addConstr(a_vars[(i, k)] <= a_vars[(k, k)], name=f"allocation_{i}_{k}")
    for k in N:
        for l in L:
            master.addConstr(z_vars[(k, l)] <= a_vars[(k, k)], name=f"assi_{k}_{l}")
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) >= a_vars[(k, k)], name=f"as_{k}")
    for l in L:
        master.addConstr(quicksum(z_vars[(k, l)] for k in N) == p, name=f"hub_count_{l}")
        master.addConstr(quicksum(y_vars[(k, m, l)] for k in N for m in N if k < m) == p - 1, name=f"y_count_{l}")
    for k in N:
        for l in L:
            expr = quicksum(y_vars[(k, m, l)] for m in N if k < m) + quicksum(y_vars[(m, k, l)] for m in N if k > m)
            master.addConstr(expr <= 2 * z_vars[(k, l)], name=f"linking_{k}_{l}")
    for k in N:
        for m in N:
            expr1 = quicksum(y_vars[(k, m, l)] for l in L)
            master.addConstr(expr1 <= 1, name=f"linking1_{k}_{m}")
    for k in N:
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) >= 2 * I[k], name=f"Inter_lb_{k}")
        master.addConstr(quicksum(z_vars[(k, l)] for l in L) <= 1 + M * I[k], name=f"Inter_ub_{k}")
    master.addConstr(quicksum(I[k] for k in N) == Junctions, name="Junctions")
    master.update()

    # Objective
    obj = quicksum(w[(i, j)] * c_param[(i, k)] * a_vars[(i, k)]
                   for i in N for j in N if i != j for k in N)
    obj += quicksum(w[(i, j)] * c_param[(k, i)] * a_vars[(i, k)]
                    for i in N for j in N if i != j for k in N)
    obj += quicksum(eta_vars[(i, j)] for (i, j) in D_set)
    master.setObjective(obj, GRB.MINIMIZE)
    master.update()
    return master, eta_vars

def addStoredCutsToModel(master, eta_vars):
    """
    Incorporates the stored optimality and feasibility cuts from Phase 1 into the MILP.
    """
    for idx, cut in enumerate(storedOptimalCuts):
        _, i, j, mu_dict, nu_dict, _ = cut
        expr = LinExpr()
        for k in N:
            expr += mu_dict[k] * (a_vars[(i, k)] - a_vars[(j, k)])
        for k in N:
            for m in N:
                if k < m:
                    for l in L:
                        expr -= nu_dict.get((k, m, l), 0) * y_vars[(k, m, l)]
        master.addConstr(eta_vars[(i, j)] >= expr, name=f"StoredOptCut_{idx}_{i}_{j}")
    for idx, cut in enumerate(storedFeasibilityCuts):
        _, i, j, mu_dict, nu_dict, _ = cut
        expr = LinExpr()
        for k in N:
            expr += mu_dict[k] * (a_vars[(i, k)] - a_vars[(j, k)])
        for k in N:
            for m in N:
                if k < m:
                    for l in L:
                        expr -= nu_dict.get((k, m, l), 0) * y_vars[(k, m, l)]
        master.addConstr(expr <= 0, name=f"StoredFeasCut_{idx}_{i}_{j}")
    master.update()

def callBackFunction(model, where):
    global iteration_count, optimal_cut_count, feasibility_cut_count
    if where == GRB.Callback.MIPSOL:
        iteration_count += 1
        print(f"\n--- Phase 2 iteration {iteration_count} ---")
        aHat = {(i, k): model.cbGetSolution(model.getVarByName(f"a_{i}_{k}"))
                for i in N for k in N}
        yHat = {(k, m, l): model.cbGetSolution(model.getVarByName(f"y_{k}_{m}_{l}"))
                for (k, m, l) in y_vars.keys()}
        for (i, j) in D_set:
            mu, nu, status, _, _ = subProblem_ij(i, j, aHat, yHat)
            if status == GRB.OPTIMAL:
                expr = LinExpr()
                for k in N:
                    expr += mu[k] * (model.getVarByName(f"a_{i}_{k}") - model.getVarByName(f"a_{j}_{k}"))
                for k in N:
                    for m in N:
                        if k < m:
                            for l in L:
                                expr -= nu.get((k, m, l), 0) * model.getVarByName(f"y_{k}_{m}_{l}")
                model.cbLazy(model.getVarByName(f"eta_{i}_{j}") >= expr)
                optimal_cut_count += 1
            elif status in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                expr = LinExpr()
                for k in N:
                    expr += mu[k] * (model.getVarByName(f"a_{i}_{k}") - model.getVarByName(f"a_{j}_{k}"))
                for k in N:
                    for m in N:
                        if k < m:
                            for l in L:
                                expr -= nu.get((k, m, l), 0) * model.getVarByName(f"y_{k}_{m}_{l}")
                model.cbLazy(expr <= 0)
                feasibility_cut_count += 1

def runPhase2Benders():
    master, eta_vars = setupMasterProblemModelPhase2()
    addStoredCutsToModel(master, eta_vars)
    print("Starting Phase 2 MILP with callback...")
    set_model_time_limit(master)
    start_phase2 = time.time()
    master.optimize(callback=callBackFunction)
    phase2_time = time.time() - start_phase2
    final_obj = master.ObjVal
    print("\nPhase 2 Final Objective:", final_obj)
    print(f"Phase 2 done in {phase2_time:.2f} seconds with {optimal_cut_count} opt-cuts,"
          f" {feasibility_cut_count} feas-cuts.")
    #return master, phase2_time, final_obj
        # Collect final decision variable values
    dec2 = [f"{v.VarName}: {v.X}" for v in master.getVars()]
    return master, phase2_time, final_obj, dec2

# Section 3: Single-Instance Driver Function
def runTwoPhaseBenders():
    global instance_start_time, iteration_count, optimal_cut_count, feasibility_cut_count
    iteration_count = optimal_cut_count = feasibility_cut_count = 0
    instance_start_time = time.time()

    print("Running Phase 1 (Warm-Start)…")
    master1, eta1, phase1_iter = runWarmStartPhase()

    print("\nRunning Phase 2 (Benders Branch-and-Cut)…")
    master2, phase2_time, final_obj, dec2 = runPhase2Benders()

    print(f"\n=== Completed ===\nFinal objective: {final_obj:.4f}\n"
          f"Phase 1 iterations: {phase1_iter}\n"
          f"Phase 2 time: {phase2_time:.2f} sec")

# Execute
if __name__ == '__main__':
    runTwoPhaseBenders()