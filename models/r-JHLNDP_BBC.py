import numpy as np
import pandas as pd
import math, time, random
from gurobipy import Model, GRB, quicksum, LinExpr

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

iteration_count = 0
optimal_count = 0
unbounded_count = 0

def setInitialSolution(a_vars, y_vars, z_vars, I, p, N, L):
    hubs = sorted(random.sample(N, p))
    print("Initial hubs chosen:", hubs)
    for i in N:
        for k in N:
            a_vars[(i, k)].start = 0
        if i in hubs:
            a_vars[(i, i)].start = 1
        else:
            a_vars[(i, hubs[0])].start = 1
    for key in y_vars:
        y_vars[key].start = 0
    
    l0 = L[0]
    for i in range(len(hubs) - 1):
        k = hubs[i]
        m = hubs[i+1]
        y_vars[(k, m, l0)].start = 1

def setupMasterProblemModel():
    master = Model("master")
    master.setParam('LazyConstraints', 1)
    master.setParam('OutputFlag', 1)
    
    global a_vars, y_vars, z_vars, I
    a_vars = {(i, k): master.addVar(vtype=GRB.BINARY, name=f"a_{i}_{k}") for i in N for k in N}
    y_vars = {(k, m, l): master.addVar(vtype=GRB.BINARY, name=f"y_{k}_{m}_{l}") 
              for k in N for m in N for l in L}
    z_vars = {(k, l): master.addVar(vtype=GRB.BINARY, name=f"z_{k}_{l}") for k in N for l in L}
    eta_vars = {(i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
                for (i, j) in D_set}
    I = master.addVars(N, vtype=GRB.BINARY, name="I")
    master.update()
    
    obj = quicksum(w[(i, j)] * c_param[(i, k)] * a_vars[(i, k)]
                   for i in N for j in N if i != j for k in N)
    obj += quicksum(w[(i, j)] * c_param[(k, i)] * a_vars[(i, k)]
                    for i in N for j in N if i != j for k in N)
    obj += quicksum(eta_vars[(i, j)] for (i, j) in D_set)
    master.setObjective(obj, GRB.MINIMIZE)
    
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
    
    master.Params.lazyConstraints = 1
    print("Master problem built.")
    return master, eta_vars

def subProblem_ij(i, j, fixed_a, fixed_y):
    sub = Model(f"dual_subproblem_{i}_{j}")
    sub.setParam('OutputFlag', 0)
    sub.setParam('DualReductions', 0)
    sub.setParam('Method', 0)
    
    a_dual = { k: sub.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"a_{i}_{j}_{k}") for k in N }
    delta_vars = {}
    for k in N:
        for m in N:
            if k < m:
                for l in L:
                    delta_vars[(k, m, l)] = sub.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"delta_{i}_{j}_{k}_{m}_{l}")
    sub.update()
    
    obj = quicksum((fixed_a[(i, k)] - fixed_a[(j, k)]) * a_dual[k] for k in N)
    obj -= quicksum(fixed_y[(k, m, l)] * delta_vars[(k, m, l)]
                    for k in N for m in N if k < m for l in L)
    sub.setObjective(obj, GRB.MAXIMIZE)
    
    for k in N:
        for m in N:
            if k != m:
                for l in L:
                    rhs = alpha_km * w[(i, j)] * c_param[(k, m)]
                    if k < m:
                        sub.addConstr(a_dual[m] - a_dual[k] - delta_vars[(k, m, l)] <= rhs,
                                      name=f"dualconstr_{i}_{j}_{k}_{m}_{l}")
                    else:
                        sub.addConstr(a_dual[m] - a_dual[k] - delta_vars[(m, k, l)] <= rhs,
                                      name=f"dualconstr_{i}_{j}_{k}_{m}_{l}")
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
    ret = (mu, nu, status)
    sub.dispose()
    return ret

def callBackFunction(model, where):
    global N, a_vars, y_vars, z_vars, I, D_set, iteration_count, optimal_count, unbounded_count
    if where == GRB.Callback.MIPSOL:
        iteration_count += 1
        print(f"\n--- Callback Iteration {iteration_count} ---")
        aHat = {(i, k): model.cbGetSolution(model.getVarByName(f"a_{i}_{k}"))
                for i in N for k in N}
        yHat = {(k, m, l): model.cbGetSolution(model.getVarByName(f"y_{k}_{m}_{l}"))
                for (k, m, l) in y_vars.keys()}
        for (i, j) in D_set:
            result = subProblem_ij(i, j, aHat, yHat)
            if result is None:
                continue
            mu, nu, status = result
            if status == GRB.OPTIMAL:
                optimal_count += 1
                expr = LinExpr()
                for k in N:
                    expr += mu[k] * (model.getVarByName(f"a_{i}_{k}") - model.getVarByName(f"a_{j}_{k}"))
                for k in N:
                    for m in N:
                        if k < m:
                            for l in L:
                                expr -= nu.get((k, m, l), 0) * model.getVarByName(f"y_{k}_{m}_{l}")
                model.cbLazy(model.getVarByName(f"eta_{i}_{j}") >= expr)
            elif status in [GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                unbounded_count += 1
                expr = LinExpr()
                for k in N:
                    expr += mu[k] * (model.getVarByName(f"a_{i}_{k}") - model.getVarByName(f"a_{j}_{k}"))
                for k in N:
                    for m in N:
                        if k < m:
                            for l in L:
                                expr -= nu.get((k, m, l), 0) * model.getVarByName(f"y_{k}_{m}_{l}")
                model.cbLazy(expr <= 0)
            else:
                print(f"Callback: (i,j)=({i},{j}) subproblem status {status}; no cut added.")

def runCallBackBenders():
    global iteration_count, optimal_count, unbounded_count, D_set
    master, eta_vars = setupMasterProblemModel()
    setInitialSolution(a_vars, y_vars, z_vars, I, p, N, L)
    
    print("Starting master problem solve with callback...")
    start_time = time.time()
    master.optimize(callback=callBackFunction)
    total_time = time.time() - start_time
    
    final_obj = master.ObjVal
    print("\nFinal master objective:", final_obj)
    for i in N:
        for k in N:
            if a_vars[(i, k)].X == 1:
                print(f"a[{i},{k}] = 1")
    for k in N:
        for l in L:
            if z_vars[(k, l)].X == 1:
                print(f"z[{k},{l}] = 1")
    for (k, m, l) in y_vars:
        if y_vars[(k, m, l)].X == 1:
            print(f"y[{k},{m},{l}] = 1")
    for k in N:
        if I[k].X == 1:
            print(f"I[{k}] = 1")
    for (i, j) in eta_vars:
        val = master.getVarByName(f"eta_{i}_{j}").X
        if val > 1e-6:
            print(f"eta[{i},{j}] = {val}")
    print("\nTotal callback iterations:", iteration_count)
    print("Optimal subproblems solved:", optimal_count)
    print("Unbounded subproblems solved:", unbounded_count)
    print("Total time taken: {:.2f} seconds".format(total_time))
    master.write("final_master.lp")
    return master

if __name__ == '__main__':
    D_set = [(i, j) for i in N for j in N if i != j]
    runCallBackBenders()