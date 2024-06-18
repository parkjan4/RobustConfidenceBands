"""
Author: Jangwon Park
Date: June 18, 2024

This file contains optimization functions including modeling in Gurobi and
the code for the bisection method.
"""

import numpy as np
from gurobipy import *
from helper_functions import calculateCoverage
np.random.seed(3)

#%% Robust optimization
def optimizeCB(samples, alpha=0.05, Gamma=0, ubs=None, lbs=None, warmstart=None, gap=0.01):
    """
    Construct minimum width CB on given samples.
    
    Parameters:
        samples: set of sample paths (e.g., 2D numpy array, list of lists)
        alpha: significance level
        Gamma: the "budget of uncertainty" parameter
        gap: minimum optimality gap to reach before termination in Gurobi.
    """
    R = len(samples)
    H = len(samples[0])
    samples = np.array(samples)
    
    m = Model()

    u = m.addVars(H,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    l = m.addVars(H,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    delta = m.addVars(R,vtype=GRB.BINARY) # 1 if covered; 0 otherwise
    w_u = m.addVars(H,lb=0,ub=GRB.INFINITY)
    w_l = m.addVars(H,lb=0,ub=GRB.INFINITY)
    
    # Objective: minimize width
    m.setObjective(quicksum(u[t] - l[t] for t in range(H)), GRB.MINIMIZE)
    
    # Calculate (1-alpha)- and alpha-quantiles
    ubar = [] # upper quantile estimates
    lbar = [] # lower quantile estimates
    for t in range(H):
        ubar.append(sorted(samples[:,t])[int((1-alpha)*R)+1])
        lbar.append(sorted(samples[:,t])[int(alpha*R)])

    # warm-start (optional)
    if warmstart is None:
        for r in range(R):
            delta[r].start = 0 # 100% coverage CB which is always feasible
    else:
        for r in range(R):
            delta[r].start = warmstart[r]

    #%% Robust constraints: calculation of "beta^u_t" and "beta^l_t" in the paper.
    
    # NOT TO BE CONFUSED:
    # Note that the "beta^u_t" from the paper ie equivalent to:
        # q_hat_u[t] + betas_u[t] - (1-Gamma)*q_hat_u_tstar in the code below.
    # Similarly, "beta^l_t" from the paper is equivalent to:
        # q_hat_l[t] + betas_l[t] - (1-Gamma)*q_hat_l_star in the code below.
    
    # Calculate q^'s 
    q_hat_u = []
    q_hat_l = []
    for t in range(H):
        ub = np.sort(np.array(samples)[:,t])[int(0.995*R)-1]
        lb = np.sort(np.array(samples)[:,t])[int(0.005*R)]
        q_hat_u.append(ub - ubar[t])
        q_hat_l.append(lbar[t] - lb)
        
    # Calculate t*:
    # minus 2 because: (1) python index starts from 0, (2) don't want to include time=0 where every path is the same.
    t_star = min(H,int(np.ceil(Gamma*H))) - 2 
    if t_star < 0:
        t_star = 0
    
    q_hat_u_tstar = sorted(q_hat_u, reverse=True)[t_star] # largest to smallest
    q_hat_l_tstar = sorted(q_hat_l, reverse=True)[t_star]
    
    betas_u = []
    betas_l = []
    for t in range(H):
        betas_u.append(np.maximum(0,q_hat_u_tstar - q_hat_u[t]))
        betas_l.append(np.maximum(0,q_hat_l_tstar - q_hat_l[t]))

    for t in range(H):
        m.addConstr(u[t] >= ubar[t])
        m.addConstr(l[t] <= lbar[t])
        
        Mtu = max(samples[:,t]) - ubar[t]
        Mtl = lbar[t] - min(samples[:,t])
        for r in range(R):
            m.addConstr(u[t] >= samples[r,t] - Mtu*(delta[r]))
            m.addConstr(l[t] <= samples[r,t] + Mtl*(delta[r]))
            
    # Robust constraints
    constant_u = sum([ubar[t] + q_hat_u[t] + betas_u[t] for t in range(H)]) - H*(1-Gamma)*q_hat_u_tstar
    constant_l = sum([lbar[t] - q_hat_l[t] - betas_l[t] for t in range(H)]) + H*(1-Gamma)*q_hat_l_tstar
    m.addConstr(quicksum(u[t] for t in range(H)) >= constant_u)
    m.addConstr(quicksum(l[t] for t in range(H)) <= constant_l)
    
    #%% Remaining constraints
    
    # Coverage constraint
    m.addConstr(quicksum(delta[r] for r in range(R)) <= alpha*R)
    
    m.params.logtoconsole = 0
    m.setParam("MIPGap", gap) # 1% default
    m.optimize()
    
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write('iismodel.ilp')
    
    UB = [u[t].x for t in range(H)]
    LB = [l[t].x for t in range(H)]
    deltas = [delta[r].x for r in range(R)]

    return UB, LB, deltas

#%% Bisection method for tuning budget of uncertainty

def runBM(samples, K, alpha, a=0, b=1, maxN=10, ubs=None, lbs=None):
    """
    Bisection method with built-in K-fold cross validation.
    
    Parameters:
        samples: set of sample paths (e.g., 2D numpy array, list of lists)
        K: number of folds in cross validation.
        alpha: significance level
        maxN: maximum number of iterations.
    
    maxN is directly related to tolerance:
        e.g., maxN = 10 => tolerance = 1/2^10 = 1/1024 = 0.1%
    """
    R = len(samples)
    indices = np.arange(R)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)

    def crossvalidate(g, warmstart_given=None):
        """Evaluate CB via K-fold CV"""
        cost = 0
        warmstart = {}
        for k in range(K):
            idx = np.array(list(set(range(K)) - {k}))
            in_sample = samples[np.concatenate([folds[i] for i in idx]),:]
            out_sample = samples[folds[k],:]
            if warmstart_given is not None:
                UB, LB, deltas = optimizeCB(in_sample, alpha=alpha, Gamma=g, ubs=ubs, lbs=lbs, warmstart=warmstart_given[k])
            else:
                UB, LB, deltas = optimizeCB(in_sample, alpha=alpha, Gamma=g, ubs=ubs, lbs=lbs)
            coverage = calculateCoverage(UB, LB, out_sample)
            cost += (coverage - 100*(1-alpha)) # linear objective
            warmstart[k] = deltas
        return cost / K, warmstart
    
    N = 0
    while N < maxN:
        N += 1
        print(N)
        c = (a + b) / 2
        f_c, _ = crossvalidate(c)
        if round(f_c) == 0:
            return c
        if round(f_c,2) < 0:
            a = c
        else:
            b = c
    return c
