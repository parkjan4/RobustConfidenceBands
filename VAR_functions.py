"""
Author: Jangwon Park
Date: June 18, 2024

This file contains functions related to the vector autoregressive model (VAR)

Pertains to Section "Illustrative example" in the paper.
"""

import numpy as np
from helper_functions import calculateCoverage
np.random.seed(3)

# Parameters for the VAR model (Schussler and Trede, 2016)
A0 = np.array([1, 1])
A1 = np.array([[0.5, 0.3], [-0.6, 1.3]])
S = np.array([[1, 0.5], [0.5, 1]])

def generateVAR(R, H, A0, A1, S, eps):
    """
    Generate R sample paths from VAR model, specified by parameters A0, A1, S.
    
    Parameters:
        R: number of sample paths
        H: time horizon
        A0: constant vector
        A1: matrix coefficient on x^1
        S: covariance matrix for noise
        k: index for row 
    """
    
    all_samples = []
    for r in range(R):
        xt = np.array([0]*len(A0)) # initial state
        path = [xt[0]]
        for t in range(1,H):
            xt = A0 + A1@xt + eps[r,t]
            path.append(xt[0])
        all_samples.append(path)
    return all_samples


def getTrueExtremesVAR(steps):
    """
    Generate many sample paths from VAR (fast) and calculate the 1- and 
    0-quantiles at every discrete time point.
    
    This function pertains to the parameters c^u_t and c^l_t in the paper.
    """
    R = 100000
    eps = np.random.multivariate_normal([0, 0], S, size=(R, steps))
    samples = np.array(generateVAR(R, steps, A0, A1, S, eps=eps))

    ub=[]
    lb=[]
    for t in range(steps):
        ub.append(np.sort(np.array(samples)[:,t])[int(1*R)-1])
        lb.append(np.sort(np.array(samples)[:,t])[int(0*R)])
    return ub, lb


def evaluate_outofsample(UB, LB, trials, steps):
    """
    Evaluate out-of-sample performance of a given CB. 
    This function can be used to reproduce Table 1 in the paper.
    
    Parameters:
        UB: upper band
        LB: lower band
        trials: number of trials (e.g., 4)
    """
    
    # Pre-generate all noise terms in VAR(1) to reduce computation time
    all_eps = np.random.multivariate_normal([0, 0], S, size=(trials, 1000, steps))
    
    avg = 0
    all_covs = []
    for t in range(trials):
        out_samples = generateVAR(1000, steps, A0, A1, S, eps=all_eps[t])
        coverage = calculateCoverage(UB, LB, out_samples)        
        print("Trial {}: {:.2f}%".format(t, coverage))
        avg += coverage
        all_covs.append(coverage/100)
    return (avg / trials), all_covs