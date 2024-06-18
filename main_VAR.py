"""
Author: Jangwon Park
Date: June 18, 2024
"""
#%% Libraries
# standard libraries
import numpy as np
np.random.seed(3)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# custom python files
from VAR_functions import generateVAR, getTrueExtremesVAR, evaluate_outofsample
from optimization_functions import optimizeCB, runBM

#%% Vector autoregressive (VAR) experiments. Section "Illustrative example" in the paper.
# Parameters for the VAR model (Schussler and Trede, 2016)
A0 = np.array([1, 1])
A1 = np.array([[0.5, 0.3], [-0.6, 1.3]])
S = np.array([[1, 0.5], [0.5, 1]])

R = 500 # number of sample paths (e.g., 100, 200, 500, 1000, ...)
steps = 12 # number of time points
eps = np.random.multivariate_normal([0, 0], S, size=(R, steps)) # noise parameter in VAR

# Generate sample paths from VAR
samples = np.array(generateVAR(R, steps, A0, A1, S, eps=eps))

ubs, lbs = getTrueExtremesVAR(steps)

alpha = 0.1

# Determine number of folds in cross validation
if R <= 300:
    K = 2 # number of folds
else:
    K = 4 # number of folds
trials = 4 # out-of-sample trials
print("(R,alpha): ({},{})".format(R, alpha))

# First, solve for "nominal" CB
UBn, LBn, warmstart_given = optimizeCB(samples, alpha=alpha, Gamma=0.0, ubs=ubs, lbs=lbs)
avg, _ = evaluate_outofsample(UBn, LBn, trials, steps)
print("Average out-of-sample coverage (NOMINAL): {:.2f}%".format(avg))
print("----------------------------------------------------------------------")

# Plot confidence band along with sample paths
plt.close("all")
plt.figure()
for r in range(R):
    plt.plot(samples[r], alpha=0.33)
plt.plot(UBn, linewidth=3, linestyle=":", color="black", label="Nominal")
plt.plot(LBn, linewidth=3, linestyle=":", color="black")

# Solve for "robust" CB
Gamma = runBM(samples, K, alpha, ubs=ubs, lbs=lbs)
print("Gamma:", Gamma)

UB, LB, _ = optimizeCB(samples, alpha=alpha, Gamma=Gamma, ubs=ubs, lbs=lbs)
avg, _ = evaluate_outofsample(UB, LB, trials, steps)
print("Average out-of-sample coverage (NOMINAL): {:.2f}%".format(avg))
print("----------------------------------------------------------------------")

# Plot robust CB along with sample paths
fs = 14
plt.plot(UB, linewidth=3, color="black", label="Robust")
plt.plot(LB, linewidth=3, color="black")
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim([-10,15])
plt.xlabel(r"$t$", fontsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.show()