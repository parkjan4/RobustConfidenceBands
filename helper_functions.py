"""
Author: Jangwon Park
Date: June 18, 2024

This file contains other helper functions.
"""

import numpy as np

def calculateCoverage(UB, LB, samples):
    prc_out = 0
    for s in samples:
        if np.sum(np.array(s) <= np.array(UB)) < len(s):
            prc_out += 1
        if np.sum(np.array(s) >= np.array(LB)) < len(s):
            prc_out += 1
    return 100-100*prc_out/len(samples)