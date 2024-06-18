# Robust Confidence Bands for Simulation Output

This is a repository for a working paper "Robust Confidence Bands for Simulation Output". The code here can be used to reproduce the experimental results in the paper.

## Abstract
The paper presents a robust optimization approach for constructing confidence bands on sample paths, representing transient simulation outputs, whose budget of uncertainty is tuned using a bisection method. Our methodology directly addresses optimization bias within the constraints, avoiding overly narrow confidence bands. Numerical results show that we achieve the desired coverage probabilities with an order-of-magnitude fewer sample paths than a non-robust approach. Our methodology can be used to validate simulation models, which we demonstrate using an Erlang-R queue as an example.
