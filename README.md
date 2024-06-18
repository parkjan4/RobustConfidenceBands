# Robust Confidence Bands for Simulation Output

This is a repository for the working paper "Robust Confidence Bands for Simulation Output." The code here can be used to reproduce the experiments in the paper.

## Abstract
The paper presents a robust optimization approach for constructing confidence bands on sample paths, representing finite-horizon simulation outputs, whose budget of uncertainty is tuned using a bisection method. Our methodology directly addresses optimization bias within the constraints, avoiding overly narrow confidence bands. Numerical results show that we achieve the desired coverage probabilities with an order-of-magnitude fewer sample paths than a non-robust approach. Our methodology can be used to validate simulation models, which we demonstrate using an Erlang-R queue as an example.

## Requirements
- Gurobi
- NumPy
- Pandas
- Matplotlib
- Seaborn

## How to run the code
In the current version, the best way to reproduce the experiments is within an IDE (PyCharm, Spyder, ...), rather than through a terminal. To reproduce the results in
- "Section 5: Illustrative example," open `main_VAR.py` and execute the whole script;
- "Section 6: Case Study," open `main_ErlangR.py` and execute the whole script.

For `main_ErlangR.py`, the current version assumes that pre-computed confidence bands (e.g., `LBr_300_5` and `UBr_300_5`) exist within the same working directory. There is an option to re-solve for the confidence bands.

## Other files
- `VAR_functions.py`: contains functions for generating sample paths from a vector autoregressive (VAR) model. Model parameters are extracted from [Schuussler and Trede (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0165176516302178).
- `optimiztion_functions.py`: contains implementations of a Gurobi model and the bisection method.
- `helper_functions.py`: contains auxiliary functions such as calculating coverage probabilities.
- `Erlang_R.py`: simulation logic for the Erlang-R queue. Model parameters are extracted from [Yom-Tov and Mandelbaum (2014)](https://pubsonline-informs-org.myaccess.library.utoronto.ca/doi/abs/10.1287/msom.2013.0474).
- Simulation classes: `SimClasses.py`, `SimFunctions.py`, `SimRNG.py` (credit to [Barry Nelson](https://www.mccormick.northwestern.edu/research-faculty/directory/profiles/nelson-barry.html)).
- Data used in Section 6, extracted from [Yom-Tov and Mandelbaum (2014)](https://pubsonline-informs-org.myaccess.library.utoronto.ca/doi/abs/10.1287/msom.2013.0474): `MCE_cumulative_arrivals.csv`, `MCE_cumulative_departures.csv`.
