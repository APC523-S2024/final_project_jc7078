# Final Project jc7078
This repository is associated with the final project for APC523. This reposirty simulated the modified 
Cahn-Hillard equation with a variety of Runge-Kutta methods. Included in this repository is a python script including the base code, a jupyter notebook exploring the code,
folders with simulation data, and a pdf of the final write-up. I would encourage you to read the write up or the jupyter notebook before diving into the codebase.

For each simulation the following parameters are controllable:
1. N : mesh size of a square
2. b : Strength of atomic interaction
3. M : mobility coefficient
4. kappa : gradient energy coefficient
5. t_final : Final time simulation will go to
6. max_niter : Maximum number of iterations to perform
7. plot : Plots intermediate plots throughout simulation

Folders are named based off these simulation parameters in the following way : {N}_{b}_{M}_{kappa}_{t_final}_{max_niter} where p means decimal point
