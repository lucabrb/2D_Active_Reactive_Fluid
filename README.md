# 2D_Active_Reactive_Fluid
This is a solver of isotropic, active-reactive fluid equations in 2D.

The current version implements a GPU-accelerated pseudospectral solver, with simple Euler-forward time integration.

It is meant to be run on GPUs, through a university cluster that uses Slurm as workload manager. To facilitate interactions with the cluster, it is written so as to be launched through the [Julia Utilities](https://github.com/Lu-Dumoulin/Utilities) package of Ludovic Dumoulin.

### To do
- [ ] Add eqn for nematic order parameter
- [ ] Test current solutions against higher-order time integrators: Midpoint, RK4, Adaptive...
- [ ] Remove all comments from [the original 1D solver](https://github.com/lucabrb/Barberi-Kruse-2022) the current code is based on.
- [ ] Add comments to clarify implementation of GPU-acceleration
- [ ] Add comments to clarify cluster-related lines (_e.g._ job array index etc.)
