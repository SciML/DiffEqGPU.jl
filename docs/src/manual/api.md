# API

## Package Module

```@docs
DiffEqGPU
```

## Reexported Ensemble API

DiffEqGPU reexports the following user-facing ensemble and initialization APIs. Their
definitions and canonical docstrings are maintained by the owning packages:

  - [`SciMLBase.EnsembleProblem`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Ensemble/)
  - [`SciMLBase.EnsembleSolution`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/)
  - [`SciMLBase.EnsembleSerial`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Ensemble/)
  - [`SciMLBase.EnsembleThreads`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Ensemble/)
  - [`SciMLBase.EnsembleDistributed`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Ensemble/)
  - [`SciMLBase.CheckInit`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Init_Solve/)
  - [`SciMLBase.terminate!`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Integrator/)
  - [`DiffEqBase.BrownFullBasicInit`](https://docs.sciml.ai/DiffEqBase/stable/)

## Lower-Level Algorithms

```@docs
LinSolveGPUSplitFactorize
DiffEqGPU.make_prob_compatible
```

## Lower-Level Solve Interface

These are the entry points used to drive the kernel solvers directly, without going
through `EnsembleGPUKernel`. See [Using the Lower Level API](https://docs.sciml.ai/DiffEqGPU/stable/tutorials/lower_level_api/)
for a worked example.

```@docs
DiffEqGPU.vectorized_solve
DiffEqGPU.vectorized_asolve
DiffEqGPU.vectorized_map_solve
```
