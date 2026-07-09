# Developer Interfaces

The APIs on this page are developer-facing. They are documented and versioned so that
DiffEqGPU, SciML, and solver-extension code can share the same contracts, but ordinary
users should prefer the documented algorithm constructors and `solve` interface.

## Ensemble Algorithms

```@docs
DiffEqGPU.EnsembleArrayAlgorithm
DiffEqGPU.EnsembleKernelAlgorithm
```

## Kernel ODE and SDE Algorithms

```@docs
DiffEqGPU.GPUODEAlgorithm
DiffEqGPU.GPUSDEAlgorithm
DiffEqGPU.GPUODEImplicitAlgorithm
```

## Kernel Nonlinear Solvers

```@docs
DiffEqGPU.AbstractNLSolver
DiffEqGPU.AbstractNLSolverCache
DiffEqGPU.NLSolver
```
