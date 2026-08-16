# Developer Interfaces

The APIs on this page are developer-facing. They are documented and versioned so that
DiffEqGPU, SciML, and solver-extension code can share the same contracts, but ordinary
users should prefer the documented algorithm constructors and `solve` interface.

## Ensemble Algorithms

```@docs
DiffEqGPU.EnsembleArrayAlgorithm
DiffEqGPU.EnsembleKernelAlgorithm
DiffEqGPU.maxthreads
DiffEqGPU.maybe_prefer_blocks
DiffEqGPU.lufact!
```

## Problem Conversion

`make_prob_compatible` is the generic conversion hook used before passing a batch of
problems to the lower-level kernel interface. Backend extensions may add methods to the
developer interfaces above, but should preserve the documented return and mutation rules.

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
