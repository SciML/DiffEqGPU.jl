# API

## Reexported Ensemble API

```@docs
SciMLBase.EnsembleProblem
SciMLBase.EnsembleSolution
SciMLBase.EnsembleSerial
SciMLBase.EnsembleThreads
SciMLBase.EnsembleDistributed
SciMLBase.CheckInit
SciMLBase.terminate!
DiffEqBase.BrownFullBasicInit
```

## Lower-Level Algorithms

```@docs
LinSolveGPUSplitFactorize
```

## Lower-Level Solve Interface

These are the entry points used to drive the kernel solvers directly, without going
through `EnsembleGPUKernel`. See [Using the Lower Level API](@ref lowerlevel) for a
worked example.

```@docs
DiffEqGPU.vectorized_solve
DiffEqGPU.vectorized_asolve
```
