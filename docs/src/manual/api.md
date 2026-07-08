# API

## Reexported Ensemble API

DiffEqGPU reexports these ensemble and callback helper APIs from SciMLBase.jl and
DiffEqBase.jl for compatibility with the SciML ensemble interface:

  - `EnsembleProblem`
  - `EnsembleSolution`
  - `EnsembleSerial`
  - `EnsembleThreads`
  - `EnsembleDistributed`
  - `BrownFullBasicInit`
  - `CheckInit`
  - `terminate!`

## Lower-Level Algorithms

```@docs
LinSolveGPUSplitFactorize
```
