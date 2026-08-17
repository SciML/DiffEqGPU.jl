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

The package-specific algorithm selectors and their usage are documented in the
[EnsembleGPUArray](@ref ensemblegpuarray) and [EnsembleGPUKernel](@ref
ensemblegpukernel) manual pages. Developer-facing lower-level interfaces are
collected on the [Developer Interfaces](@ref developer_interfaces) page.
