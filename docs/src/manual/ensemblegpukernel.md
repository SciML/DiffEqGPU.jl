# [EnsembleGPUKernel](@id ensemblegpukernel)

## [API](@id egk_doc)

```@docs
EnsembleGPUKernel
```

### [Specialized Solvers](@id specialsolvers)

```@docs
GPUTsit5
GPUTsit5IController
GPUVern7
GPUVern9
GPUEM
GPUSIEA
GPURosenbrock23
GPURodas4
GPURodas5P
GPUKvaerno3
GPUKvaerno5
```

### Adaptive controllers and benchmark comparisons

The adaptive `GPUTsit5` implementation uses a PI controller. Keep this controller
for general use. A pure I controller responds only to the current error estimate
and generally provides less stable step-size control, so it is not recommended as
a general replacement for the PI controller.

On simple nonstiff benchmarks such as Lorenz, an I-controlled Tsit5 implementation
can appear faster at the same `abstol` and `reltol`, partly because it delivers lower
achieved accuracy. Equal tolerances do not imply equal accuracy. Compare execution
time at matched achieved error, and account for rejected steps, before concluding
that one controller is more efficient.

Changing the controller preserves the Tsit5 Runge–Kutta tableau but changes step
selection and tolerance behavior. Any I-controlled variant should remain an
explicit alternative, preserving the existing `GPUTsit5` behavior.
`GPUTsit5IController()` provides this opt-in I-controller variant.
