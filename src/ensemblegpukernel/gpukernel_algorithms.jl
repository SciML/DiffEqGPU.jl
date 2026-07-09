"""
    GPUTsit5()

Fifth-order Tsitouras Runge-Kutta method specialized for `EnsembleGPUKernel` ODE
solves.

Use `GPUTsit5` as the ODE algorithm when solving an `EnsembleProblem` with
`EnsembleGPUKernel`:

```julia
solve(
    ensemble_prob, GPUTsit5(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0
)
```

`GPUTsit5` supports the `EnsembleGPUKernel` restrictions, including out-of-place ODE
functions over GPU-compatible static state containers. For a similar CPU implementation,
see `SimpleATsit5` from SimpleDiffEq.jl.
"""
struct GPUTsit5 <: GPUODEAlgorithm end

"""
    GPUVern7()

Seventh-order Verner Runge-Kutta method specialized for `EnsembleGPUKernel` ODE
solves.

Use `GPUVern7` for non-stiff ODE ensemble solves that satisfy the `EnsembleGPUKernel`
kernel-generation restrictions:

```julia
solve(
    ensemble_prob, GPUVern7(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0
)
```
"""
struct GPUVern7 <: GPUODEAlgorithm end

"""
    GPUVern9()

Ninth-order Verner Runge-Kutta method specialized for `EnsembleGPUKernel` ODE solves.

Use `GPUVern9` for high-accuracy non-stiff ODE ensemble solves that satisfy the
`EnsembleGPUKernel` kernel-generation restrictions:

```julia
solve(
    ensemble_prob, GPUVern9(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0
)
```
"""
struct GPUVern9 <: GPUODEAlgorithm end

"""
    GPURosenbrock23(; autodiff = Val{true}())

Second/third-order Rosenbrock-W method specialized for stiff `EnsembleGPUKernel` ODE
solves.

# Keyword Arguments

  - `autodiff`: whether automatic differentiation is used for derivative generation.
    Pass `Val{false}()` when providing the required derivatives manually.

```julia
solve(
    ensemble_prob, GPURosenbrock23(), EnsembleGPUKernel(backend);
    trajectories = 10_000
)
```
"""
struct GPURosenbrock23{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
    GPURodas4(; autodiff = Val{true}())

Fourth-order Rosenbrock method specialized for stiff `EnsembleGPUKernel` ODE solves.

# Keyword Arguments

  - `autodiff`: whether automatic differentiation is used for derivative generation.
    Pass `Val{false}()` when providing the required derivatives manually.

```julia
solve(ensemble_prob, GPURodas4(), EnsembleGPUKernel(backend); trajectories = 10_000)
```
"""
struct GPURodas4{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
    GPURodas5P(; autodiff = Val{true}())

Fifth-order Rosenbrock method specialized for stiff `EnsembleGPUKernel` ODE solves.

# Keyword Arguments

  - `autodiff`: whether automatic differentiation is used for derivative generation.
    Pass `Val{false}()` when providing the required derivatives manually.

```julia
solve(ensemble_prob, GPURodas5P(), EnsembleGPUKernel(backend); trajectories = 10_000)
```
"""
struct GPURodas5P{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
    GPUKvaerno3(; autodiff = Val{true}())

Third-order ESDIRK method specialized for stiff `EnsembleGPUKernel` ODE solves.

# Keyword Arguments

  - `autodiff`: whether automatic differentiation is used for derivative generation.
    Pass `Val{false}()` when providing the required derivatives manually.

```julia
solve(ensemble_prob, GPUKvaerno3(), EnsembleGPUKernel(backend); trajectories = 10_000)
```
"""
struct GPUKvaerno3{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
    GPUKvaerno5(; autodiff = Val{true}())

Fifth-order ESDIRK method specialized for stiff `EnsembleGPUKernel` ODE solves.

# Keyword Arguments

  - `autodiff`: whether automatic differentiation is used for derivative generation.
    Pass `Val{false}()` when providing the required derivatives manually.

```julia
solve(ensemble_prob, GPUKvaerno5(), EnsembleGPUKernel(backend); trajectories = 10_000)
```
"""
struct GPUKvaerno5{AD} <: GPUODEImplicitAlgorithm{AD} end

for Alg in [:GPURosenbrock23, :GPURodas4, :GPURodas5P, :GPUKvaerno3, :GPUKvaerno5]
    @eval begin
        function $Alg(; autodiff = Val{true}())
            return $Alg{_unwrap_val(autodiff)}()
        end
    end
end

"""
    GPUEM()

Euler-Maruyama method with weak order 1.0 specialized for `EnsembleGPUKernel` SDE
solves.

```julia
solve(
    ensemble_prob, GPUEM(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0
)
```
"""
struct GPUEM <: GPUSDEAlgorithm end

"""
    GPUSIEA()

Weak order 2.0 SIEA method for Ito SDEs specialized for `EnsembleGPUKernel` SDE
solves.

```julia
solve(
    ensemble_prob, GPUSIEA(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0
)
```
"""
struct GPUSIEA <: GPUSDEAlgorithm end
