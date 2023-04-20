# [Massively Data-Parallel ODE Solving the Lorenz Equation](@id lorenz)

For example, the following solves the Lorenz equation with 10,000 separate random parameters on the GPU. To start, we create a normal
[`EnsembleProblem` as per DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/). Here's a perfectly good multithreaded CPU parallelized Lorenz solve
over randomized parameters:

```@example lorenz
using DiffEqGPU, OrdinaryDiffEq, CUDA
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 100.0f0)
p = [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(monteprob, Tsit5(), EnsembleThreads(), trajectories = 10_000, saveat = 1.0f0);
```

Changing this to being GPU-parallelized is as simple as changing the ensemble method to
`EnsembleGPUArray`:

```@example lorenz
sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()), trajectories = 10_000, saveat = 1.0f0);
```

and voilà, the method is re-compiled to parallelize the solves over a GPU!

While `EnsembleGPUArray` has a bit of overhead due to its form of GPU code construction,
`EnsembleGPUKernel` is a more restrictive GPU-itizing algorithm that achieves a much lower
overhead in kernel launching costs. However, it requires this problem to be written in
out-of-place form and use [special solvers](@ref specialsolvers). This looks like:

```@example lorenz
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDA

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = 10_000,
            saveat = 1.0f0)
```

Note that this form is also compatible with `EnsembleThreads()`, and `EnsembleGPUArray()`,
so `EnsembleGPUKernel()` simply supports a subset of possible problem types. For more
information on the limitations of `EnsembleGPUKernel()`, see [its docstring](@ref egk_doc).

## Using Stiff ODE Solvers with EnsembleGPUArray

DiffEqGPU also supports more advanced features than shown above. Other tutorials dive into
[handling events or callbacks](@ref events) and [multi-GPU parallelism](@ref multigpu).
But the simplest thing to show is that the generality of solvers allows for other types of
equations. For example, one can handle stiff ODEs with `EnsembleGPUArray` simply by using a
stiff ODE solver. Note that, as explained in the docstring, analytical derivatives
(Jacobian and time gradient) must be supplied. For the Lorenz equation, this looks like:

```@example lorenz
function lorenz_jac(J, u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    x = u[1]
    y = u[2]
    z = u[3]
    J[1, 1] = -σ
    J[2, 1] = ρ - z
    J[3, 1] = y
    J[1, 2] = σ
    J[2, 2] = -1
    J[3, 2] = x
    J[1, 3] = 0
    J[2, 3] = -x
    J[3, 3] = -β
end

function lorenz_tgrad(J, u, p, t)
    nothing
end

func = ODEFunction(lorenz, jac = lorenz_jac, tgrad = lorenz_tgrad)
prob_jac = ODEProblem(func, u0, tspan, p)
monteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)

solve(monteprob_jac, Rodas5(), EnsembleGPUArray(CUDA.CUDABackend()), trajectories = 10_000, saveat = 1.0f0)
```
