# Using GPU-accelerated Ensembles with Automatic Differentiation

`EnsembleGPUArray` comes with derivative overloads for reverse mode automatic differentiation,
and thus can be thrown into deep learning training loops. The following is an example
of this use:

!!! warning
    
    Reverse mode over `EnsembleGPUArray` is currently broken and this example does not
    run. `Zygote.gradient` fails inside the pullback of `batch_solve` with
    `DimensionMismatch: array with ndims(x) == 1 > 0 cannot have dx::Number`, because the
    cotangent reaching `solus[i]` is a scalar where a `Vector{Vector}` is required. The
    forward pass and the forward-mode example below are unaffected. This block is left
    unevaluated until that is fixed.

```julia
using OrdinaryDiffEq, SciMLSensitivity, Lux, Optimisers, Zygote, DiffEqGPU, CUDA, Random

CUDA.allowscalar(false)

# A tiny Lux model whose parameters are what we train. It maps a constant
# input to the two ODE parameters.
const dense = Dense(1 => 2)
const x = Float32[1.0]
rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = Lux.setup(rng, dense)

u0 = Float32[3.0]

function modelf(du, u, p, t)
    du[1] = 1.01f0 * u[1] * p[1] * p[2]
    return
end

function model(p)
    prob = ODEProblem(modelf, u0, (0.0f0, 1.0f0), p)

    function prob_func(prob, ctx)
        return remake(prob, u0 = 0.5f0 .+ Float32(ctx.sim_id) / 100 .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob; prob_func)
    return solve(
        ensemble_prob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
        saveat = 0.1f0, trajectories = 10
    )
end

# loss function: run the Lux model to produce ODE parameters, then score the ensemble
function loss(ps)
    p_vec, _ = dense(x, ps, st)
    return sum(abs2, 1.0f0 .- Array(model(p_vec)))
end

println("Starting to train")

l1 = loss(ps)
@show l1

# Optimisers.jl handles parameter updates; Zygote.jl handles gradients
opt_state = Optimisers.setup(Optimisers.Adam(0.1f0), ps)
for epoch in 1:10
    grads = Zygote.gradient(loss, ps)
    Optimisers.update!(opt_state, ps, grads[1])
    @show loss(ps)
end
```

Forward-mode automatic differentiation works as well, as demonstrated by its capability
to recompile for Dual number arithmetic:

```@example ad
using OrdinaryDiffEq, DiffEqGPU, ForwardDiff, Test, CUDA

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    return
end

u0 = [
    ForwardDiff.Dual(1.0f0, (1.0, 0.0, 0.0)),
    ForwardDiff.Dual(0.0f0, (0.0, 1.0, 0.0)),
    ForwardDiff.Dual(0.0f0, (0.0, 0.0, 1.0)),
]
tspan = (0.0f0, 100.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob; prob_func)
@time sol = solve(
    monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
    trajectories = 10_000,
    saveat = 1.0f0
)
```

## Enzyme reverse mode with `EnsembleGPUKernel`

For `GPUTsit5`, use Enzyme to differentiate a scalar loss of the ensemble's final
states. Set `cpu_offload` to zero and `save_everystep = false`. Keep the parameters
on the host when constructing the problems; the ensemble solver transfers the
compatible problems to the selected backend.

```@example enzyme_kernel
using DiffEqGPU, Enzyme, KernelAbstractions, SciMLBase, StaticArrays, Test

function ensemble_loss(p, backend)
    rhs(u, p, t) = p[1] * u
    prob = ODEProblem{false}(rhs, SVector(1.0), (0.0, 1.0), SVector(p[1]))
    prob_func = (prob, ctx) -> remake(prob; p = SVector(p[ctx.sim_id]))
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = length(p), adaptive = false, dt = 0.05,
        save_everystep = false
    )
    return sum(s -> sum(s.u[end]), sol.u)
end

p = [0.2, -0.3]
dp = zero(p)
backend = CPU()
Enzyme.autodiff(
    Reverse, ensemble_loss, Active, Duplicated(p, dp), Const(backend)
)
@test dp ≈ exp.(p)
dp
```

The example uses KernelAbstractions' CPU backend. For NVIDIA GPUs, load `CUDA` and
use `CUDA.CUDABackend()`. Reset `dp` to zero before each reverse-mode call: Enzyme
accumulates into this buffer. Fixed-step gradients differentiate the numerical
steps; adaptive gradients differentiate the executed solver path and are not a
record-and-replay adjoint with a frozen mesh.

Callbacks and DAE initialization are solver features; their absence from this
example does not imply that they are unsupported. Representative Float64 CPU
checks also validate Enzyme gradients through a fixed-time discrete callback,
nonlinear initialization, the final integration time, and requested `saveat`
times. These checks do not establish differentiation support for every callback
or a complete singular-mass DAE solve. In particular, a continuous callback with
a parameter-dependent event time can produce
[incorrect gradients](https://github.com/SciML/DiffEqGPU.jl/issues/533); event-time
sensitivities need separate validation.
