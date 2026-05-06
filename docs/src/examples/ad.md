# Using GPU-accelerated Ensembles with Automatic Differentiation

`EnsembleGPUArray` carries a custom forward-mode rrule that propagates
`ForwardDiff.Dual` numbers through the parallel solve, so it can sit inside
a Lux/Optimisers training loop driven by ForwardDiff.jl gradients. Below
the parameters of a tiny Lux model become the inputs to a parallel ensemble
of ODEs, and the `Dense` weights get updated to drive the integrated states
toward 1.

```@example ad
using OrdinaryDiffEq, Lux, Optimisers, ForwardDiff, DiffEqGPU, CUDA, Random

CUDA.allowscalar(false)

# A tiny Lux model whose parameters are what we train. It maps a constant
# input to the two ODE parameters.
const dense = Dense(1 => 2)
const x = Float32[1.0]
rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = Lux.setup(rng, dense)
# Optimisers.destructure flattens `ps` into a vector + reconstructor closure;
# ForwardDiff differentiates over that flat vector.
flat_ps, restore = Optimisers.destructure(ps)

u0 = Float32[3.0]

function modelf(du, u, p, t)
    du[1] = 1.01f0 * u[1] * p[1] * p[2]
end

# Element type T flows in from ForwardDiff so that u0 / tspan / saveat all
# match the Dual eltype of the perturbed parameters during gradient calls.
function model(p, ::Type{T} = Float32) where {T}
    prob = ODEProblem(modelf, T.(u0), T.((0.0f0, 1.0f0)), p)

    function prob_func(prob, ctx)
        remake(prob, u0 = T(0.5) .+ T(ctx.sim_id) / T(100) .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    solve(
        ensemble_prob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
        saveat = T(0.1), trajectories = 10
    )
end

# loss function: run the Lux model to produce ODE parameters, then score the ensemble
function loss(flat_ps)
    p_vec, _ = dense(x, restore(flat_ps), st)
    T = eltype(p_vec)
    sum(abs2, T(1.0) .- Array(model(p_vec, T)))
end

println("Starting to train")

l1 = loss(flat_ps)
@show l1

# Optimisers.jl handles parameter updates; ForwardDiff.jl handles gradients.
# Reverse-mode (Zygote / SciMLSensitivity) through the EnsembleProblem adjoint
# is currently broken on the CI Julia + Zygote stack
# (https://github.com/SciML/SciMLSensitivity.jl reports the `ProjectTo`
# DimensionMismatch on `Vector{Vector{Vector{Float32}}}`); ForwardDiff is
# the supported path through `EnsembleGPUArray` for now.
opt_state = Optimisers.setup(Optimisers.Adam(0.1f0), flat_ps)
for epoch in 1:10
    grads = ForwardDiff.gradient(loss, flat_ps)
    Optimisers.update!(opt_state, flat_ps, grads)
    @show loss(flat_ps)
end
```

Forward-mode automatic differentiation works as well, as demonstrated by its capability
to recompile for Dual number arithmetic:

```@example ad
using OrdinaryDiffEq, DiffEqGPU, ForwardDiff, Test

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = [ForwardDiff.Dual(1.0f0, (1.0, 0.0, 0.0)), ForwardDiff.Dual(0.0f0, (0.0, 1.0, 0.0)),
    ForwardDiff.Dual(0.0f0, (0.0, 0.0, 1.0))]
tspan = (0.0f0, 100.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
    trajectories = 10_000,
    saveat = 1.0f0)
```
