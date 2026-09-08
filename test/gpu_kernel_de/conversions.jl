using DiffEqGPU, OrdinaryDiffEq, SciMLBase, StaticArrays, LinearAlgebra, Test
include("../utils.jl")

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob; prob_func, safetycopy = false)

## Don't test the problems in which GPUs don't support FP64 completely yet
## Creating StepRangeLen causes some param types to be FP64 inferred by `float` function
if ENV["GROUP"] ∉ ("Metal", "oneAPI")
    @test solve(
        monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
        trajectories = 10_000,
        saveat = 1:10
    ).u[1].t == Float32.(1:10)

    @test solve(
        monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
        trajectories = 10_000,
        saveat = 1:0.1:10
    ).u[1].t == 1.0f0:0.1f0:10.0f0

    @test solve(
        monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
        trajectories = 10_000,
        saveat = 1:(1.0f0):10
    ).u[1].t == 1:1.0f0:10

    @test solve(
        monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
        trajectories = 10_000,
        saveat = 1.0
    ).u[1].t == 0.0f0:1.0f0:10.0f0
end

@test solve(
    monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = [1.0f0, 5.0f0, 10.0f0]
).u[1].t == [1.0f0, 5.0f0, 10.0f0]

@test solve(
    monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = [1.0, 5.0, 10.0]
).u[1].t == [1.0f0, 5.0f0, 10.0f0]

@testset "Host problems and construction count" for safetycopy in (false, true)
    calls = zeros(Int, 3)
    analytic(u0, p, t) = u0 * exp(p[1] * t)
    f = ODEFunction{false}((u, p, t) -> p[1] * u; analytic)
    prob = ODEProblem(f, [1.0f0], (0.0f0, 1.0f0), SVector(0.0f0))
    prob_func = function (prob, ctx)
        calls[ctx.sim_id] += 1
        return remake(prob; p = SVector(Float32(ctx.sim_id)))
    end
    ensemble = EnsembleProblem(prob; prob_func, safetycopy)
    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 3, adaptive = false, dt = 0.01f0, save_everystep = false
    )
    @test calls == ones(Int, 3)
    @test [s.prob.p for s in sol.u] == [SVector(Float32(i)) for i in 1:3]
    @test all(s -> s.prob.f.analytic === analytic, sol.u)
    @test all(s -> s.prob.u0 isa SVector{1, Float32}, sol.u)
end

@testset "Initialization preprocessing runs once" begin
    updates = Ref(0)
    initprob = NonlinearProblem{false}((u, p) -> u .- 1.0f0, SVector(0.0f0))
    update_init = function (initprob, prob)
        updates[] += 1
        return initprob
    end
    initdata = SciMLBase.OverrideInitData(
        initprob, update_init, sol -> sol.u, nothing, nothing, Val(true)
    )
    f = ODEFunction{false}((u, p, t) -> zero(u); initialization_data = initdata)
    prob = ODEProblem(f, SVector(0.0f0), (0.0f0, 0.1f0))
    sol = solve(
        EnsembleProblem(prob; safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(backend, 0.0); trajectories = 3,
        adaptive = false, dt = 0.1f0, save_everystep = false
    )
    @test updates[] == 3
    @test all(s -> s.u[end] ≈ SVector(1.0f0), sol.u)
end
