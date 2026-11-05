using Adapt, DiffEqGPU, KernelAbstractions, SciMLBase, StaticArrays, Test

const backend = if get(ENV, "GROUP", "CPU") == "CUDA"
    using CUDA
    CUDA.CUDABackend()
elseif get(ENV, "GROUP", "CPU") == "JLArrays"
    using JLArrays
    JLArrays.JLBackend()
else
    KernelAbstractions.CPU()
end

struct HostAdaptRecipe{T}
    rate::T
end
Adapt.adapt_structure(::typeof(backend), p::HostAdaptRecipe) = SVector(p.rate)

struct PrecisionRecipe <: AbstractVector{Float32}
    rate::Float32
end
Base.size(::PrecisionRecipe) = (1,)
Base.getindex(p::PrecisionRecipe, i::Int) = p.rate
Adapt.adapt_structure(::typeof(backend), p::PrecisionRecipe) = SVector(Float64(p.rate))

struct RateRecipe
    rate::Float64
end
const _ADAPT_CALLS = Ref(0)
Adapt.adapt_structure(::typeof(backend), p::RateRecipe) =
    (_ADAPT_CALLS[] += 1; SVector(p.rate * _ADAPT_CALLS[]))

@testset "Primal kernel problem representation" for T in (Float32, Float64)
    f = let rate = T(0.25)
        (u, p, t) -> rate * p[1] * u
    end
    prob = ODEProblem{false}(f, SVector(one(T)), (zero(T), one(T)), SVector(T(0.2)))
    for safetycopy in (false, true)
        ensemble = EnsembleProblem(
            prob; safetycopy,
            prob_func = (prob, ctx) -> remake(prob; p = SVector(T(ctx.sim_id) / 10))
        )
        host, adapted = DiffEqGPU._prepare_kernel_problems(
            ensemble, backend, 1:3, nothing, SciMLBase.default_rng_func, nothing
        )
        @test eltype(host) <: SciMLBase.ImmutableODEProblem
        @test eltype(adapted) <: SciMLBase.ImmutableODEProblem
        @test adapted[1] === Adapt.adapt(backend, host[1])
        @test [p.p[1] for p in host] == T[0.1, 0.2, 0.3]
        @test all(p -> isbits(p), adapted)
        @test all(p -> p.f(p.u0, p.p, zero(T)) == T(0.25) * p.p[1] * p.u0, adapted)
    end
end

@testset "Kernel host storage respects user adapt_structure" begin
    f(u, p, t) = p[1] * u
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), HostAdaptRecipe(0.2))
    @test isbits(DiffEqGPU.make_prob_compatible(prob))
    for safetycopy in (true, false)
        ensemble = EnsembleProblem(prob; safetycopy)
        sol = solve(
            ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
            trajectories = 2, adaptive = false, dt = 0.01, save_everystep = false
        )
        @test only(sol.u[1].u[end]) ≈ exp(0.2) rtol = 1.0e-10
        @test sol.u isa Vector
        @test sol.u[1].prob isa SciMLBase.ImmutableODEProblem
        @test sol.u[1].prob.p === prob.p
    end
end

@testset "Kernel host storage honors precision-changing adaptation" begin
    # Float32 `p[1] + 1 - p[1]` is zero for rate 2^24; the adapted Float64
    # parameter makes it one, so the adapted endpoint must reach the kernel.
    f(u, p, t) = (p[1] + 1 - p[1]) * u
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), PrecisionRecipe(2.0f0^24))
    compatible = DiffEqGPU.make_prob_compatible(prob)
    adapted = Adapt.adapt(backend, compatible)
    @test isbits(compatible)
    @test isequal(compatible.p, adapted.p)
    @test f(compatible.u0, compatible.p, 0.0) == SVector(0.0)
    @test f(adapted.u0, adapted.p, 0.0) == SVector(1.0)
    for safetycopy in (true, false)
        ensemble = EnsembleProblem(prob; safetycopy)
        sol = solve(
            ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
            trajectories = 2, adaptive = false, dt = 0.01, save_everystep = false
        )
        @test only(sol.u[1].u[end]) ≈ exp(1.0) rtol = 1.0e-10
    end
end

@testset "Kernel host storage adapts once per trajectory" begin
    f(u, p, t) = p[1] * u
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), RateRecipe(0.1))
    for safetycopy in (true, false)
        _ADAPT_CALLS[] = 0
        ensemble = EnsembleProblem(prob; safetycopy)
        sol = solve(
            ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
            trajectories = 3, adaptive = false, dt = 0.01, save_everystep = false
        )
        endpoints = [only(s.u[end]) for s in sol.u]
        @test _ADAPT_CALLS[] == 3
        @test endpoints ≈ exp.([0.1, 0.2, 0.3]) rtol = 1.0e-10
        @test all(s -> s.prob.p === prob.p, sol.u)
    end
end
