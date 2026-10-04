using Adapt, DiffEqGPU, KernelAbstractions, SciMLBase, StaticArrays, Test

struct HostAdaptRecipe{T}
    rate::T
end
Adapt.adapt_structure(::KernelAbstractions.CPU, p::HostAdaptRecipe) = SVector(p.rate)

struct PrecisionRecipe <: AbstractVector{Float32}
    rate::Float32
end
Base.size(::PrecisionRecipe) = (1,)
Base.getindex(p::PrecisionRecipe, i::Int) = p.rate
Adapt.adapt_structure(::KernelAbstractions.CPU, p::PrecisionRecipe) = SVector(Float64(p.rate))

struct RateRecipe
    rate::Float64
end
const _ADAPT_CALLS = Ref(0)
Adapt.adapt_structure(::KernelAbstractions.CPU, p::RateRecipe) =
    (_ADAPT_CALLS[] += 1; SVector(p.rate * _ADAPT_CALLS[]))

@testset "Kernel ensemble host storage" begin
    f(u, p, t) = SVector(p[1] * u[1])
    prob = ODEProblem{false}(f, SVector(1.0f0), (0.0f0, 1.0f0), SVector(0.2f0))
    prob_func = (prob, ctx) -> remake(prob; p = SVector(0.2f0 + Float32(ctx.sim_id % 4) / 10))
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
    n = 4096
    backend = KernelAbstractions.CPU()

    probs = DiffEqGPU._prepare_kernel_problems(
        ensemble, backend, 1:n, nothing, SciMLBase.default_rng_func, nothing
    )
    # Prep stores one original and one adapted problem per trajectory.
    @test Base.summarysize(probs) < 45n

    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = n, adaptive = true, dt = 0.2f0, save_everystep = false
    )
    @test sol.u isa Vector
    @test length(sol.u) == n
    @test sol.u[1].prob isa SciMLBase.ImmutableODEProblem
    @test sol.u[2] === sol.u[2]
    first_sol = sol.u[1]
    sol.u[1] = sol.u[2]
    @test sol.u[1] === sol.u[2]
    sol.u[1] = first_sol
    push!(sol.u, first_sol)
    @test length(sol.u) == n + 1
    pop!(sol.u)
    @test all(i -> sol.u[i].t == Float32[0, 1], (1, 2, n - 1, n))
    @test all(i -> isapprox(sol.u[i].u[end][1], exp(0.2f0 + Float32(i % 4) / 10); rtol = 1.0f-6), (1, 2, n - 1, n))

    batched = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 4, batch_size = 2, adaptive = true, dt = 0.2f0,
        save_everystep = false
    )
    @test batched.u isa Vector
    @test length(batched.u) == 4
    @test all(i -> isapprox(batched.u[i].u[end][1], exp(0.2f0 + Float32(i % 4) / 10); rtol = 1.0f-6), 1:4)

    output = EnsembleProblem(
        prob; prob_func, safetycopy = false,
        output_func = (sol, ctx) -> (sol.u[end][1], false)
    )
    out = solve(
        output, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 4, adaptive = true, dt = 0.2f0, save_everystep = false
    )
    @test all(i -> isapprox(out.u[i], exp(0.2f0 + Float32(i % 4) / 10); rtol = 1.0f-6), 1:4)

    mutable_prob = ODEProblem{false}(f, SVector(1.0f0), (0.0f0, 1.0f0), Float32[0.2])
    mutating = EnsembleProblem(
        mutable_prob;
        prob_func = (prob, ctx) -> (prob.p[1] = Float32(ctx.sim_id); prob),
        safetycopy = true
    )
    DiffEqGPU._make_kernel_problem(mutating, 1, nothing, SciMLBase.default_rng_func, nothing)
    @test mutable_prob.p == Float32[0.2]

    # All-isbits-field ODEProblems take the shallow wrapper reconstruct, not deepcopy.
    bits_prob = ODEProblem{false}(f, SVector(1.0f0), (0.0f0, 1.0f0), SVector(0.2f0))
    @test !isbits(bits_prob)
    @test DiffEqGPU._ensemble_problem_fields_isbits(bits_prob)
    shallow = DiffEqGPU._safety_copy_ensemble_prob(bits_prob)
    @test shallow !== bits_prob
    @test shallow.f === bits_prob.f
    @test shallow.u0 === bits_prob.u0
    @test shallow.p === bits_prob.p
    seen = Ref(false)
    bits_ens = EnsembleProblem(
        bits_prob;
        prob_func = (prob, ctx) -> begin
            seen[] = prob !== bits_prob
            remake(prob; p = SVector(Float32(ctx.sim_id)))
        end,
        safetycopy = true
    )
    DiffEqGPU._make_kernel_problem(bits_ens, 1, nothing, SciMLBase.default_rng_func, nothing)
    @test seen[]
    @test bits_prob.p == SVector(0.2f0)
end

@testset "Kernel host storage respects user adapt_structure" begin
    f(u, p, t) = p[1] * u
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), HostAdaptRecipe(0.2))
    @test isbits(DiffEqGPU.make_prob_compatible(prob))
    backend = KernelAbstractions.CPU()
    for safetycopy in (true, false)
        ensemble = EnsembleProblem(prob; safetycopy)
        sol = solve(
            ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
            trajectories = 2, adaptive = false, dt = 0.01, save_everystep = false
        )
        @test only(sol.u[1].u[end]) ≈ exp(0.2) rtol = 1.0e-10
    end
end

@testset "Kernel host storage honors precision-changing adaptation" begin
    # Float32 `p[1] + 1 - p[1]` is zero for rate 2^24; the adapted Float64
    # parameter makes it one, so the adapted endpoint must reach the kernel.
    f(u, p, t) = (p[1] + 1 - p[1]) * u
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), PrecisionRecipe(2.0f0^24))
    backend = KernelAbstractions.CPU()
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
    backend = KernelAbstractions.CPU()
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
    end
end
