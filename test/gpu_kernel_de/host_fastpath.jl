using DiffEqGPU, KernelAbstractions, SciMLBase, StaticArrays, Test

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
    @test Base.summarysize(probs) < 30n

    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = n, adaptive = true, dt = 0.2f0, save_everystep = false
    )
    @test Base.summarysize(sol.u) < 100n
    @test length(sol.u) == n
    @test sol.u[1].prob isa SciMLBase.ImmutableODEProblem
    @test all(i -> sol.u[i].t == Float32[0, 1], (1, 2, n - 1, n))
    @test all(i -> isapprox(sol.u[i].u[end][1], exp(0.2f0 + Float32(i % 4) / 10); rtol = 1.0f-6), (1, 2, n - 1, n))

    batched = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 4, batch_size = 2, adaptive = true, dt = 0.2f0,
        save_everystep = false
    )
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
end
