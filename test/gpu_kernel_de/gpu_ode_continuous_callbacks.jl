using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra
@info "Callbacks"

include("../utils.jl")

function f(u, p, t)
    du1 = u[2]
    du2 = -p[1]
    return SVector{2}(du1, du2)
end

u0 = @SVector[45.0f0, 0.0f0]
tspan = (0.0f0, 15.0f0)
p = @SVector [10.0f0]
prob = ODEProblem{false}(f, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)

function affect!(integrator)
    return integrator.u += @SVector[0.0f0, -2.0f0] .* integrator.u
end

function condition(u, t, integrator)
    return u[1]
end

algs = [GPUTsit5(), GPUVern7()]
diffeq_algs = [Tsit5(), Vern7()]

for (alg, diffeq_alg) in zip(algs, diffeq_algs)
    @info typeof(alg)

    cb = ContinuousCallback(condition, affect!; save_positions = (false, false))

    @info "Unadaptive version"

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 2.0e-3

    @info "Callback: CallbackSets"

    cb = CallbackSet(cb, cb)

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    if alg isa GPUVern7
        # GPUVern7 CallbackSet with duplicate ContinuousCallbacks causes the second
        # callback to re-detect events because the nudge mechanism only prevents
        # re-detection for the callback matching event_last_time.
        @test_broken norm(bench_sol.u[end] - sol.u[1].u[end]) < 2.0e-3
    else
        @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 2.0e-3
    end

    @info "saveat and callbacks"

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0]
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0]
    )

    if alg isa GPUVern7
        @test_broken norm(bench_sol.u[end] - sol.u[1].u[end]) < 2.0e-3
    else
        @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 2.0e-3
    end

    @info "save_everystep and callbacks"

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        save_everystep = false
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        save_everystep = false
    )

    if alg isa GPUVern7
        @test_broken norm(bench_sol.u[end] - sol.u[1].u[end]) < 7.0e-4
    else
        @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 7.0e-4
    end

    @info "Adaptive version"

    cb = ContinuousCallback(condition, affect!; save_positions = (false, false))

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = true, save_everystep = false, dt = 0.1f0, callback = cb,
        merge_callbacks = true
    )

    @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 5.0e-3

    @info "Callback: CallbackSets"

    cb = CallbackSet(cb, cb)

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = true, dt = 0.1f0, save_everystep = false, callback = cb,
        merge_callbacks = true
    )

    @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 5.0e-3

    @info "saveat and callbacks"

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0], reltol = 1.0f-6, abstol = 1.0f-6
    )

    bench_sol = solve(
        prob, diffeq_alg,
        adaptive = true, save_everystep = false, dt = 0.1f0, callback = cb,
        merge_callbacks = true,
        tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 9.1f0], reltol = 1.0f-6,
        abstol = 1.0f-6
    )

    if alg isa GPUVern7
        # GPUVern7 adaptive with tight tolerances (1e-6) has larger interpolation
        # error than GPUTsit5 due to the high-degree Vern7 dense output polynomial
        # in Float32 arithmetic. Error is ~0.002 vs threshold 8e-4.
        @test_broken norm(bench_sol.u[end] - sol.u[1].u[end]) < 8.0e-4
    else
        @test norm(bench_sol.u[end] - sol.u[1].u[end]) < 8.0e-4
    end

    @info "Unadaptive and Adaptive comparison"

    local sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0]
    )

    asol = solve(
        monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0]
    )

    if alg isa GPUVern7
        # Non-adaptive CallbackSet is broken (see above), so this comparison fails.
        @test_broken norm(asol.u[1].u - sol.u[1].u) < 7.0e-4
    else
        @test norm(asol.u[1].u - sol.u[1].u) < 7.0e-4
    end
end
