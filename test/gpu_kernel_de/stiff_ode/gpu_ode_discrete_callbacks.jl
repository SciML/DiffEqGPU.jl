using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra
@info "Callbacks"

include("../../utils.jl")

function f(u, p, t)
    du1 = -u[1]
    return SVector{1}(du1)
end

function f_jac(u, p, t)
    return SMatrix{1, 1, eltype(u)}(-1)
end

function f_tgrad(u, p, t)
    return SVector{1, eltype(u)}(0.0)
end

func = ODEFunction(f, jac = f_jac, tgrad = f_tgrad)
u0 = @SVector [10.0f0]
prob = ODEProblem{false}(func, u0, (0.0f0, 10.0f0))
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)

algs = [GPURosenbrock23(), GPURodas4()]
diffeq_algs = [Rosenbrock23(), Rodas4()]

for (alg, diffeq_alg) in zip(algs, diffeq_algs)
    @info typeof(alg)

    condition(u, t, integrator) = t == 2.40f0

    affect!(integrator) = integrator.u += @SVector[10.0f0]

    cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

    @info "Unadaptive version"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0])

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 2e-3
    @test norm(bench_sol.u - sol.u[1].u) < 5e-3

    #Test the truncation error due to floating point math, encountered when adjusting t for tstops
    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.01f0, callback = cb, merge_callbacks = true,
        tstops = [4.0f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 0.01f0, callback = cb, merge_callbacks = true,
        tstops = [4.0f0])

    @test norm(bench_sol(4.0f0) - sol.u[1](4.0f0)) < 2e-6
    @test norm(bench_sol.u - sol.u[1].u) < 3e-5

    @info "Callback: CallbackSets"

    condition_1(u, t, integrator) = t == 2.40f0

    condition_2(u, t, integrator) = t == 4.0f0

    cb_1 = DiscreteCallback(condition_1, affect!; save_positions = (false, false))
    cb_2 = DiscreteCallback(condition_2, affect!; save_positions = (false, false))

    cb = CallbackSet(cb_1, cb_2)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0])

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 2e-3
    @test norm(bench_sol(4.0f0) - sol.u[1](4.0f0)) < 3e-3
    @test norm(bench_sol.u - sol.u[1].u) < 7e-3

    @info "saveat and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], saveat = [0.0f0, 6.0f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], saveat = [0.0f0, 6.0f0])

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 1e-3
    @test norm(bench_sol(6.0f0) - sol.u[1](6.0f0)) < 3e-3
    @test norm(bench_sol.u - sol.u[1].u) < 3e-3

    @info "save_everystep and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], save_everystep = false)

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], save_everystep = false)

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 3e-5
    @test norm(bench_sol(4.0f0) - sol.u[1](4.0f0)) < 5e-5
    @test norm(bench_sol.u - sol.u[1].u) < 2e-4

    @info "Adaptive version"

    cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [4.0f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = true, save_everystep = false, dt = 1.0f0, callback = cb,
        merge_callbacks = true,
        tstops = [4.0f0])

    @test norm(bench_sol(4.0f0) - sol.u[1](4.0f0)) < 5e-5
    @test norm(bench_sol.u - sol.u[1].u) < 2e-4

    @info "Callback: CallbackSets"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = true, dt = 1.0f0, save_everystep = false, callback = cb,
        merge_callbacks = true,
        tstops = [2.40f0, 4.0f0])

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 6e-4
    @test norm(bench_sol(4.0f0) - sol.u[1](4.0f0)) < 1e-3
    @test norm(bench_sol.u - sol.u[1].u) < 3e-3

    @info "saveat and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], saveat = [0.0f0, 6.0f0], reltol = 1.0f-7,
        abstol = 1.0f-7)

    bench_sol = solve(prob, diffeq_alg,
        adaptive = true, save_everystep = false, dt = 1.0f0, callback = cb,
        merge_callbacks = true,
        tstops = [2.40f0, 4.0f0], saveat = [0.0f0, 6.0f0], reltol = 1.0f-7,
        abstol = 1.0f-7)

    @test norm(bench_sol(2.40f0) - sol.u[1](2.40f0)) < 7e-3
    @test norm(bench_sol(6.0f0) - sol.u[1](6.0f0)) < 2e-2
    @test norm(bench_sol.u - sol.u[1].u) < 2e-2

    @info "Terminate callback"

    cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0])

    bench_sol = solve(prob, diffeq_alg,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        tstops = [2.40f0])

    @test norm(bench_sol.t - sol.u[1].t) < 2e-3
    @test norm(bench_sol.u - sol.u[1].u) < 5e-3
end
