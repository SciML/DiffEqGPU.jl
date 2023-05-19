using DiffEqGPU, StaticArrays, OrdinaryDiffEq, LinearAlgebra

include("../../utils.jl")

function f(u, p, t)
    du1 = -u[1]
    return SVector{1}(du1)
end

function f_jac(u, p, t)
    return @SMatrix [-1.0f0]
end

function f_tgrad(u, p, t)
    return SVector{1, eltype(u)}(0.0)
end

u0 = @SVector [10.0f0]
tspan = (0.0f0, 10.0f0)
func = ODEFunction(f, jac = f_jac, tgrad = f_tgrad)
prob = ODEProblem{false}(func, u0, tspan)

algs = (GPURosenbrock23(), GPURodas4(), GPURodas5P(), GPUKvaerno3(), GPUKvaerno5())
for alg in algs
    prob_func = (prob, i, repeat) -> remake(prob, p = p)
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
    @info typeof(alg)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 10,
                      adaptive = false, dt = 0.01f0)
    asol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 10,
                 adaptive = true, dt = 0.1f-1)

    @test sol.converged == true
    @test asol.converged == true

    ## Regression test

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7)

    @test norm(bench_sol.u[end] - sol[1].u[end]) < 5e-3
    @test norm(bench_asol.u - asol[1].u) < 6e-3

    ### solve parameters

    saveat = [2.0f0, 4.0f0]

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 2,
                      adaptive = false, dt = 0.01f0, saveat = saveat)

    asol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 2,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
                 saveat = saveat)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7, saveat = saveat)

    @test norm(asol[1].u[end] - sol[1].u[end]) < 4e-2

    @test norm(bench_sol.u - sol[1].u) < 2e-4
    #Use to fail for 2e-4
    @test norm(bench_asol.u - asol[1].u) < 2e-3

    @test length(sol[1].u) == length(saveat)
    @test length(asol[1].u) == length(saveat)

    saveat = collect(0.0f0:0.1f0:10.0f0)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend), trajectories = 2,
                      adaptive = false, dt = 0.01f0, saveat = saveat)

    asol = solve(monteprob, alg, EnsembleGPUKernel(backend), trajectories = 2,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
                 saveat = saveat)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7, saveat = saveat)

    #Fails also OrdinaryDiffEq.jl
    # @test norm(asol[1].u[end] - sol[1].u[end]) < 6e-3

    @test norm(bench_sol.u - sol[1].u) < 2e-3
    @test norm(bench_asol.u - asol[1].u) < 3e-2

    @test length(sol[1].u) == length(saveat)
    @test length(asol[1].u) == length(saveat)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend), trajectories = 2,
                      adaptive = false, dt = 0.01f0, save_everystep = false)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0,
                      save_everystep = false)

    @test norm(bench_sol.u - sol[1].u) < 5e-3

    @test length(sol[1].u) == length(bench_sol.u)

    ### Huge number of threads
    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0),
                      trajectories = 10_000,
                      adaptive = false, dt = 0.01f0, save_everystep = false)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0),
                      trajectories = 10_000,
                      adaptive = true, dt = 0.01f0, save_everystep = false)

    ## With random parameters

    prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 10,
                      adaptive = false, dt = 0.1f0)
    asol = solve(monteprob, alg, EnsembleGPUKernel(backend, 0.0), trajectories = 10,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7)
end
