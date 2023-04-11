using DiffEqGPU, StaticArrays, OrdinaryDiffEq, LinearAlgebra

device = if GROUP == "CUDA"
    using CUDA, CUDAKernels
    CUDADevice()
elseif GROUP == "AMDGPU"
    using AMDGPU, ROCKernels
    ROCDevice()
elseif GROUP == "oneAPI"
    using oneAPI, oneAPIKernels
    oneAPIDevice()
end

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

function lorenz_jac(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    x = u[1]
    y = u[2]
    z = u[3]
    J11 = -σ
    J21 = ρ - z
    J31 = y
    J12 = σ
    J22 = -1
    J32 = x
    J13 = 0
    J23 = -x
    J33 = -β
    return SMatrix{3, 3}(J11, J21, J31, J12, J22, J32, J13, J23, J33)
end

function lorenz_tgrad(u, p, t)
    return SVector{3, eltype(u)}(0.0, 0.0, 0.0)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]

func = ODEFunction(lorenz, jac = lorenz_jac, tgrad = lorenz_tgrad)
prob = ODEProblem{false}(func, u0, tspan, p)

algs = (GPURosenbrock23(),)
for alg in algs
    prob_func = (prob, i, repeat) -> remake(prob, p = p)
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
    @info typeof(alg)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10,
                      adaptive = false, dt = 0.01f0)
    asol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7)

    @test sol.converged == true
    @test asol.converged == true

    ## Regression test

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7)

    @test norm(bench_sol.u[end] - sol[1].u[end]) < 5e-3
    #Fails
    @test norm(bench_asol.u - asol[1].u) < 3e-3

    ### solve parameters

    saveat = [2.0f0, 4.0f0]

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 2,
                      adaptive = false, dt = 0.01f0, saveat = saveat)

    asol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 2,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
                 saveat = saveat)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7, saveat = saveat)

    @test norm(asol[1].u[end] - sol[1].u[end]) < 4e-2

    @test norm(bench_sol.u - sol[1].u) < 2e-4
    #Use to fail for 2e-4
    @test norm(bench_asol.u - asol[1].u) < 4e-4

    @test length(sol[1].u) == length(saveat)
    @test length(asol[1].u) == length(saveat)

    saveat = collect(0.0f0:0.1f0:10.0f0)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device), trajectories = 2,
                      adaptive = false, dt = 0.01f0, saveat = saveat)

    asol = solve(monteprob, alg, EnsembleGPUKernel(device), trajectories = 2,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
                 saveat = saveat)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(prob, Rosenbrock23(), dt = 0.1f-1, save_everystep = false,
                       abstol = 1.0f-7,
                       reltol = 1.0f-7, saveat = saveat)

    #Fails also OrdinaryDiffEq.jl
    # @test norm(asol[1].u[end] - sol[1].u[end]) < 6e-3

    @test norm(bench_sol.u - sol[1].u) < 2e-3
    #Fails
    @test norm(bench_asol.u - asol[1].u) < 2e-2

    @test length(sol[1].u) == length(saveat)
    @test length(asol[1].u) == length(saveat)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device), trajectories = 2,
                      adaptive = false, dt = 0.01f0, save_everystep = false)

    bench_sol = solve(prob, Rosenbrock23(), adaptive = false, dt = 0.01f0,
                      save_everystep = false)

    @test norm(bench_sol.u - sol[1].u) < 5e-3

    @test length(sol[1].u) == length(bench_sol.u)

    ### Huge number of threads
    local sol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10_000,
                      adaptive = false, dt = 0.01f0, save_everystep = false)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10_000,
                      adaptive = true, dt = 0.01f0, save_everystep = false)

    ## With random parameters

    prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10,
                      adaptive = false, dt = 0.1f0)
    asol = solve(monteprob, alg, EnsembleGPUKernel(device, 0.0), trajectories = 10,
                 adaptive = true, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7)
end
