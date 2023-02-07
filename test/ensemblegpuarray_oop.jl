using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDAKernels

function lorenz(u, p, t)
    du1 = p[1] * (u[2] - u[1])
    du2 = u[1] * (p[2] - u[3]) - u[2]
    du3 = u[1] * u[2] - p[3] * u[3]
    SA[du1, du2, du3]
end

function lorenz_jac(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    x = u[1]
    y = u[2]
    z = u[3]
    SA[-σ σ 0
       ρ-z -1 -x
       y x -β]
end

function lorenz_tgrad(u, p, t)
    SA[0.0, 0.0, 0.0]
end

func = ODEFunction(lorenz, jac = lorenz_jac, tgrad = lorenz_tgrad)
u0 = SA[1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 100.0f0)
p = SA[10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem(func, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDADevice()), trajectories = 10_000,
                  saveat = 1.0f0)
@test_broken @time sol = solve(monteprob, Rosenbrock23(), EnsembleGPUArray(CUDADevice()),
                               trajectories = 10_000,
                               saveat = 1.0f0)
@test_broken @time sol = solve(monteprob, TRBDF2(), EnsembleGPUArray(CUDADevice()),
                               trajectories = 10_000,
                               saveat = 1.0f0)
