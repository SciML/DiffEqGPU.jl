using DiffEqGPU, OrdinaryDiffEq, StaticArrays
function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p=p)
monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUAutonomous(), trajectories=10, adaptive=false, dt=0.1f0)
asol = solve(monteprob, GPUTsit5(), EnsembleGPUAutonomous(), trajectories=10, adaptive=true, dt=0.1f-1)

@test sol.converged == true
@test asol.converged == true