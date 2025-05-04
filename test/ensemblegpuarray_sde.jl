using DiffEqGPU, StochasticDiffEq, Test

include("utils.jl")

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

function multiplicative_noise(du, u, p, t)
    du[1] = 0.1 * u[1]
    du[2] = 0.1 * u[2]
    du[3] = 0.1 * u[3]
end

u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 10.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = SDEProblem(lorenz, multiplicative_noise, u0, tspan, p)
const pre_p = [rand(Float32, 3) for i in 1:10]
prob_func = (prob, i, repeat) -> remake(prob, p = pre_p[i] .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)

@info "Explicit Methods"

#Performance check with nvvp
# CUDAnative.CUDAdrv.@profile
@time sol = solve(monteprob, SOSRI(), EnsembleGPUArray(backend), trajectories = 10,
    saveat = 1.0f0)

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    du[4] = 0
end

function multiplicative_noise(du, u, p, t)
    du[1, 1] = 0.1
    du[2, 2] = 0.4
    du[4, 1] = 1.0
end

NRate = spzeros(4, 2)
NRate[1, 1] = 1
NRate[4, 1] = 1
NRate[2, 2] = 1

u0 = ComplexF32[1.0; 0.0; 0.0; 0.0]
tspan = (0.0f0, 10.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = SDEProblem(lorenz, multiplicative_noise, u0, tspan, p, noise_rate_prototype=NRate)

prob_func = (prob, i, repeat) -> remake(prob, p=p)
monteprob = EnsembleProblem(prob, prob_func=prob_func)

@test_throws "Incompatible problem detected. EnsembleGPUArray currently requires `prob.noise_rate_prototype === nothing`, i.e. only diagonal noise is currently supported. Track https://github.com/SciML/DiffEqGPU.jl/issues/331 for more information." sol = solve(monteprob, SRA1(), EnsembleCPUArray(), trajectories=10_000, saveat=1.0f0)
