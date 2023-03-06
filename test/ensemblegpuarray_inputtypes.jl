using OrdinaryDiffEq, DiffEqGPU, ForwardDiff, Test

include("utils.jl")

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = [ForwardDiff.Dual(1.0f0, (1.0, 0.0, 0.0)); ForwardDiff.Dual(0.0f0, (0.0, 1.0, 0.0));
      ForwardDiff.Dual(0.0f0, (0.0, 0.0, 1.0))]
tspan = (0.0f0, 100.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(device), trajectories = 10_000,
                  saveat = 1.0f0)

#=
u0 = [1f0u"m";0u"m";0u"m"]
tspan = (0.0f0u"s",100.0f0u"s")
p = (10.0f0,28.0f0,8/3f0)
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@test_broken sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0u"s")
=#
