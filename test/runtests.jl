using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test

function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = p[1]*(u[2]-u[1])
     du[2] = u[1]*(p[2]-u[3]) - u[2]
     du[3] = u[1]*u[2] - p[3]*u[3]
 end
 nothing
end

CuArrays.allowscalar(false)
u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = (10.0f0,28.0f0,8/3f0)
prob = ODEProblem(lorenz,u0,tspan,p)
const pre_p = [rand(Float32,3) for i in 1:100_000]
prob_func = (prob,i,repeat) -> remake(prob,p=pre_p[i].*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)

#Performance check with nvvp
# CUDAnative.CUDAdrv.@profile
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,saveat=1.0f0)
@test length(filter(x -> x.u != sol.u[1].u, sol.u)) != 0 # 0 element array
@time sol = solve(monteprob,ROCK4(),EnsembleGPUArray(),trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,
                                                 batch_size=50_000,saveat=1.0f0)
@test length(filter(x -> x.u != sol.u[1].u, sol.u)) != 0 # 0 element array
@time solve(monteprob,Tsit5(),EnsembleCPUArray(),trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleThreads(), trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleSerial(),  trajectories=100_000,saveat=1.0f0)


solve(monteprob,TRBDF2(),EnsembleCPUArray(),dt=0.1,trajectories=2,saveat=1.0f0)
solve(monteprob,TRBDF2(),EnsembleGPUArray(),dt=0.1,trajectories=2,saveat=1.0f0)
@test_broken solve(monteprob,TRBDF2(linsolve=LinSolveGPUSplitFactorize()),EnsembleGPUArray(),dt=0.1,trajectories=2,saveat=1.0f0)

function lorenz_jac(J,u,p,t)
 @inbounds begin
     σ = p[1]
     ρ = p[2]
     β = p[3]
     x = u[1]
     y = u[2]
     z = u[3]
     J[1,1] = -σ
     J[2,1] = ρ - z
     J[3,1] = y
     J[1,2] = σ
     J[2,2] = -1
     J[3,2] = x
     J[1,3] = 0
     J[2,3] = -x
     J[3,3] = -β
 end
 nothing
end

func = ODEFunction(lorenz,jac=lorenz_jac)
prob_jac = ODEProblem(func,u0,tspan,p)
monteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)

@time solve(monteprob_jac,TRBDF2(),EnsembleCPUArray(),dt=0.1,trajectories=2,saveat=1.0f0)
@time solve(monteprob_jac,TRBDF2(),EnsembleGPUArray(),dt=0.1,trajectories=100,saveat=1.0f0)

condition = function (u,t,integrator)
    @inbounds u[1] > 5
end

affect! = function (integrator)
    @inbounds integrator.u[1] = -4
end

callback_prob = ODEProblem(lorenz,u0,tspan,p,callback=DiscreteCallback(condition,affect!,save_positions=(false,false)))
callback_monteprob = EnsembleProblem(callback_prob, prob_func = prob_func)
solve(callback_monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100,saveat=1.0f0)

c_condition = function (u,t,integrator)
    @inbounds u[1] - 3
end

c_affect! = function (integrator)
    @inbounds integrator.u[1] += 20
end

callback_prob = ODEProblem(lorenz,u0,tspan,p,callback=ContinuousCallback(c_condition,c_affect!,save_positions=(false,false)))
callback_monteprob = EnsembleProblem(callback_prob, prob_func = prob_func)
CuArrays.@allowscalar solve(callback_monteprob,Tsit5(),EnsembleGPUArray(),trajectories=2,saveat=1.0f0)
@test_broken solve(callback_monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10,saveat=1.0f0)[1].retcode == :Success
