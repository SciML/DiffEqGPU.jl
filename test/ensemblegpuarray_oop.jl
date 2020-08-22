using DiffEqGPU, OrdinaryDiffEq, StaticArrays

function lorenz(u,p,t)
 @inbounds begin
     du1 = p[1]*(u[2]-u[1])
     du2 = u[1]*(p[2]-u[3]) - u[2]
     du3 = u[1]*u[2] - p[3]*u[3]
     SA[du1,du2,du3]
 end
end

u0 = SA[1f0;0f0;0f0]
tspan = (0.0f0,100.0f0)
p = SA[10.0f0,28.0f0,8/3f0]
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
