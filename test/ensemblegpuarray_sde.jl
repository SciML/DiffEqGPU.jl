using DiffEqGPU, CuArrays, StochasticDiffEq, Test

function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = p[1]*(u[2]-u[1])
     du[2] = u[1]*(p[2]-u[3]) - u[2]
     du[3] = u[1]*u[2] - p[3]*u[3]
 end
 nothing
end

function multiplicative_noise(du,u,p,t)
 @inbounds begin
     du[1] = 0.1*u[1]
     du[2] = 0.1*u[2]
     du[3] = 0.1*u[3]
 end
 nothing
end

CuArrays.allowscalar(false)
u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,10.0f0)
p = (10.0f0,28.0f0,8/3f0)
prob = SDEProblem(lorenz,multiplicative_noise,u0,tspan,p)
const pre_p = [rand(Float32,3) for i in 1:10]
prob_func = (prob,i,repeat) -> remake(prob,p=pre_p[i].*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)

@info "Explicit Methods"

#Performance check with nvvp
# CUDAnative.CUDAdrv.@profile
@time sol = solve(monteprob,SOSRI(),EnsembleGPUArray(),trajectories=10,saveat=1.0f0)
