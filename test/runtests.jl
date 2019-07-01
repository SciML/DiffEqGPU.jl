using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test

function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = 10.0f0(u[2]-u[1])
     du[2] = u[1]*(28.0f0-u[3]) - u[2]
     du[3] = u[1]*u[2] - (8/3f0)*u[3]
 end
 nothing
end

CuArrays.allowscalar(false)
u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
prob = ODEProblem(lorenz,u0,tspan)
monteprob = EnsembleProblem(prob)

#Performance check with nvvp
# CUDAnative.CUDAdrv.@profile
@time solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,
                                                 batch_size=50_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleCPUArray(),trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleThreads(), trajectories=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),EnsembleSerial(),  trajectories=100_000,saveat=1.0f0)
