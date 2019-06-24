using DiffEqGPU, OrdinaryDiffEq, Test

#Performance check with nvvp

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
prob = ODEProblem(lorenz,u0,tspan)
monteprob = MonteCarloProblem(prob)

# CUDAnative.CUDAdrv.@profile
@time solve(monteprob,Tsit5(),MonteGPUArray(),num_monte=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),MonteThreads(), num_monte=100_000,saveat=1.0f0)
@time solve(monteprob,Tsit5(),MonteSerial(), num_monte=100_000,saveat=1.0f0)
