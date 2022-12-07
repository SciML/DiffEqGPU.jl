# [Massively Data-Parallel ODE Solving the Lorenz Equation](@id lorenz)

For example, the following solves the Lorenz equation with 10,000 separate random parameters on the GPU:

```julia
using DiffEqGPU, OrdinaryDiffEq
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = [10.0f0,28.0f0,8/3f0]
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0);
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0);
```

```julia
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
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = false, dt = 0.1f0);

@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = false, dt = 0.1f0);
```
## Using Stiff ODE Solvers with EnsembleGPUArray

```julia
function lorenz_jac(J,u,p,t)
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

function lorenz_tgrad(J,u,p,t)
    nothing
end

func = ODEFunction(lorenz,jac=lorenz_jac,tgrad=lorenz_tgrad)
prob_jac = ODEProblem(func,u0,tspan,p)
monteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)

@time solve(monteprob_jac,Rodas5(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)
@time solve(monteprob_jac,TRBDF2(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)
```
