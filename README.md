# DiffEqGPU

[![GitlabCI](https://gitlab.com/JuliaGPU/DiffEqGPU-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/DiffEqGPU-jl/pipelines)

This library is a component package of the DifferentialEquations.jl ecosystem. It includes functionality for making
use of GPUs in the differential equation solvers.

## Within-Method GPU Parallelism with Direct CuArray Usage

The native Julia libraries, including (but not limited to) OrdinaryDiffEq, StochasticDiffEq, and DelayDiffEq, are
compatible with `u0` being a `CuArray`. When this occurs, all array operations take place on the GPU, including
any implicit solves. This is independent of the DiffEqGPU library. These speedup the solution of a differential
equation which is sufficiently large or expensive. This does not require DiffEqGPU.jl.

For example, the following is a GPU-accelerated solve with `Tsit5`:

```julia
using OrdinaryDiffEq, CuArrays, LinearAlgebra
u0 = cu(rand(1000))
A  = cu(randn(1000,1000))
f(du,u,p,t)  = mul!(du,A,u)
prob = ODEProblem(f,u0,(0.0f0,1.0f0)) # Float32 is better on GPUs!
sol = solve(prob,Tsit5())
```

## Parameter-Parallelism with GPU Ensemble Methods

Parameter-parallel GPU methods are provided for the case where a single solve is too cheap to benefit from
within-method parallelism, but the solution of the same structure (same `f`) is required for very many
different choices of `u0` or `p`. For these cases, DiffEqGPU exports the following ensemble algorithms:

- `EnsembleGPUArray`: Utilizes the CuArray setup to parallelize ODE solves across the GPU.
- `EnsembleCPUArray`: A test version for analyzing the overhead of the array-based parallelism setup.

For more information on using the ensemble interface, see
[the DiffEqDocs page on ensembles](http://docs.juliadiffeq.org/dev/features/ensemble)

For example, the following solves the Lorenz equation with 10,000 separate random parameters on the GPU:

```julia
function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = p[1]*(u[2]-u[1])
     du[2] = u[1]*(p[2]-u[3]) - u[2]
     du[3] = u[1]*u[2] - p[3]*u[3]
 end
 nothing
end

u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = [10.0f0,28.0f0,8/3f0]
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
```

#### Current Support

Automated GPU parameter parallelism support is continuing to be improved, so there are currently a few limitations.
Not everything is supported yet, but most of the standard features have support, including:

- Explicit Runge-Kutta methods
- Implicit Runge-Kutta methods
- Rosenbrock methods
- DiscreteCallbacks and ContinuousCallbacks
- Multiple GPUs over clusters

#### Current Limitations

If you receive a compilation error, it is likely because something is not allowed by the automated
kernel building of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). The most common issues are:

- Bounds checking is not allowed
- Return values are not allowed

Thus you have to make sure your derivative function wraps the whole thing in `@inbounds` and explicitly does `return nothing`,
like:

```julia
function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = p[1]*(u[2]-u[1])
     du[2] = u[1]*(p[2]-u[3]) - u[2]
     du[3] = u[1]*u[2] - p[3]*u[3]
 end
 nothing
end
```

This is a current limitation that will be fixed over time.

Another detail is that stiff ODEs require the analytical solution of every derivative function it requires. For example,
Rosenbrock methods require the Jacobian and the gradient with respect to time, and so these two functions are required to
be given. Note that they can be generated by the
[modelingtoolkitize](https://docs.juliadiffeq.org/latest/tutorials/advanced_ode_example/#Automatic-Derivation-of-Jacobian-Functions-1)
approach. In addition to supplying the derivative functions, it is required that one sets the linear solver via
`linsolve=LinSolveGPUSplitFactorize()`. For example, 10,000 trajectories solved with Rodas5 and TRBDF2 is done via:

```julia
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

function lorenz_tgrad(J,u,p,t)
 nothing
end

func = ODEFunction(lorenz,jac=lorenz_jac,tgrad=lorenz_tgrad)
prob_jac = ODEProblem(func,u0,tspan,p)
monteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)

@time solve(monteprob_jac,Rodas5(linsolve=LinSolveGPUSplitFactorize()),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)
@time solve(monteprob_jac,TRBDF2(linsolve=LinSolveGPUSplitFactorize()),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)
```

These limitations are not fundamental and will be eased over time.

#### Setting Up Multi-GPU

To setup a multi-GPU environment, first setup a processes such that every process
has a different GPU. For example:

```julia
# Setup processes with different CUDA devices
using Distributed
addprocs(numgpus)
import CUDAdrv, CUDAnative

let gpuworkers = asyncmap(collect(zip(workers(), CUDAdrv.devices()))) do (p, d)
  remotecall_wait(CUDAnative.device!, p, d)
  p
end
```

Then setup the calls to work with distributed processes:

```julia
@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random

@everywhere begin
    function lorenz_distributed(du,u,p,t)
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
    p = [10.0f0,28.0f0,8/3f0]
    Random.seed!(1)
    function prob_func_distributed(prob,i,repeat)
        remake(prob,p=rand(3).*p)
    end
end
```

Now each batch will run on separate GPUs. Thus we need to use the `batch_size`
keyword argument from the Ensemble interface to ensure there are multiple batches.
Let's solve 40,000 trajectories, batching 10,000 trajectories at a time:

```julia
prob = ODEProblem(lorenz_distributed,u0,tspan,p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)

@time sol2 = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=40_000,
                                                 batch_size=10_000,saveat=1.0f0)
```

This will `pmap` over the batches, and thus if you have 4 processes each with
a GPU, each batch of 10,000 trajectories will be run simultaneously. If you have
two processes with two GPUs, this will do two sets of 10,000 at a time.

#### Example Multi-GPU Script

In this example we know we have a 2-GPU system (1 eGPU), and we split the work
across the two by directly defining the devices on the two worker processes:

```julia
using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test
CuArrays.device!(0)

using Distributed
addprocs(2)
@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random

@everywhere begin
    function lorenz_distributed(du,u,p,t)
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
    p = [10.0f0,28.0f0,8/3f0]
    Random.seed!(1)
    pre_p_distributed = [rand(Float32,3) for i in 1:100_000]
    function prob_func_distributed(prob,i,repeat)
        remake(prob,p=pre_p_distributed[i].*p)
    end
end

@sync begin
    @spawnat 2 begin
        CuArrays.allowscalar(false)
        CuArrays.device!(0)
    end
    @spawnat 3 begin
        CuArrays.allowscalar(false)
        CuArrays.device!(1)
    end
end

CuArrays.allowscalar(false)
prob = ODEProblem(lorenz_distributed,u0,tspan,p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)

@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,batch_size=50_000,saveat=1.0f0)
```

#### Optimal Numbers of Trajectories

There is a balance between two things for choosing the number of trajectories:

- The number of trajectories needs to be high enough that the work per kernel
  is sufficient to overcome the kernel call cost.
- More trajectories means that every trajectory will need more time steps since
  the adaptivity syncs all solves.

From our testing, the balance is found at around 10,000 trajectories being optimal.
Thus for larger sets of trajectories, use a batch size of 10,000. Of course,
benchmark for yourself on your own setup!
