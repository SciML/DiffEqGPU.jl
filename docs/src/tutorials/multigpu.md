# [Setting Up Multi-GPU Parallel Parameter Sweeps](@id multigpu)

!!! note

    This tutorial assumes one already has familiarity with EnsembleGPUArray and
    EnsembleGPUKernel. Please see [the Lorenz equation tutorial](@ref lorenz) before
    reading this tutorial!

In this tutorial, we will show how to increase the number of trajectories that can be
computed in parallel by setting up and using a multi-GPU solve. For this, we will set up
one Julia process for each GPU and let the internal `pmap` system of `EnsembleGPUArray`
parallelize the system across all of our GPUs. Let's dig in.

## Setting Up a Multi-GPU Julia Environment

To set up a multi-GPU environment, first set up processes such that each process
has a different GPU. For example:

```julia
# Setup processes with different CUDA devices
using Distributed
numgpus = 1
addprocs(numgpus)
import CUDA

let gpuworkers = asyncmap(collect(zip(workers(), CUDA.devices()))) do (p, d)
        remotecall_wait(CUDA.device!, p, d)
        p
    end
end
```

Then set up the calls to work with distributed processes:

```@example multi
@everywhere using DiffEqGPU, CUDA, OrdinaryDiffEq, Test, Random

@everywhere begin
    function lorenz_distributed(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    CUDA.allowscalar(false)
    u0 = Float32[1.0; 0.0; 0.0]
    tspan = (0.0f0, 100.0f0)
    p = [10.0f0, 28.0f0, 8 / 3.0f0]
    Random.seed!(1)
    function prob_func_distributed(prob, i, repeat)
        remake(prob, p = rand(3) .* p)
    end
end
```

Now each batch will run on separate GPUs. Thus, we need to use the `batch_size`
keyword argument from the Ensemble interface to ensure there are multiple batches.
Let's solve 40,000 trajectories, batching 10,000 trajectories at a time:

```julia
prob = ODEProblem(lorenz_distributed, u0, tspan, p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)

@time sol2 = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()), trajectories = 40_000,
                   batch_size = 10_000, saveat = 1.0f0)
```

This will `pmap` over the batches, and thus if you have 4 processes each with
a GPU, each batch of 10,000 trajectories will be run simultaneously. If you have
two processes with two GPUs, this will do two sets of 10,000 at a time.

## Example Multi-GPU Script

In this example, we know we have a 2-GPU system (1 eGPU), and we split the work
across the two by directly defining the devices on the two worker processes:

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq, Test
CUDA.device!(0)

using Distributed
addprocs(2)
@everywhere using DiffEqGPU, CUDA, OrdinaryDiffEq, Test, Random

@everywhere begin
    function lorenz_distributed(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    CUDA.allowscalar(false)
    u0 = Float32[1.0; 0.0; 0.0]
    tspan = (0.0f0, 100.0f0)
    p = [10.0f0, 28.0f0, 8 / 3.0f0]
    Random.seed!(1)
    pre_p_distributed = [rand(Float32, 3) for i in 1:100_000]
    function prob_func_distributed(prob, i, repeat)
        remake(prob, p = pre_p_distributed[i] .* p)
    end
end

@sync begin
    @spawnat 2 begin
        CUDA.allowscalar(false)
        CUDA.device!(0)
    end
    @spawnat 3 begin
        CUDA.allowscalar(false)
        CUDA.device!(1)
    end
end

CUDA.allowscalar(false)
prob = ODEProblem(lorenz_distributed, u0, tspan, p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)

@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()), trajectories = 100_000,
                  batch_size = 50_000, saveat = 1.0f0)
```
