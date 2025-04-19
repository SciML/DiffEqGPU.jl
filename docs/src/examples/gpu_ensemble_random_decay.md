# GPU Ensemble Simulation with Random Decay Rates

In this tutorial, we demonstrate how to perform GPU-accelerated ensemble simulations using DiffEqGPU.jl. We model an exponential decay ODE:
\[ u'(t) = -\lambda \, u(t) \]
with the twist that each trajectory uses a random decay rate \(\lambda\) sampled uniformly from \([0.5, 1.5]\).

## Setup

We first define the ODE and set up an `EnsembleProblem` that randomizes the decay rate for each trajectory.

```julia
# Make sure you have the necessary packages installed
# using Pkg
# Pkg.add(["OrdinaryDiffEq", "DiffEqGPU", "CUDA", "Random", "Statistics", "Plots"])
# # Depending on your system, you might need to configure CUDA.jl:
# # import Pkg; Pkg.build("CUDA")
using OrdinaryDiffEq, DiffEqGPU, CUDA, Random, Statistics, Plots

# Set a random seed for reproducibility
Random.seed!(123)

# Define the decay ODE: du/dt = -λ * u, with initial value u(0) = 1.
decay(u, p, t) = -p * u

# Setup initial condition and time span (using Float32 for GPU efficiency)
u0    = 1.0f0
tspan = (0.0f0, 5.0f0)
base_param = 1.0f0
prob = ODEProblem(decay, u0, tspan, base_param)

# Define a probability function that randomizes λ for each ensemble member.
# Each trajectory's λ is sampled uniformly from [0.5, 1.5].
prob_func = (prob, i, repeat) -> begin
    new_λ = 0.5f0 + 1.0f0 * rand()
    remake(prob, p = new_λ)
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
```

# Solving on GPU and CPU

Here we solve the ensemble problem on both GPU and CPU. We use 10,000 trajectories with a fixed time step to facilitate performance comparison.

```julia
# Number of trajectories
num_trajectories = 10_000

# Solve on GPU (check for CUDA availability)
if CUDA.has_cuda()
    @info "Running GPU simulation..."
    gpu_sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                    trajectories = num_trajectories, dt = 0.01f0, adaptive = false)
else
    @warn "CUDA not available. Skipping GPU simulation."
    gpu_sol = nothing
end

# Solve on CPU using multi-threading
@info "Running CPU simulation..."
cpu_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                trajectories = num_trajectories, dt = 0.01f0, adaptive = false)


```

# Performance Comparison

We measure the performance of each simulation. (Note: The first run may include compilation time.)

```julia

# Warm-up (first run) for GPU if applicable
if gpu_sol !== nothing
    @time solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                trajectories = num_trajectories, dt = 0.01f0, adaptive = false)
end

@time cpu_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                      trajectories = num_trajectories, dt = 0.01f0, adaptive = false)
```

# Statistical Analysis and Visualization
We analyze the ensemble by computing the mean and standard deviation of u(t)u(t) across trajectories, and then visualize the results.

```julia

# Assuming all solutions have the same time points (fixed dt & saveat)
t_vals = cpu_sol[1].t
num_times = length(t_vals)
ensemble_vals = reduce(hcat, [sol.u for sol in cpu_sol])  # each column corresponds to one trajectory

# Compute ensemble statistics
mean_u = [mean(ensemble_vals[i, :]) for i in 1:num_times]
std_u  = [std(ensemble_vals[i, :]) for i in 1:num_times]

# Plot the mean trajectory with ±1 standard deviation
p1 = plot(t_vals, mean_u, ribbon = std_u, xlabel = "Time", ylabel = "u(t)",
          title = "Ensemble Mean and ±1σ", label = "Mean ± σ", legend = :topright)

# Histogram of final values (u at t=5)
final_vals = ensemble_vals[end, :]
p2 = histogram(final_vals, bins = 30, xlabel = "Final u", ylabel = "Frequency",
               title = "Distribution of Final Values", label = "")
plot(p1, p2, layout = (1,2), size = (900,400))


