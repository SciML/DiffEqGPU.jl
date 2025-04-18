# GPU Ensemble Simulation with Random Decay Rates

In this tutorial, we demonstrate how to perform GPU-accelerated ensemble simulations using DiffEqGPU.jl. We model an exponential decay ODE:
\[ u'(t) = -\lambda \, u(t) \]
with the twist that each trajectory uses a random decay rate \(\lambda\) sampled uniformly from \([0.5, 1.5]\).

## Setup

We first define the ODE and set up an `EnsembleProblem` that randomizes the decay rate for each trajectory.

```julia
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


