# GPU Ensemble Simulation with Random Decay Rates

In this tutorial, we demonstrate how to perform GPU-accelerated ensemble simulations using DiffEqGPU.jl. We model an exponential decay ODE:
\[ u'(t) = -\lambda \, u(t) \]
with the twist that each trajectory uses a random decay rate \(\lambda\) sampled uniformly from \([0.5, 1.5]\). This version uses `StaticArrays` for the state vector, which is often more robust and performant for small ODEs on the GPU.

## Setup

We first load the necessary packages, define the ODE using `StaticArrays`, and set up an `EnsembleProblem` that randomizes the decay rate for each trajectory.

```@example decay
# Make sure you have the necessary packages installed
# using Pkg
# Pkg.add(["OrdinaryDiffEq", "DiffEqGPU", "CUDA",  "StaticArrays", "Random", "Statistics", "Plots"])
# # Depending on your system, you might need to configure CUDA.jl:
# # import Pkg; Pkg.build("CUDA")
using OrdinaryDiffEq, DiffEqGPU, CUDA, StaticArrays, Random, Statistics, Plots

# Set a random seed for reproducibility
Random.seed!(123)

# Define the decay ODE using the OUT-OF-PLACE form for StaticArrays:
# f(u, p, t) should return a new SVector representing the derivative du/dt.
# This form is generally preferred for StaticArrays on the GPU.
function decay_static(u::SVector, p, t)
    λ = p[1] # Parameter is expected as a scalar or single-element container
    return @SVector [-λ * u[1]]
end

# Setup initial condition as a 1-element SVector (Static Array).
# Using StaticArrays explicitly helps the GPU compiler generate efficient, static code.
u0    = @SVector [1.0f0]

# Define time span (using Float32 for GPU efficiency)
tspan = (0.0f0, 5.0f0)

# Define the base parameter (will be overridden by prob_func)
# We wrap it in an SVector to match how parameters might be handled internally,
# though a scalar Float32 often works too. Using SVector can sometimes avoid type issues.
base_param = @SVector [1.0f0]

# Create the ODEProblem using the static function and SVector initial condition
prob = ODEProblem{false}(decay_static, u0, tspan, base_param) # Use {false} for out-of-place

# Define a probability function that randomizes λ for each ensemble member.
# Each trajectory's λ is sampled uniformly from [0.5, 1.5].
prob_func = (prob, i, repeat) -> begin
    new_λ = 0.5f0 + 1.0f0 * rand(Float32)
    remake(prob, p = @SVector [new_λ])

end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
```

# Solving on GPU and CPU

Here we solve the ensemble problem on both GPU and CPU. We use 10,000 trajectories with a fixed time step to facilitate performance comparison. For performance benchmarking, we initially solve without saving every step (save_everystep=false).

```@example decay
# Number of trajectories
num_trajectories = 10_000

# --- GPU Simulation ---
gpu_sol_perf = nothing # Initialize variable for performance run
if CUDA.has_cuda() && CUDA.functional()
    @info "Running GPU simulation (initial run for performance, includes compilation)..."
    # Use EnsembleGPUKernel with the CUDABackend.
    # GPUTsit5 is suitable for non-stiff ODEs on the GPU.
    # save_everystep=false reduces memory overhead and transfer time if only final states are needed.
    gpu_sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                    trajectories = num_trajectories, 
                    save_everystep=false, # Crucial for performance measurement
                    dt = 0.01f0, adaptive = false)
else
    @warn "CUDA not available. Skipping GPU simulation."
    gpu_sol = nothing
end

# --- CPU Simulation ---
@info "Running CPU simulation (initial run for performance, includes compilation)..."
# Use EnsembleThreads for multi-threaded CPU execution.
# Tsit5 is the CPU counterpart to GPUTsit5.
# Match GPU saving options for fair comparison.
cpu_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                trajectories = num_trajectories, 
                save_everystep=false, # Match GPU setting
                dt = 0.01f0, adaptive = false)


```

# Performance Comparison

We re-run the simulations using @time to get a cleaner measurement of the execution time, excluding the initial compilation overhead.

```@example decay

# --- GPU Timing (Second Run) ---
if gpu_sol_perf !== nothing
    @info "Timing GPU simulation (second run, no data saving)..."
    @time solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                trajectories = num_trajectories, save_everystep=false,
                dt = 0.01f0, adaptive = false)
else
    @info "Skipping GPU timing (CUDA not available)."
end

# --- CPU Timing (Second Run) ---
@info "Timing CPU simulation (second run, no data saving)..."
@time cpu_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                      trajectories = num_trajectories, save_everystep=false,
                      dt = 0.01f0, adaptive = false)

# Note: The first @time includes compilation and setup, the second is more representative
# of the pure computation time for subsequent runs. Expect GPU to be significantly
# faster for a large number of trajectories like 10,000.
```

# CPU Statistical Analysis and Visualization
To visualize the evolution of the ensemble statistics (mean and standard deviation) over time using the *CPU results*, we need the solutions at multiple time points. We re-solve the problem on the CPU, this time saving the results at each step (save_everystep=true). We then process the results and plot them.

```@example decay
# Re-solve on CPU, saving all steps for plotting
@info "Re-solving CPU simulation to collect data for plotting..."
cpu_sol_plot = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                     trajectories = num_trajectories,
                     save_everystep=true, # Save data at each dt step
                     dt = 0.01f0,
                     adaptive = false)
                     
# Extract time points from the first trajectory's solution (assuming all are same)
t_vals_cpu = cpu_sol_plot[1].t
num_times_cpu = length(t_vals_cpu)

# Create a matrix to hold the results: rows=time, columns=trajectories
# Initialize with NaN in case some trajectories fail
ensemble_vals_cpu = fill(NaN32, num_times_cpu, num_trajectories) # Use Float32

# Extract the state value (u[1]) from each trajectory at each time point
for i in 1:num_trajectories
    # Check if the trajectory simulation was successful and data looks valid
    if cpu_sol_plot[i].retcode == ReturnCode.Success && length(cpu_sol_plot[i].u) == num_times_cpu
        # sol.u is a Vector{SVector{1, Float32}}. We need the element from each SVector.
        ensemble_vals_cpu[:, i] .= getindex.(cpu_sol_plot[i].u, 1)
    else
        @warn "CPU Trajectory $i failed or had unexpected length. Retcode: $(cpu_sol_plot[i].retcode). Length: $(length(cpu_sol_plot[i].u)). Skipping."
        # Column remains NaN
    end
end

# Filter out failed trajectories (columns with NaN)
successful_traj_indices_cpu = findall(j -> !all(isnan, view(ensemble_vals_cpu, :, j)), 1:num_trajectories)
num_successful_cpu = length(successful_traj_indices_cpu)

if num_successful_cpu == 0
    @error "No successful CPU trajectories to analyze!"
else
    if num_successful_cpu < num_trajectories
        @warn "$(num_trajectories - num_successful_cpu) CPU trajectories failed. Analysis based on $num_successful_cpu trajectories."
        ensemble_vals_cpu = ensemble_vals_cpu[:, successful_traj_indices_cpu] # Keep only successful ones
    end

    # Compute ensemble statistics over successful CPU trajectories
    mean_u_cpu = mapslices(mean, ensemble_vals_cpu, dims=2)[:]
    std_u_cpu  = mapslices(std, ensemble_vals_cpu, dims=2)[:]

    # --- Plotting CPU Results ---
    p1_cpu = plot(t_vals_cpu, mean_u_cpu, ribbon = std_u_cpu, xlabel = "Time (t)", ylabel = "u(t)",
                  title = "CPU Ensemble Mean ±1σ ($num_successful_cpu Trajectories)",
                  label = "Mean u(t)", fillalpha = 0.3, lw=2)
   

    final_vals_cpu = ensemble_vals_cpu[end, :]
    p2_cpu = histogram(final_vals_cpu, bins = 30, normalize=:probability,
                       xlabel = "Final u(T)", ylabel = "Probability Density",
                       title = "CPU Distribution of Final Values (t=$(tspan[2]))",
                       label = "", legend=false)

    plot_cpu = plot(p1_cpu, p2_cpu, layout = (1,2), size = (1000, 450), legend=:outertopright)
    @info "Displaying CPU analysis plot..."
    display(plot_cpu)
end
```

# GPU Statistical Analysis and Visualization
Similarly, we can analyze the results from the *GPU simulation*. This requires re-running the simulation to save the time series data and then transferring the data from the GPU memory to the CPU RAM for analysis and plotting using standard tools. Note that this data transfer can be a significant bottleneck for large numbers of trajectories or time steps.

```@example decay
# Check if GPU simulation was successful initially before proceeding
gpu_analysis_plot = nothing # Initialize plot variable
if gpu_sol_perf !== nothing && CUDA.has_cuda() && CUDA.functional()
    @info "Re-solving GPU simulation to collect data for plotting..."
    # Add @time to see the impact of saving data
    @time gpu_sol_plot = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                           trajectories = num_trajectories,
                           save_everystep=true, # <<-- Save data at each dt step on GPU
                           dt = 0.01f0,
                           adaptive = false)

    # --- Data Transfer and Analysis ---
    # The result gpu_sol_plot should be an EnsembleSolution containing a Vector{ODESolution}
    # Accessing it might implicitly transfer, or we can use Array()
    @info "Transferring GPU solution objects (if needed) and processing..."
    # Let's try accessing .u directly first, assuming it holds the Vector{ODESolution}
    # If this fails, we might try Array(gpu_sol_plot) -> Vector{ODESolution}
    solutions_vector = gpu_sol_plot.u

    # Check if the transfer actually happened / if we have the right type
    if !(solutions_vector isa AbstractVector{<:ODESolution})
        @warn "gpu_sol_plot.u is not a Vector{ODESolution}. Trying Array(gpu_sol_plot)..."
        # This might explicitly trigger the transfer and construction of ODESolution objects on CPU
        # Note: This might be slow/memory intensive!
        @time solutions_vector = Array(gpu_sol_plot)
        if !(solutions_vector isa AbstractVector{<:ODESolution})
             @error "Could not obtain Vector{ODESolution} from GPU result. Type is $(typeof(solutions_vector)). Aborting GPU analysis."
             solutions_vector = nothing # Mark as failed
        end
    end

    if solutions_vector !== nothing
        # Extract time points from the first successful trajectory's solution
        first_successful_gpu_idx = findfirst(sol -> sol.retcode == ReturnCode.Success, solutions_vector)

        if first_successful_gpu_idx === nothing
            @error "No successful GPU trajectories found in the returned solutions vector!"
        else
            t_vals_gpu = solutions_vector[first_successful_gpu_idx].t
            num_times_gpu = length(t_vals_gpu)

            # Create a matrix to hold the results from GPU (now on CPU)
            ensemble_vals_gpu = fill(NaN32, num_times_gpu, num_trajectories) # Use Float32

            # Extract the state value (u[1]) from each trajectory
            num_processed = 0
            for i in 1:num_trajectories
                sol = solutions_vector[i] # Access the i-th ODESolution
                if sol.retcode == ReturnCode.Success
                   # Check consistency of time points (optional but good)
                   if length(sol.t) == num_times_gpu # && sol.t == t_vals_gpu (can be slow check)
                        # sol.u is likely Vector{SVector{1, Float32}} after transfer
                        ensemble_vals_gpu[:, i] .= getindex.(sol.u, 1)
                        num_processed += 1
                   else
                       @warn "GPU Trajectory $i succeeded but time points mismatch (Expected $(num_times_gpu), Got $(length(sol.t))). Skipping."
                       # Column remains NaN
                   end
                else
                    # @warn "GPU Trajectory $i failed with retcode: $(sol.retcode). Skipping." # Potentially verbose
                    # Column remains NaN
                end
            end
            @info "Processed $num_processed successful GPU trajectories."

            # Filter out failed trajectories (columns with NaN)
            successful_traj_indices_gpu = findall(j -> !all(isnan, view(ensemble_vals_gpu, :, j)), 1:num_trajectories)
            num_successful_gpu = length(successful_traj_indices_gpu)

            if num_successful_gpu == 0
                @error "No successful GPU trajectories suitable for analysis after processing!"
            else
                if num_successful_gpu < num_trajectories
                    # This count includes those skipped due to time mismatch or failure
                    @warn "Analysis based on $num_successful_gpu trajectories (out of $num_trajectories initial)."
                    # Keep only successful, valid ones
                    ensemble_vals_gpu = ensemble_vals_gpu[:, successful_traj_indices_gpu]
                end

                # Compute ensemble statistics over successful GPU trajectories
                mean_u_gpu = mapslices(mean, ensemble_vals_gpu, dims=2)[:]
                std_u_gpu  = mapslices(std, ensemble_vals_gpu, dims=2)[:]

                # --- Plotting GPU Results ---
                p1_gpu = plot(t_vals_gpu, mean_u_gpu, ribbon = std_u_gpu, xlabel = "Time (t)", ylabel = "u(t)",
                              title = "GPU Ensemble Mean ±1σ ($num_successful_gpu Trajectories)",
                              label = "Mean u(t)", fillalpha = 0.3, lw=2)

                final_vals_gpu = ensemble_vals_gpu[end, :]
                p2_gpu = histogram(final_vals_gpu, bins = 30, normalize=:probability,
                                   xlabel = "Final u(T)", ylabel = "Probability Density",
                                   title = "GPU Distribution of Final Values (t=$(tspan[2]))",
                                   label = "", legend=false)

                gpu_analysis_plot = plot(p1_gpu, p2_gpu, layout = (1,2), size = (1000, 450), legend=:outertopright)
                @info "Displaying GPU analysis plot..."
                display(gpu_analysis_plot)

                # Cleanup large structures if memory is a concern
                 ensemble_vals_gpu = nothing
                 solutions_vector = nothing
                # gc()
            end
        end
    end # End if solutions_vector !== nothing
else
    @warn "Skipping GPU analysis section because initial GPU performance run failed or CUDA is unavailable."
end
```

