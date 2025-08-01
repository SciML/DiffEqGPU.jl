using ModelingToolkit, DiffEqGPU, OrdinaryDiffEq, LinearAlgebra, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using KernelAbstractions: CPU

# ModelingToolkit problems are too complex for GPU array adaptation,
# so we use CPU backend for DAE testing
const backend = CPU()

# Define the cartesian pendulum DAE system
@parameters g = 9.81 L = 1.0
@variables x(t) y(t) [state_priority = 10] λ(t)

# The cartesian pendulum DAE system:
# m*ddot(x) = (x/L)*λ  (simplified with m=1)
# m*ddot(y) = (y/L)*λ - mg  (simplified with m=1) 
# x^2 + y^2 = L^2  (constraint equation)
eqs = [D(D(x)) ~ λ * x / L
       D(D(y)) ~ λ * y / L - g
       x^2 + y^2 ~ L^2]

@named pendulum = ODESystem(eqs, t, [x, y, λ], [g, L])

# Perform structural simplification with index reduction
pendulum_sys = structural_simplify(dae_index_lowering(pendulum))

# Initial conditions: pendulum starts at bottom right position
u0 = [x => 1.0, y => 0.0, λ => -g]  # λ initial guess for tension

# Time span  
tspan = (0.0f0, 1.0f0)

# Create the ODE problem
prob = ODEProblem(pendulum_sys, u0, tspan, Float32[])

# Verify DAE properties
@test SciMLBase.has_initialization_data(prob.f) == true
@test prob.f.mass_matrix !== nothing

# Create ensemble problem for GPU testing
monteprob = EnsembleProblem(prob, safetycopy = false)

# Test with GPURosenbrock23
sol = solve(monteprob, GPURosenbrock23(), EnsembleGPUKernel(backend),
    trajectories = 2,
    dt = 0.01f0,
    adaptive = false)

@test length(sol.u) == 2

# Check constraint satisfaction: x^2 + y^2 ≈ L^2
final_state = sol.u[1][end]
x_final, y_final = final_state[1], final_state[2]
constraint_error = abs(x_final^2 + y_final^2 - 1.0f0)
@test constraint_error < 0.1f0  # Reasonable tolerance for fixed timestep

# Test with GPURodas4
sol2 = solve(monteprob, GPURodas4(), EnsembleGPUKernel(backend),
    trajectories = 2,
    dt = 0.01f0,
    adaptive = false)

@test length(sol2.u) == 2