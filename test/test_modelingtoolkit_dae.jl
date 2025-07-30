using ModelingToolkit, DiffEqGPU, OrdinaryDiffEq
using KernelAbstractions: CPU
using ModelingToolkit: t_nounits as t, D_nounits as D

println("Testing ModelingToolkit DAE support with Cartesian Pendulum...")

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

println("ModelingToolkit DAE problem created successfully!")
println("Number of equations: ", length(equations(pendulum_sys)))
println("Variables: ", unknowns(pendulum_sys))

# Test if the problem has initialization data
if SciMLBase.has_initialization_data(prob.f)
    println("Problem has initialization data: ✓")
else
    println("Problem has initialization data: ✗")
end

# Test mass matrix presence
if prob.f.mass_matrix !== nothing
    println("Problem has mass matrix: ✓")
    println("Mass matrix size: ", size(prob.f.mass_matrix))
else
    println("Problem has mass matrix: ✗")
end

# Create ensemble problem for GPU testing
monteprob = EnsembleProblem(prob, safetycopy = false)

# Test with CPU backend first
println("\nTesting with GPURosenbrock23 on CPU backend...")
try
    sol = solve(monteprob, GPURosenbrock23(), EnsembleGPUKernel(CPU()),
        trajectories = 4,
        dt = 0.01f0,
        adaptive = false)

    println("✓ ModelingToolkit DAE GPU solution successful!")
    println("Number of solutions: ", length(sol.u))
    
    if length(sol.u) > 0
        println("Final state of first trajectory: ", sol.u[1][end])
        
        # Check constraint satisfaction: x^2 + y^2 ≈ L^2
        final_state = sol.u[1][end]
        x_final, y_final = final_state[1], final_state[2]
        constraint_error = abs(x_final^2 + y_final^2 - 1.0f0)
        println("Constraint error |x² + y² - L²|: ", constraint_error)
        
        if constraint_error < 0.1f0  # Reasonable tolerance for fixed timestep
            println("✓ Constraint satisfied within tolerance")
        else
            println("⚠ Constraint violation detected")
        end
    end
    
catch e
    println("✗ ModelingToolkit DAE GPU solution failed: ", e)
    println("Detailed error: ")
    println(sprint(showerror, e, catch_backtrace()))
end

# Test with Rodas4 as well to show mass matrix support
println("\nTesting with GPURodas4 on CPU backend...")
try
    sol = solve(monteprob, GPURodas4(), EnsembleGPUKernel(CPU()),
        trajectories = 4,
        dt = 0.01f0,
        adaptive = false)

    println("✓ ModelingToolkit DAE with GPURodas4 successful!")
    println("Number of solutions: ", length(sol.u))
    
catch e
    println("✗ ModelingToolkit DAE with GPURodas4 failed: ", e)
    println("Error details: ", sprint(showerror, e, catch_backtrace()))
end

println("\nModelingToolkit DAE testing completed!")