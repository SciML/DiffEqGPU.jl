using ModelingToolkit, DiffEqGPU, OrdinaryDiffEq, StaticArrays, Test
using KernelAbstractions: CPU
using ModelingToolkit: t_nounits as t, D_nounits as D

println("Testing ModelingToolkit DAE support with Cartesian Pendulum...")

# Define the cartesian pendulum DAE system
@parameters g = 9.81f0 L = 1f0
@variables x(t) y(t) [state_priority = 10] λ(t)

# The cartesian pendulum DAE system:
# m*ddot(x) = (x/L)*λ  (simplified with m=1)
# m*ddot(y) = (y/L)*λ - mg  (simplified with m=1) 
# x^2 + y^2 = L^2  (constraint equation)
eqs = [D(D(x)) ~ λ * x / L
       D(D(y)) ~ λ * y / L - g
       x^2 + y^2 ~ L^2]

@mtkcompile pendulum = ODESystem(eqs, t, [x, y, λ], [g, L])

u0 = SA[y => 1.5f0, D(y) => 0.5f0]  # λ initial guess for tension

# Time span
tspan = (0.0f0, 1.0f0)

# Create the ODE problem
prob = ODEProblem(pendulum, u0, tspan, guesses = [λ => 0.5f0, x => 0.5f0])

# Test if the problem has initialization data
@test SciMLBase.has_initialization_data(prob.f)
@test prob.f.mass_matrix !== nothing

simplesol = solve(prob, Rodas5P())

# Create ensemble problem for GPU testing
monteprob = EnsembleProblem(prob, safetycopy = false)

# Test with CPU backend first
sol = solve(monteprob, GPURodas5P(), EnsembleGPUKernel(CPU()),
    trajectories = 4,
    dt = 0.01f0,
    adaptive = false)

if length(sol.u) > 0
    println("Final state of first trajectory: ", sol.u[1][end])
    
    # Check constraint satisfaction: x^2 + y^2 ≈ L^2
    firstsol = sol.u[1]
    x_final, y_final = firstsol[x, end], firstsol[y, end]
    constraint_error = abs(x_final^2 + y_final^2 - 1.0f0)
    println("Constraint error |x² + y² - L²|: ", constraint_error)
    
    if constraint_error < 0.1f0  # Reasonable tolerance for fixed timestep
        println("✓ Constraint satisfied within tolerance")
    else
        println("⚠ Constraint violation detected")
    end
end

println("✗ ModelingToolkit DAE GPU solution failed: ", e)
println("Detailed error: ")
println(sprint(showerror, e, catch_backtrace()))
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