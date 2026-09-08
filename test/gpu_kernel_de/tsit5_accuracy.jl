using DiffEqGPU, SciMLBase, StaticArrays, Test
include("../utils.jl")

@testset "Tsit5 nonautonomous accuracy ($T, adaptive=$adaptive)" for
    T in (GROUP in ("CPU", "CUDA", "AMDGPU") ? (Float32, Float64) : (Float32,)),
        adaptive in (false, true)
    rhs(u, p, t) = SVector(p[1] * u[1], cos(t))
    prob = ODEProblem{false}(
        rhs, SVector(T(1), T(0)), (T(0), T(1)), SVector(T(0.5))
    )
    sol = solve(
        EnsembleProblem(prob), GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 3, adaptive, dt = T(0.01),
        abstol = T(1.0e-7), reltol = T(1.0e-6), saveat = T[0, 0.25, 0.5, 1]
    )
    for trajectory in sol.u, (t, u) in zip(trajectory.t, trajectory.u)
        @test u ≈ SVector(exp(T(0.5) * t), sin(t)) rtol = 2.0e-6 atol = 2.0e-7
    end
end
