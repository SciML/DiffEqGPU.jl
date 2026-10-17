using DiffEqGPU, SciMLBase, StaticArrays, Test
include("../utils.jl")

@testset "Tsit5 nonautonomous accuracy ($alg, $T, adaptive=$adaptive)" for
    alg in (GPUTsit5(), GPUTsit5IController()),
        T in (GROUP in ("CPU", "CUDA", "AMDGPU") ? (Float32, Float64) : (Float32,)),
        adaptive in (false, true)
    rhs(u, p, t) = SVector(p[1] * u[1], cos(t))
    prob = ODEProblem{false}(
        rhs, SVector(T(1), T(0)), (T(0), T(1)), SVector(T(0.5))
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(backend, 0.0);
        trajectories = 3, adaptive, dt = T(0.01),
        abstol = T(1.0e-7), reltol = T(1.0e-6), saveat = T[0, 0.25, 0.5, 1]
    )
    for trajectory in sol.u, (t, u) in zip(trajectory.t, trajectory.u)
        @test u ≈ SVector(exp(T(0.5) * t), sin(t)) rtol = 2.0e-6 atol = 2.0e-7
    end
end

@testset "Tsit5 short endpoint ($alg, $T, reltol=$tol)" for
    T in (GROUP in ("CPU", "CUDA", "AMDGPU") ? (Float32, Float64) : (Float32,)),
        tol in (1.0e-3, 1.0e-4), alg in (GPUTsit5(), GPUTsit5IController())
    rhs(u, p, t) = SVector(-u[1], -100 * u[2])
    prob = ODEProblem{false}(rhs, SVector(one(T), one(T)), (zero(T), T(0.1)))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(backend, 0.0);
        trajectories = 2, adaptive = true, dt = T(0.1),
        abstol = T(tol / 1000), reltol = T(tol), save_everystep = false
    )
    exact = SVector(exp(-T(0.1)), exp(-T(10)))
    @test all(s -> s.t[end] == T(0.1), sol.u)
    @test all(s -> isapprox(s.u[end], exact; atol = T(2.0e-6), rtol = T(2.0e-6)), sol.u)
end

@testset "Tsit5 overshoot ($alg, $T)" for
    T in (GROUP in ("CPU", "CUDA", "AMDGPU") ? (Float32, Float64) : (Float32,)),
        alg in (GPUTsit5(), GPUTsit5IController())
    rhs(u, p, t) = SVector(-u[1], -100 * u[2])
    prob = ODEProblem{false}(rhs, SVector(one(T), one(T)), (zero(T), T(0.01)))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(backend, 0.0);
        trajectories = 2, adaptive = true, dt = T(0.1),
        abstol = T(1.0e-5), reltol = T(1.0e-2), save_everystep = false
    )
    exact = SVector(exp(-T(0.01)), exp(-T(1)))
    @test all(s -> s.t[end] == T(0.01), sol.u)
    @test all(s -> isapprox(s.u[end], exact; rtol = T(1.0e-2)), sol.u)
end
