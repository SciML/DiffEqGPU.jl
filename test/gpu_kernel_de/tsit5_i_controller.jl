function controller_integrator(alg, rhs, ::Type{T}; qold = T(1.0e-4)) where {T}
    integ = DiffEqGPU.init(
        alg, rhs, false, SVector(one(T)), zero(T), T(10), T(0.5),
        nothing, T(1.0e-10), T(1.0e-8), nothing, nothing, CallbackSet(), nothing
    )
    integ.qold = qold
    return integ
end

@testset "I-controller convergence beyond Lorenz ($T)" for T in
    (GROUP in ("CPU", "CUDA", "AMDGPU") ? (Float32, Float64) : (Float32,))
    # The slow decay component reaches Float32 roundoff before the tighter tolerance.
    # Keep the coarse solve above that floor so the error-ratio check measures convergence.
    tolerances = T === Float64 ? (T(1.0e-4), T(1.0e-8)) : (T(1.0e-2), T(1.0e-5))
    cases = (
        ((u, p, t) -> SVector(10 * u[2], -10 * u[1]), SVector(one(T), zero(T)), T(10), SVector(cos(T(100)), -sin(T(100)))),
        ((u, p, t) -> SVector(-u[1], -100 * u[2]), SVector(one(T), one(T)), one(T), SVector(exp(-one(T)), exp(T(-100)))),
    )
    for (rhs, u0, tf, exact) in cases
        prob = ODEProblem{false}(rhs, u0, (zero(T), tf))
        errors = map(tolerances) do tol
            sol = solve(
                EnsembleProblem(prob), GPUTsit5IController(), EnsembleGPUKernel(backend, 0.0);
                trajectories = 2, adaptive = true, dt = T(0.1),
                abstol = tol / T(1000), reltol = tol, save_everystep = false
            )
            maximum(abs.(sol.u[1].u[end] - exact))
        end
        @test errors[2] < errors[1] / 10
        @test errors[2] < (T === Float64 ? 1.0e-5 : 1.0e-3)
    end
end

@testset "Tsit5 controller step response ($T)" for T in (Float32, Float64)
    for (alg, growth) in ((GPUTsit5(), T(9)), (GPUTsit5IController(), T(5)))
        integ = controller_integrator(alg, (u, p, t) -> zero(u), T)
        initial_dt = integ.dtnew
        DiffEqGPU.step!(integ, nothing, nothing)
        @test integ.dt == initial_dt
        @test integ.dtnew ≈ growth * initial_dt
    end

    for alg in (GPUTsit5(), GPUTsit5IController())
        first_history = controller_integrator(alg, (u, p, t) -> u, T; qold = T(0.01))
        second_history = controller_integrator(alg, (u, p, t) -> u, T; qold = one(T))
        DiffEqGPU.step!(first_history, nothing, nothing)
        DiffEqGPU.step!(second_history, nothing, nothing)
        @test first_history.dt < T(0.5)
        @test first_history.u ≈ SVector(exp(first_history.t))
        if alg isa GPUTsit5IController
            @test first_history.dtnew == second_history.dtnew
            @test T(0.2) <= first_history.dtnew / first_history.dt <= T(5)
        else
            @test first_history.dtnew != second_history.dtnew
        end
    end
end

@testset "Tsit5 fixed-step equivalence" begin
    rhs(u, p, t) = SVector(10 * (u[2] - u[1]), u[1] * (28 - u[3]) - u[2], u[1] * u[2] - (8.0f0 / 3.0f0) * u[3])
    prob = ODEProblem{false}(rhs, SVector(1.0f0, 0.0f0, 0.0f0), (0.0f0, 1.0f0))
    ensemble = EnsembleProblem(prob)
    results = map((GPUTsit5(), GPUTsit5IController())) do alg
        solve(
            ensemble, alg, EnsembleGPUKernel(backend, 0.0);
            trajectories = 2, adaptive = false, dt = 0.01f0, saveat = Float32[0, 0.5, 1]
        )
    end
    @test all(results[1].u[i].u == results[2].u[i].u for i in 1:2)
    @test_throws ArgumentError solve(
        ensemble, GPUTsit5IController(), EnsembleGPUKernel(backend, 0.5);
        trajectories = 2, adaptive = true, dt = 0.01f0
    )
end
