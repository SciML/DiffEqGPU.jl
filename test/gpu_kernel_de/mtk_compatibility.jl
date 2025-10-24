using DiffEqGPU, Test, ModelingToolkit, OrdinaryDiffEqTsit5, CUDA

@testset "ModelingToolkit GPU Compatibility" begin
    # Test case from issue #375
    # ModelingToolkit-generated problems should work with EnsembleGPUKernel

    @parameters σ ρ β
    @variables t x(t) y(t) z(t)
    D = Differential(t)

    eqs = [D(x) ~ σ * (y - x),
        D(y) ~ x * (ρ - z) - y,
        D(z) ~ x * y - β * z]

    @named sys = ODESystem(eqs, t)
    sys = structural_simplify(sys)

    u0 = [x => 1.0, y => 0.0, z => 0.0]
    tspan = (0.0f0, 1.0f0)
    p = [σ => 10.0, ρ => 28.0, β => 8 / 3]

    prob = ODEProblem(sys, u0, tspan, p)

    # Test that we can create an ensemble problem
    function prob_func(prob, i, repeat)
        remake(prob, p = [σ => 10.0 + i * 0.1, ρ => 28.0, β => 8 / 3])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

    # This should not error with "CuArray only supports element types that are allocated inline"
    @test_nowarn begin
        sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories = 10, dt = 0.1f0, adaptive = false)
    end

    # Actually test that it works
    sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories = 10, dt = 0.1f0, adaptive = false)

    @test length(sol) == 10
    @test all(s.retcode == :Success || s.retcode == :Terminated for s in sol)
end
