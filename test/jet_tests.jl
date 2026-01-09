using Test
using JET
using DiffEqGPU
using StaticArrays
using SciMLBase
using ForwardDiff

@testset "JET static analysis" begin
    # Test that algorithm constructors are type-stable
    @testset "Algorithm constructors" begin
        # Explicit algorithms
        @test_opt GPUTsit5()
        @test_opt GPUVern7()
        @test_opt GPUVern9()

        # SDE algorithms
        @test_opt GPUEM()
        @test_opt GPUSIEA()

        # Implicit algorithms (with default AD)
        @test_opt GPURosenbrock23()
        @test_opt GPURodas4()
        @test_opt GPURodas5P()
        @test_opt GPUKvaerno3()
        @test_opt GPUKvaerno5()

        # Implicit algorithms with explicit autodiff setting
        @test_opt GPURosenbrock23(; autodiff = Val{true}())
        @test_opt GPURosenbrock23(; autodiff = Val{false}())
        @test_opt GPURodas4(; autodiff = Val{true}())
        @test_opt GPUKvaerno5(; autodiff = Val{false}())
    end

    @testset "Ensemble algorithm constructors" begin
        @test_opt EnsembleCPUArray()
    end

    @testset "alg_order type stability" begin
        @test_opt DiffEqGPU.alg_order(GPUTsit5())
        @test_opt DiffEqGPU.alg_order(GPUVern7())
        @test_opt DiffEqGPU.alg_order(GPUVern9())
        @test_opt DiffEqGPU.alg_order(GPUEM())
        @test_opt DiffEqGPU.alg_order(GPUSIEA())
        # Implicit algorithm orders
        @test_opt DiffEqGPU.alg_order(GPURosenbrock23())
        @test_opt DiffEqGPU.alg_order(GPURodas4())
        @test_opt DiffEqGPU.alg_order(GPURodas5P())
        @test_opt DiffEqGPU.alg_order(GPUKvaerno3())
        @test_opt DiffEqGPU.alg_order(GPUKvaerno5())
    end

    @testset "alg_autodiff type stability" begin
        @test_opt DiffEqGPU.alg_autodiff(GPURosenbrock23())
        @test_opt DiffEqGPU.alg_autodiff(GPURodas4())
        @test_opt DiffEqGPU.alg_autodiff(GPURodas5P())
        @test_opt DiffEqGPU.alg_autodiff(GPUKvaerno3())
        @test_opt DiffEqGPU.alg_autodiff(GPUKvaerno5())
    end

    @testset "Utility functions type stability" begin
        # Test diffeqgpunorm with different input types
        @test_opt DiffEqGPU.diffeqgpunorm([1.0f0, 2.0f0, 3.0f0], 0.0f0)
        @test_opt DiffEqGPU.diffeqgpunorm(1.0f0, 0.0f0)
        @test_opt DiffEqGPU.diffeqgpunorm(SA[1.0f0, 2.0f0, 3.0f0], 0.0f0)

        # Test with Float64
        @test_opt DiffEqGPU.diffeqgpunorm([1.0, 2.0, 3.0], 0.0)
        @test_opt DiffEqGPU.diffeqgpunorm(1.0, 0.0)
        @test_opt DiffEqGPU.diffeqgpunorm(SA[1.0, 2.0, 3.0], 0.0)

        # Test with Complex
        @test_opt DiffEqGPU.diffeqgpunorm(1.0 + 2.0im, 0.0)
        @test_opt DiffEqGPU.diffeqgpunorm(1.0f0 + 2.0f0im, 0.0f0)

        # Test with ForwardDiff.Dual types
        @test_opt DiffEqGPU.diffeqgpunorm(ForwardDiff.Dual(1.0f0, 1.0f0), 0.0f0)
        @test_opt DiffEqGPU.diffeqgpunorm(
            SA[ForwardDiff.Dual(1.0f0, 1.0f0), ForwardDiff.Dual(2.0f0, 1.0f0)], 0.0f0)
    end

    @testset "make_prob_compatible type stability" begin
        lorenz(u, p, t) = SA[p[1] * (u[2] - u[1]),
            u[1] * (p[2] - u[3]) - u[2],
            u[1] * u[2] - p[3] * u[3]]
        u0 = SA[1.0f0, 0.0f0, 0.0f0]
        tspan = (0.0f0, 10.0f0)
        p = SA[10.0f0, 28.0f0, 8.0f0 / 3.0f0]
        prob = ODEProblem{false}(lorenz, u0, tspan, p)
        @test_opt DiffEqGPU.make_prob_compatible(prob)
    end

    @testset "Integrator initialization type stability" begin
        f_test(u, p, t) = SA[10.0f0 * (u[2] - u[1]),
            u[1] * (28.0f0 - u[3]) - u[2],
            u[1] * u[2] - 8.0f0 / 3.0f0 * u[3]]
        u0 = SA[1.0f0, 0.0f0, 0.0f0]
        t0 = 0.0f0
        dt = 0.01f0
        p = SA[10.0f0, 28.0f0, 8.0f0 / 3.0f0]
        tstops = (1.0f0, 5.0f0)
        callback = SciMLBase.CallbackSet((), ())
        save_everystep = true
        saveat = ()

        # Fixed-step integrator initialization
        @test_opt DiffEqGPU.init(GPUTsit5(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPUVern7(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPUVern9(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPURosenbrock23(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPURodas4(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPURodas5P(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPUKvaerno3(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)
        @test_opt DiffEqGPU.init(GPUKvaerno5(), f_test, false, u0, t0, dt, p, tstops,
            callback, save_everystep, saveat)

        # Adaptive integrator initialization
        tf = 10.0f0
        abstol = 1.0f-6
        reltol = 1.0f-3
        internalnorm = DiffEqGPU.diffeqgpunorm
        saveat_adaptive = ()

        @test_opt DiffEqGPU.init(GPUTsit5(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPUVern7(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPUVern9(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPURosenbrock23(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPURodas4(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPURodas5P(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPUKvaerno3(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
        @test_opt DiffEqGPU.init(GPUKvaerno5(), f_test, false, u0, t0, tf, dt, p,
            abstol, reltol, internalnorm, tstops, callback, saveat_adaptive)
    end

    @testset "Tableaus type stability" begin
        # Vern7 tableau
        @test_opt DiffEqGPU.Vern7Tableau(Float32, Float32)
        @test_opt DiffEqGPU.Vern7Tableau(Float64, Float64)

        # Vern9 tableau
        @test_opt DiffEqGPU.Vern9Tableau(Float32, Float32)
        @test_opt DiffEqGPU.Vern9Tableau(Float64, Float64)

        # Rodas tableaus
        @test_opt DiffEqGPU.Rodas4Tableau(Float32, Float32)
        @test_opt DiffEqGPU.Rodas4Tableau(Float64, Float64)
        @test_opt DiffEqGPU.Rodas5PTableau(Float32, Float32)
        @test_opt DiffEqGPU.Rodas5PTableau(Float64, Float64)

        # Kvaerno tableaus
        @test_opt DiffEqGPU.Kvaerno3Tableau(Float32, Float32)
        @test_opt DiffEqGPU.Kvaerno3Tableau(Float64, Float64)
        @test_opt DiffEqGPU.Kvaerno5Tableau(Float32, Float32)
        @test_opt DiffEqGPU.Kvaerno5Tableau(Float64, Float64)
    end

    @testset "Adaptive controller cache type stability" begin
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPUTsit5(), Float32)
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPUTsit5(), Float64)
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPUVern7(), Float32)
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPUVern9(), Float32)
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPURosenbrock23(), Float32)
        @test_opt DiffEqGPU.build_adaptive_controller_cache(GPURodas4(), Float32)
    end
end
