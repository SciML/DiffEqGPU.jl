using Test
using JET
using DiffEqGPU
using StaticArrays

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
    end

    @testset "Utility functions type stability" begin
        # Test diffeqgpunorm with different input types
        @test_opt DiffEqGPU.diffeqgpunorm([1.0f0, 2.0f0, 3.0f0], 0.0f0)
        @test_opt DiffEqGPU.diffeqgpunorm(1.0f0, 0.0f0)
        @test_opt DiffEqGPU.diffeqgpunorm(SA[1.0f0, 2.0f0, 3.0f0], 0.0f0)
    end
end
