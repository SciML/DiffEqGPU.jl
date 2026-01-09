# DiffEqGPU Allocation Tests
# These tests verify that critical inner-loop functions do not allocate

using Test
using StaticArrays
using DiffEqGPU

# Note: AllocCheck.@check_allocs is not compatible with GPU kernels and complex
# dispatch, so we use @allocated instead for testing allocation counts.

# Test Lorenz system for allocation tests
function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

@testset "Allocation Tests" begin
    @testset "User-defined function should not allocate with SVector" begin
        u = @SVector [1.0f0, 0.0f0, 0.0f0]
        p = @SVector [10.0f0, 28.0f0, 8.0f0 / 3.0f0]
        t = 0.0f0

        # Warmup
        lorenz(u, p, t)

        # Test allocations
        allocs = @allocated lorenz(u, p, t)
        @test allocs == 0
    end

    @testset "make_prob_compatible should be low allocation" begin
        using OrdinaryDiffEq

        u0 = @SVector [1.0f0, 0.0f0, 0.0f0]
        tspan = (0.0f0, 10.0f0)
        p = @SVector [10.0f0, 28.0f0, 8.0f0 / 3.0f0]
        prob = ODEProblem{false}(lorenz, u0, tspan, p)

        # Warmup
        DiffEqGPU.make_prob_compatible(prob)

        # Test - some allocations are expected for problem conversion
        allocs = @allocated DiffEqGPU.make_prob_compatible(prob)
        # Should be reasonably low (less than 1KB)
        @test allocs < 1024
    end

    @testset "diffeqgpunorm should not allocate for SVector" begin
        u = @SVector [1.0f0, 2.0f0, 3.0f0]
        t = 0.0f0

        # Warmup
        DiffEqGPU.diffeqgpunorm(u, t)

        # Test allocations
        allocs = @allocated DiffEqGPU.diffeqgpunorm(u, t)
        @test allocs == 0
    end

    @testset "diffeqgpunorm should not allocate for scalars" begin
        u = 3.14f0
        t = 0.0f0

        # Warmup
        DiffEqGPU.diffeqgpunorm(u, t)

        # Test allocations
        allocs = @allocated DiffEqGPU.diffeqgpunorm(u, t)
        @test allocs == 0
    end
end
