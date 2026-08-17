using Adapt
using DiffEqGPU
using KernelAbstractions: CPU
using OrdinaryDiffEq: Tsit5
using SciMLBase: EnsembleProblem, ImmutableODEProblem, ODEProblem, SDEProblem, solve
using StaticArrays: SVector, @SVector
using Test

function rhs(u, p, t)
    return SVector(u[1])
end

ode_prob = ODEProblem{false}(
    rhs, @SVector([1.0f0]), (0.0f0, 0.2f0), @SVector([1.0f0])
)
compatible_prob = DiffEqGPU.make_prob_compatible(ode_prob)
@test compatible_prob isa ImmutableODEProblem
ode_probs = [compatible_prob for _ in 1:2]
cpu_probs = adapt(CPU(), ode_probs)

@test DiffEqGPU.EnsembleCPUArray() isa DiffEqGPU.EnsembleArrayAlgorithm
@test DiffEqGPU.EnsembleGPUKernel(CPU()) isa DiffEqGPU.EnsembleKernelAlgorithm
@test DiffEqGPU.GPUTsit5() isa DiffEqGPU.GPUODEAlgorithm
@test DiffEqGPU.GPUEM() isa DiffEqGPU.GPUSDEAlgorithm

@testset "generic lower-level ODE interface" begin
    ts, us = DiffEqGPU.vectorized_solve(
        cpu_probs, ode_prob, DiffEqGPU.GPUTsit5();
        dt = 0.1f0, save_everystep = false
    )
    @test size(ts) == (2, 2)
    @test size(us) == (2, 2)
    @test all(u -> u == @SVector([1.0f0]), us[1, :])

    ats, aus = DiffEqGPU.vectorized_asolve(
        cpu_probs, ode_prob, DiffEqGPU.GPUTsit5();
        dt = 0.1f0, saveat = 0.1f0, save_everystep = false
    )
    @test size(ats, 2) == 2
    @test size(aus, 2) == 2
    @test first(aus[:, 1]) == @SVector([1.0f0])
end

@testset "generic lower-level SDE interface" begin
    sde_drift(u, p, t) = u
    sde_noise(u, p, t) = u
    sde_prob = SDEProblem{false}(
        sde_drift, sde_noise, @SVector([1.0f0]), (0.0f0, 0.2f0), @SVector([1.0f0])
    )
    sde_probs = adapt(CPU(), [sde_prob for _ in 1:2])

    sts, sus = DiffEqGPU.vectorized_solve(
        sde_probs, sde_prob, DiffEqGPU.GPUEM();
        dt = 0.1f0, save_everystep = false
    )
    @test size(sts) == (2, 2)
    @test size(sus) == (2, 2)
    @test all(x -> isfinite(x[1]), sus)
end

@testset "generic array ensemble interface" begin
    sols = DiffEqGPU.vectorized_map_solve(
        ode_probs, Tsit5(), DiffEqGPU.EnsembleCPUArray(), 1:2, false;
        dt = 0.1f0, save_everystep = false, dense = false
    )
    @test length(sols.t) == length(sols.u)
    @test size(first(sols.u), 2) == 2
    @test first(sols.u)[:, 1] == @SVector([1.0f0])
end

@testset "generic high-level kernel interface" begin
    ensemble_prob = EnsembleProblem(ode_prob)
    sol = solve(
        ensemble_prob, DiffEqGPU.GPUTsit5(),
        DiffEqGPU.EnsembleGPUKernel(CPU());
        trajectories = 2, adaptive = false, dt = 0.1f0
    )
    @test length(sol.u) == 2
    @test all(sol -> sol.u[1] == @SVector([1.0f0]), sol.u)
end
