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

# Analytic regression for Issue #551: EnsembleGPUArray CPU workgroups must not
# leak the tspan-reparameterized time from one lane into the next.
@testset "EnsembleGPUArray per-trajectory tspan (CPU)" begin
    f_poly(u, p, t) = SVector(p[1] + t)
    poly_prob = ODEProblem{false}(f_poly, SVector(1.0), (0.0, 1.0), SVector(1.0))
    ens = EnsembleProblem(
        poly_prob;
        safetycopy = false,
        prob_func = (prob, ctx) -> begin
            i = ctx.sim_id
            remake(
                prob;
                u0 = SVector(Float64(i)),
                p = SVector(Float64(i)),
                tspan = (i / 8, i / 8 + 1 / 2)
            )
        end
    )
    sol = solve(
        ens, Tsit5(), EnsembleGPUArray(CPU(), 0.0);
        trajectories = 4, batch_size = 4, dt = 1 / 16, adaptive = false,
        save_everystep = false
    )
    for i in 1:4
        a, b = i / 8, i / 8 + 1 / 2
        # Exact for u' = i + t, u(a) = i: u(b) = i + i*(b-a) + (b^2-a^2)/2
        exact = i + i * (b - a) + (b^2 - a^2) / 2
        @test sol.u[i].u[end][1] ≈ exact atol = 1e-12
    end
end
