using DiffEqGPU, StaticArrays, Test
using OrdinaryDiffEq: Tsit5
using SciMLBase: EnsembleProblem, ODEProblem, remake, solve

include("utils.jl")

# Issue #552: scalar parameters must reach the RHS as scalars (not length-1 columns),
# including when batch_size is 1. Vector parameters keep column slices.
#
# Analytic solution of du/dt = λ u, u(0) = 1 is u(t) = exp(λ t). With tspan (0, 1)
# and λ = -sim_id, the endpoint is exp(-sim_id).
# Fixed-step Tsit5 (order 5) with dt = 1/16 has global truncation ~ O(dt^5) ≈ 9.5e-7.

const dt = 1 / 16
const atol = 1.0e-6
const rtol = 1.0e-6

expect_endpoint(sim_id) = exp(-Float64(sim_id))

function solve_batch(ens; batch_size)
    return solve(
        ens, Tsit5(), EnsembleGPUArray(backend, 0.0);
        trajectories = 4, batch_size = batch_size,
        dt = dt, adaptive = false, save_everystep = false
    )
end

@testset "EnsembleGPUArray scalar p (issue #552 RHS) batch_size=$batch_size" for batch_size in (
        1, 2, 4,
    )
    # Unchanged issue RHS: p is a scalar multiplier, not indexed as p[1].
    f_scalar(u, p, t) = SVector(p * u[1])
    prob = ODEProblem{false}(f_scalar, SVector(1.0), (0.0, 1.0), 1.0)
    ens = EnsembleProblem(
        prob;
        safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = -Float64(ctx.sim_id))
    )
    sol = solve_batch(ens; batch_size = batch_size)
    @test length(sol.u) == 4
    for i in 1:4
        @test sol.u[i].u[end][1] ≈ expect_endpoint(i) atol = atol rtol = rtol
    end
end

@testset "EnsembleGPUArray inplace scalar p batch_size=1" begin
    function f_scalar!(du, u, p, t)
        return du[1] = p * u[1]
    end
    prob = ODEProblem{true}(f_scalar!, [1.0], (0.0, 1.0), 1.0)
    ens = EnsembleProblem(
        prob;
        safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = -Float64(ctx.sim_id))
    )
    sol = solve_batch(ens; batch_size = 1)
    @test length(sol.u) == 4
    for i in 1:4
        @test sol.u[i].u[end][1] ≈ expect_endpoint(i) atol = atol rtol = rtol
    end
end

@testset "EnsembleGPUArray vector p batch_size=$batch_size" for batch_size in (1, 2, 4)
    f_vector(u, p, t) = SVector(p[1] * u[1])
    prob = ODEProblem{false}(f_vector, SVector(1.0), (0.0, 1.0), [1.0])
    ens = EnsembleProblem(
        prob;
        safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = [-Float64(ctx.sim_id)])
    )
    sol = solve_batch(ens; batch_size = batch_size)
    @test length(sol.u) == 4
    for i in 1:4
        @test sol.u[i].u[end][1] ≈ expect_endpoint(i) atol = atol rtol = rtol
    end
end
