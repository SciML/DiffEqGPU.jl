using DiffEqGPU, StaticArrays, Test
using OrdinaryDiffEq: Tsit5
using SciMLBase: EnsembleProblem, ODEProblem, remake, solve

include("utils.jl")

# Issue #552: scalar parameters must reach the RHS as scalars (not length-1 columns).
# Array parameters must reach the RHS as full column views, including singleton batches.
#
# Analytic solution of du/dt = λ u, u(0) = 1 is u(t) = exp(λ t). With tspan (0, 1):
#   scalar λ = -sim_id              → endpoint exp(-sim_id)
#   vector p = [-sim_id, -2*sim_id] → λ = sum(p) = -3*sim_id → endpoint exp(-3*sim_id)
# Fixed-step Tsit5 (order 5) with dt = 1/16 has global truncation ~ O(dt^5) ≈ 9.5e-7.

const dt = 1 / 16
const atol = 1.0e-6
const rtol = 1.0e-6

expect_endpoint_scalar(sim_id) = exp(-Float64(sim_id))
expect_endpoint_sum2(sim_id) = exp(-3 * Float64(sim_id))

function solve_batch(ens; batch_size, trajectories = 4)
    return solve(
        ens, Tsit5(), EnsembleGPUArray(backend, 0.0);
        trajectories = trajectories, batch_size = batch_size,
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
        @test sol.u[i].u[end][1] ≈ expect_endpoint_scalar(i) atol = atol rtol = rtol
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
        @test sol.u[i].u[end][1] ≈ expect_endpoint_scalar(i) atol = atol rtol = rtol
    end
end

@testset "EnsembleGPUArray vector p sum batch_size=$batch_size" for batch_size in (
        1, 2, 3, 4,
    )
    # Vector-typed RHS: rejects a scalar `p` (unlike bare `p[1]` on a Number).
    f_vector(u, p::AbstractVector, t) = SVector(sum(p) * u[1])
    prob = ODEProblem{false}(f_vector, SVector(1.0), (0.0, 1.0), [-1.0, -2.0])
    ens = EnsembleProblem(
        prob;
        safetycopy = false,
        prob_func = (prob, ctx) -> remake(
            prob;
            p = [-Float64(ctx.sim_id), -2 * Float64(ctx.sim_id)]
        )
    )
    sol = solve_batch(ens; batch_size = batch_size)
    @test length(sol.u) == 4
    for i in 1:4
        @test sol.u[i].u[end][1] ≈ expect_endpoint_sum2(i) atol = atol rtol = rtol
    end
end

@testset "EnsembleGPUArray length-1 vector typed RHS batch_size=1" begin
    f_vector(u, p::AbstractVector, t) = SVector(p[1] * u[1])
    prob = ODEProblem{false}(f_vector, SVector(1.0), (0.0, 1.0), [-1.0])
    ens = EnsembleProblem(prob; safetycopy = false)
    sol = solve_batch(ens; batch_size = 1, trajectories = 2)
    @test sol.u[1].u[end][1] ≈ exp(-1.0) atol = atol rtol = rtol
end

@testset "EnsembleGPUArray lower-level singleton state shape" begin
    f(u, p::AbstractVector, t) = SVector(sum(p) * u[1])
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), [-1.0, -2.0])
    sol = DiffEqGPU.vectorized_map_solve(
        [prob], Tsit5(), EnsembleGPUArray(backend, 0.0), 1:1, false;
        dt = dt, save_everystep = false
    )
    @test size(sol.u[end]) == (1, 1)
    @test Array(sol.u[end])[1] ≈ exp(-3.0) atol = atol rtol = rtol
end

@testset "batch_solve_up rrule scalar p (vector pack)" begin
    # Loss = sum of final states; du(T)/dp = exp(p) with p = -1 ⇒ exp(-1) per trajectory.
    f(u, p, t) = SVector(p * u[1])
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), -1.0)
    probs = [prob, prob]
    ens = EnsembleProblem(prob; safetycopy = false)
    p = DiffEqGPU.pack_ordinary_parameters(probs)
    @test p isa AbstractVector
    _, pb = DiffEqGPU.ChainRulesCore.rrule(
        DiffEqGPU.batch_solve_up, ens, probs, Tsit5(),
        EnsembleGPUArray(backend, 0.0), 1:2, ones(1, 2), p;
        dt = 1 / 64, adaptive = false, save_everystep = false, save_start = false
    )
    grad = pb((nothing, [[ones(1)], [ones(1)]]))[end]
    @test grad isa AbstractVector
    @test length(grad) == 2
    @test grad ≈ fill(exp(-1.0), 2) atol = 1.0e-8 rtol = 1.0e-8
end

@testset "batch_solve_up rrule scalar p singleton" begin
    f(u, p, t) = SVector(p * u[1])
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), -1.0)
    probs = [prob]
    ens = EnsembleProblem(prob; safetycopy = false)
    p = DiffEqGPU.pack_ordinary_parameters(probs)
    @test p isa AbstractVector && length(p) == 1
    _, pb = DiffEqGPU.ChainRulesCore.rrule(
        DiffEqGPU.batch_solve_up, ens, probs, Tsit5(),
        EnsembleGPUArray(backend, 0.0), 1:1, ones(1, 1), p;
        dt = 1 / 64, adaptive = false, save_everystep = false, save_start = false
    )
    grad = pb((nothing, [[ones(1)]]))[end]
    @test grad isa AbstractVector
    @test length(grad) == 1
    @test only(grad) ≈ exp(-1.0) atol = 1.0e-8 rtol = 1.0e-8
end

@testset "batch_solve_up rrule array p (matrix pack)" begin
    # p = [-1, -2] ⇒ λ = -3, u(T) = exp(-3), ∂u/∂p₁ = ∂u/∂p₂ = T*exp(λT) = exp(-3).
    f(u, p::AbstractVector, t) = SVector(sum(p) * u[1])
    prob = ODEProblem{false}(f, SVector(1.0), (0.0, 1.0), [-1.0, -2.0])
    probs = [prob, remake(prob; p = [-1.0, -2.0])]
    ens = EnsembleProblem(prob; safetycopy = false)
    p = DiffEqGPU.pack_ordinary_parameters(probs)
    @test p isa AbstractMatrix && size(p) == (2, 2)
    _, pb = DiffEqGPU.ChainRulesCore.rrule(
        DiffEqGPU.batch_solve_up, ens, probs, Tsit5(),
        EnsembleGPUArray(backend, 0.0), 1:2, ones(1, 2), p;
        dt = 1 / 64, adaptive = false, save_everystep = false, save_start = false
    )
    grad = pb((nothing, [[ones(1)], [ones(1)]]))[end]
    @test grad isa AbstractMatrix
    @test size(grad) == (2, 2)
    @test grad ≈ fill(exp(-3.0), 2, 2) atol = 1.0e-7 rtol = 1.0e-7
end
