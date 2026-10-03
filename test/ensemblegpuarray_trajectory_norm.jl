using DiffEqGPU, ForwardDiff, Test
using Adapt: adapt
using OrdinaryDiffEq: Tsit5
using SciMLBase: EnsembleProblem, ODEProblem, remake, solve

include("utils.jl")

@testset "TrajectoryNorm" begin
    norm = DiffEqGPU.TrajectoryNorm(2)
    U = [1.0 0.0 0.0; 1.0 0.0 0.0]
    # The worst trajectory's RMS, not the RMS over the batch.
    @test norm(U, 0.0) ≈ 1.0
    @test norm(vec(U), 0.0) ≈ 1.0
    @test norm(adapt(backend, Float32.(U)), 0.0f0) ≈ 1.0f0
    @test norm(-3.0, 0.0) == 3.0
    @test norm(ForwardDiff.Dual(-2.0, 1.0), 0.0) == 2.0
    @test norm(ForwardDiff.Dual.(U, 1.0), 0.0) ≈ 1.0
    @test norm(Float64[], 0.0) == 0.0
    # Not a whole number of trajectories: RMS over all entries.
    @test norm([3.0, 4.0, 0.0], 0.0) ≈ sqrt(25 / 3)
end

# u' = ω cos(ω t), u(0) = 0, so u = sin(ω t). Trajectory 1 needs small steps (ω = 50);
# the others are easy (ω = 1). With a batch-wide RMS the easy trajectories outvote the hard
# one and its error grows with the batch size; each trajectory has to meet its own tolerance.
f_osc!(du, u, p, t) = (du[1] = p[1] * cos(p[1] * t); nothing)

@testset "EnsembleGPUArray meets each trajectory's tolerance" begin
    prob = ODEProblem(f_osc!, [0.0], (0.0, 10.0), [1.0])
    tol = 1.0e-8
    alone = solve(remake(prob; p = [50.0]), Tsit5(); abstol = tol, reltol = tol)
    err_alone = abs(alone.u[end][1] - sin(500.0))
    ep = EnsembleProblem(
        prob; safetycopy = false,
        prob_func = (pr, ctx) -> remake(pr; p = [ctx.sim_id == 1 ? 50.0 : 1.0])
    )
    for ntraj in (2, 1000)
        sol = solve(
            ep, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = ntraj,
            abstol = tol, reltol = tol, save_everystep = false
        )
        err = abs(sol.u[1].u[end][1] - sin(500.0))
        @test err <= 5 * max(err_alone, tol / 10)
        # The easy trajectories ride along on the hard one's steps.
        @test abs(sol.u[end].u[end][1] - sin(10.0)) <= 10 * tol
    end
end
