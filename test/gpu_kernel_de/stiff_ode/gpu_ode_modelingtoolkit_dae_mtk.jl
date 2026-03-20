using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

# NOTE: This test is currently broken because ModelingToolkit problems with initialization
# data contain MTKParameters which use Vector types that cannot be stored inline in CuArrays.
# This is a known limitation: GPU kernels require element types that are allocated inline.
# Once MTK supports GPU-compatible parameter storage, this test can be re-enabled.
@testset "MTK Pendulum DAE with initialization" begin
    @parameters g = 9.81 L = 1.0
    @variables px(t) py(t) [state_priority = 10] pλ(t)

    eqs = [
        D(D(px)) ~ pλ * px / L
        D(D(py)) ~ pλ * py / L - g
        px^2 + py^2 ~ L^2
    ]

    @mtkcompile pendulum = ODESystem(eqs, t, [px, py, pλ], [g, L])

    mtk_prob = ODEProblem(
        pendulum, [py => 0.99], (0.0, 1.0),
        guesses = [pλ => 0.0, px => 0.1, D(px) => 0.0, D(py) => 0.0]
    )

    # Verify it has initialization data and a mass matrix
    @test SciMLBase.has_initialization_data(mtk_prob.f)
    @test mtk_prob.f.mass_matrix !== LinearAlgebra.I

    # Reference solution with OrdinaryDiffEq
    ref_sol = solve(mtk_prob, Rodas5P())
    @test ref_sol.retcode == SciMLBase.ReturnCode.Success

    # GPU ensemble solve - currently broken due to MTKParameters containing non-inline types
    # Skip actual GPU solve test until MTK supports GPU-compatible parameters
    if backend isa CPU
        monteprob_mtk = EnsembleProblem(mtk_prob, safetycopy = false)
        sol_mtk = solve(
            monteprob_mtk, GPURodas5P(), EnsembleGPUKernel(backend),
            trajectories = 2,
            dt = 0.01,
            adaptive = false
        )
        @test length(sol_mtk.u) == 2
        @test !any(isnan, sol_mtk.u[1][end])

        # GPU solution should be close to reference (fixed step so moderate tolerance)
        @test norm(sol_mtk.u[1][end] - ref_sol.u[end]) < 1.0
    else
        @test_broken false # MTK DAE with initialization not yet supported on GPU
    end
end
