using DiffEqGPU, StaticArrays, SciMLBase, LinearAlgebra, Test
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using KernelAbstractions: CPU

const GROUP = get(ENV, "GROUP", "CUDA")

if GROUP == "CUDA"
    using CUDA
    const backend = CUDABackend()
elseif GROUP == "AMDGPU"
    using AMDGPU
    const backend = ROCBackend()
elseif GROUP == "JLArrays"
    using JLArrays
    const backend = JLBackend()
else
    const backend = CPU()
end

# ============================================================================
# Test 1: Direct mass matrix DAE (no MTK, no initialization)
# ============================================================================

mm = SA[
    1.0f0 0.0f0
    0.0f0 0.0f0
]

function dae_f(u, p, t)
    return SA[
        -0.04f0 * u[1] + 1.0f4 * u[2],
        u[1] + u[2] - 1.0f0,
    ]
end

function dae_jac(u, p, t)
    return SA[
        -0.04f0 1.0f4
        1.0f0 1.0f0
    ]
end

u0 = SA[1.0f0, 0.0f0]
tspan = (0.0f0, 0.1f0)

f = SciMLBase.ODEFunction(dae_f, mass_matrix = mm, jac = dae_jac)
prob = SciMLBase.ODEProblem{false}(f, u0, tspan)
monteprob = SciMLBase.EnsembleProblem(prob, safetycopy = false)

@testset "GPURosenbrock23 DAE" begin
    sol = solve(
        monteprob, GPURosenbrock23(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1].u[end])
    @test abs(sol.u[1].u[end][1] + sol.u[1].u[end][2] - 1.0f0) < 0.01f0
end

@testset "GPURodas4 DAE" begin
    sol = solve(
        monteprob, GPURodas4(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1].u[end])
    @test abs(sol.u[1].u[end][1] + sol.u[1].u[end][2] - 1.0f0) < 0.01f0
end

@testset "GPURodas5P DAE" begin
    sol = solve(
        monteprob, GPURodas5P(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1].u[end])
    @test abs(sol.u[1].u[end][1] + sol.u[1].u[end][2] - 1.0f0) < 0.01f0
end

@testset "GPUKvaerno3 DAE" begin
    sol = solve(
        monteprob, GPUKvaerno3(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1].u[end])
    @test abs(sol.u[1].u[end][1] + sol.u[1].u[end][2] - 1.0f0) < 0.01f0
end

@testset "GPUKvaerno5 DAE" begin
    sol = solve(
        monteprob, GPUKvaerno5(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1].u[end])
    @test abs(sol.u[1].u[end][1] + sol.u[1].u[end][2] - 1.0f0) < 0.01f0
end

# ============================================================================
# Test 2: Structured ModelingToolkit parameter storage
# ============================================================================

@testset "Structured MTKParameters storage" begin
    p = MTKParameters(
        [1.0, 2.0],
        ([3.0], (offset = [4.0],)),
        (mode = ([5.0],),),
        (6.0, MVector(7.0)),
        (),
        ([8.0],)
    )
    compatible_p = DiffEqGPU.make_parameter_compatible(p)

    @test compatible_p.tunable isa SVector
    @test compatible_p.initials[1] isa SVector
    @test compatible_p.initials[2].offset isa SVector
    @test compatible_p.discrete.mode[1] isa SVector
    @test compatible_p.constant[2] isa SVector
    @test compatible_p.caches[1] isa SVector
    @test isbitstype(typeof(compatible_p))
end

# ============================================================================
# Test 3: Non-square and bounded nonlinear least-squares initialization
# ============================================================================

function initialization_test_problem(initprob; metadata = nothing)
    initdata = SciMLBase.OverrideInitData(
        initprob, nothing, sol -> sol.u, nothing, metadata, Val(false)
    )
    f = SciMLBase.ODEFunction{false}(
        (u, p, t) -> zero(u); initialization_data = initdata
    )
    return SciMLBase.ODEProblem{false}(f, initprob.u0, (0.0f0, 0.1f0))
end

function solve_initialization_test(initprob; metadata = nothing)
    prob = initialization_test_problem(initprob; metadata)
    ensemble_prob = EnsembleProblem(prob, safetycopy = false)
    return solve(
        ensemble_prob, GPUTsit5(), EnsembleGPUKernel(backend);
        trajectories = 2, dt = 0.1f0, adaptive = false, save_everystep = false
    )
end

@testset "Underdetermined initialization" begin
    initf = SciMLBase.NonlinearFunction{false}(
        (u, p) -> SA[u[1] + u[2] - 3.0f0]; resid_prototype = SA[0.0f0]
    )
    initprob = SciMLBase.NonlinearLeastSquaresProblem{false}(
        initf, SA[0.0f0, 0.0f0], nothing
    )
    sol = solve_initialization_test(initprob)

    @test length(sol.u) == 2
    @test sol.u[1].u[1] ≈ SA[1.5f0, 1.5f0] atol = 1.0f-5
end

@testset "Overdetermined initialization" begin
    initf = SciMLBase.NonlinearFunction{false}(
        (u, p) -> SA[u[1] - 2.0f0, 2.0f0 * u[1] - 4.0f0];
        resid_prototype = SA[0.0f0, 0.0f0]
    )
    initprob = SciMLBase.NonlinearLeastSquaresProblem{false}(
        initf, SA[0.0f0], nothing
    )
    sol = solve_initialization_test(initprob)

    @test length(sol.u) == 2
    @test sol.u[1].u[1] ≈ SA[2.0f0] atol = 1.0f-5
end

@testset "Bounded initialization" begin
    initf = SciMLBase.NonlinearFunction{false}(
        (u, p) -> SA[u[1] - 0.75f0]; resid_prototype = SA[0.0f0]
    )
    initprob = SciMLBase.NonlinearLeastSquaresProblem{false}(
        initf, SA[0.2f0], nothing; lb = SA[0.0f0], ub = SA[1.0f0]
    )
    compatible_prob = DiffEqGPU.make_prob_compatible(initialization_test_problem(initprob))
    compatible_initprob = compatible_prob.f.initialization_data.initializeprob
    @test compatible_initprob.lb === nothing
    @test compatible_initprob.ub === nothing
    @test isbitstype(typeof(compatible_initprob))

    sol = solve_initialization_test(initprob)
    @test length(sol.u) == 2
    @test 0.0f0 < only(sol.u[1].u[1]) < 1.0f0
    @test sol.u[1].u[1] ≈ SA[0.75f0] atol = 1.0f-5
end

@testset "Immutable SCC initialization" begin
    initprob = SciMLBase.ImmutableNonlinearProblem{false}(
        (u, p) -> SA[u[1]^2 - 4.0f0, 3.0f0 * u[2] - u[1] - 1.0f0],
        SA[1.0f0, 0.0f0]
    )
    metadata = DiffEqGPU.ImmutableSCCInitialization(
        (
            DiffEqGPU.ImmutableSCCBlock{1, 1, false}(),
            DiffEqGPU.ImmutableSCCBlock{2, 1, true}(),
        )
    )
    sol = solve_initialization_test(initprob; metadata)

    @test length(sol.u) == 2
    @test sol.u[1].u[1] ≈ SA[2.0f0, 1.0f0] atol = 1.0f-5
    @test isbitstype(typeof(metadata))
end

@testset "SCC initialization rejects non-triangular blocks" begin
    # Row 1 depends on the downstream block, so the claimed SCC layout is wrong and the
    # sequential solve leaves a nonzero full residual.
    initprob = SciMLBase.ImmutableNonlinearProblem{false}(
        (u, p) -> SA[u[1] - u[2], u[2] - 2.0f0],
        SA[0.0f0, 0.0f0]
    )
    metadata = DiffEqGPU.ImmutableSCCInitialization(
        (
            DiffEqGPU.ImmutableSCCBlock{1, 1, true}(),
            DiffEqGPU.ImmutableSCCBlock{2, 1, true}(),
        )
    )
    sol = DiffEqGPU.solve_initialization_problem(
        initprob, metadata, nothing, 1.0f-6, 1.0f-6
    )
    @test !SciMLBase.successful_retcode(sol)
end

@testset "Stateless all-linear SCC initialization" begin
    @variables lscc_x lscc_y
    linear_scc_sys = ModelingToolkit.complete(
        ModelingToolkit.System(
            [0 ~ 2.0f0 * lscc_x - 4.0f0, 0 ~ lscc_x + lscc_y - 5.0f0],
            [lscc_x, lscc_y], []; name = :linear_scc_sys
        )
    )
    # MTK's all-linear SCC problems can carry no state at all; the block contents are
    # placeholders because the lowered solve only uses the flat residual.
    linear_blocks = (
        SciMLBase.LinearProblem(ones(Float32, 1, 1), zeros(Float32, 1)),
        SciMLBase.LinearProblem(ones(Float32, 1, 1), zeros(Float32, 1)),
    )
    sccprob = SciMLBase.SCCNonlinearProblem(
        linear_blocks, (Returns(nothing), Returns(nothing)); sys = linear_scc_sys
    )
    @test SciMLBase.state_values(sccprob) === nothing

    lowered = DiffEqGPU.lower_initialization_problem(sccprob)
    compat = DiffEqGPU.make_nonlinear_problem_compatible(lowered)
    sol = DiffEqGPU.solve_initialization_problem(
        compat.problem, DiffEqGPU.ImmutableSCCInitialization(compat.blocks),
        nothing, 1.0f-6, 1.0f-6
    )
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ SA[2.0f0, 3.0f0] atol = 1.0f-5
end

# ============================================================================
# Test 4: Host symbolic setters and trivial initialization
# ============================================================================

@testset "Host symbolic setter with trivial initialization" begin
    @parameters decay_rate = 1.0
    @variables population(t) = 1.0
    @mtkcompile decay_system = ODESystem(
        [D(population) ~ -decay_rate * population], t
    )

    decay_prob = ODEProblem(decay_system, [], (0.0, 0.1))
    @test SciMLBase.has_initialization_data(decay_prob.f)
    @test SciMLBase.is_trivial_initialization(decay_prob)

    symbolic_setter = ModelingToolkit.SymbolicIndexingInterface.setsym_oop(
        decay_system, [decay_rate]
    )
    decay_prob_func = function (prob, ctx)
        u0, p = symbolic_setter(prob, SA[Float64(ctx.sim_id)])
        return remake(prob; u0, p)
    end

    ensemble_prob = EnsembleProblem(
        decay_prob; prob_func = decay_prob_func, safetycopy = false
    )
    sol = solve(
        ensemble_prob, GPUTsit5(), EnsembleGPUKernel(backend);
        trajectories = 2, dt = 0.001, adaptive = false, save_everystep = false
    )

    @test length(sol.u) == 2
    @test only(sol.u[1].u[end]) ≈ exp(-0.1) atol = 1.0e-5
    @test only(sol.u[2].u[end]) ≈ exp(-0.2) atol = 1.0e-5
end

# ============================================================================
# Test 5: ModelingToolkit cartesian pendulum DAE with initialization
# ============================================================================

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
        pendulum, [py => 0.99, D(px) => 0.0], (0.0, 1.0),
        guesses = [pλ => 0.0, px => 0.1, D(py) => 0.0]
    )

    @test SciMLBase.has_initialization_data(mtk_prob.f)
    @test mtk_prob.f.mass_matrix !== LinearAlgebra.I
    @test mtk_prob.f.initialization_data.initializeprob isa SciMLBase.SCCNonlinearProblem

    compatible_prob = DiffEqGPU.make_prob_compatible(mtk_prob)
    @test isbitstype(typeof(compatible_prob))
    @test SciMLBase.has_initialization_data(compatible_prob.f)
    @test isbitstype(typeof(compatible_prob.f.initialization_data.initializeprob))
    @test compatible_prob.f.initialization_data.initializeprob isa
        SciMLBase.ImmutableNonlinearProblem
    metadata = compatible_prob.f.initialization_data.metadata
    @test metadata isa DiffEqGPU.ImmutableSCCInitialization
    @test length(metadata.blocks) == 3
    @test metadata.blocks[1] isa DiffEqGPU.ImmutableSCCBlock{1, 1, false}
    @test metadata.blocks[2] isa DiffEqGPU.ImmutableSCCBlock{2, 1, true}
    @test metadata.blocks[3] isa DiffEqGPU.ImmutableSCCBlock{3, 1, true}
    @test isbitstype(typeof(metadata))
    @test compatible_prob.u0 == SVector{length(mtk_prob.u0)}(mtk_prob.u0)

    ref_sol = solve(mtk_prob, Rodas5P())
    @test SciMLBase.successful_retcode(ref_sol)

    monteprob_mtk = EnsembleProblem(mtk_prob, safetycopy = false)
    sol_mtk = solve(
        monteprob_mtk, GPURodas5P(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.01,
        adaptive = false
    )
    @test length(sol_mtk.u) == 2
    @test !any(isnan, sol_mtk.u[1].u[end])
    @test norm(sol_mtk.u[1].u[1] - ref_sol.u[1]) < 1.0e-5
    @test norm(sol_mtk.u[1].u[end] - ref_sol.u[end]) < 1.0
end
