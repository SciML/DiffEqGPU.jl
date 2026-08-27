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
# Test 3: ModelingToolkit cartesian pendulum DAE with initialization
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
        pendulum, [py => 0.99], (0.0, 1.0),
        guesses = [pλ => 0.0, px => 0.1, D(px) => 0.0, D(py) => 0.0]
    )

    @test SciMLBase.has_initialization_data(mtk_prob.f)
    @test mtk_prob.f.mass_matrix !== LinearAlgebra.I

    compatible_prob = DiffEqGPU.make_prob_compatible(mtk_prob)
    @test isbitstype(typeof(compatible_prob))
    @test !SciMLBase.has_initialization_data(compatible_prob.f)

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
    @test norm(sol_mtk.u[1].u[end] - ref_sol.u[end]) < 1.0
end
