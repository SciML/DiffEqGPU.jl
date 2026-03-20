using DiffEqGPU, StaticArrays, SciMLBase, LinearAlgebra, Test
using OrdinaryDiffEq
using KernelAbstractions: CPU

const GROUP = get(ENV, "GROUP", "CUDA")

if GROUP == "CUDA"
    using CUDA
    const backend = CUDABackend()
elseif GROUP == "AMDGPU"
    using AMDGPU
    const backend = ROCBackend()
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
    @test !any(isnan, sol.u[1][end])
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

@testset "GPURodas4 DAE" begin
    sol = solve(
        monteprob, GPURodas4(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1][end])
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

@testset "GPURodas5P DAE" begin
    sol = solve(
        monteprob, GPURodas5P(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1][end])
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

@testset "GPUKvaerno3 DAE" begin
    sol = solve(
        monteprob, GPUKvaerno3(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1][end])
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

@testset "GPUKvaerno5 DAE" begin
    sol = solve(
        monteprob, GPUKvaerno5(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1][end])
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

# ============================================================================
# Test 2: ModelingToolkit cartesian pendulum DAE with initialization
# ============================================================================

# ModelingToolkit is an optional test dependency — skip this test if not available.
# This avoids compat conflicts in the alldeps minimum-version resolution test.
# The MTK test is in a separate file because macros need to be available at parse time.
if Base.identify_package("ModelingToolkit") !== nothing
    include("gpu_ode_modelingtoolkit_dae_mtk.jl")
else
    @info "ModelingToolkit not available, skipping MTK DAE test"
end
