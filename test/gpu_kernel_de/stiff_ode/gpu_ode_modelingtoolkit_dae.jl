using DiffEqGPU, StaticArrays, SciMLBase, LinearAlgebra, Test
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

# Test DAE support with singular mass matrices
# This is a simple DAE system: M * u' = f(u, p, t)
# where M = [1 0; 0 0] (singular mass matrix)
# u1' = -0.04*u1 + 1e4*u2  (differential equation)
# 0 = u1 + u2 - 1           (algebraic constraint)

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

# Test with GPURosenbrock23
@testset "GPURosenbrock23 DAE" begin
    sol = solve(
        monteprob, GPURosenbrock23(), EnsembleGPUKernel(backend),
        trajectories = 2,
        dt = 0.001f0,
        adaptive = false
    )
    @test length(sol.u) == 2
    @test !any(isnan, sol.u[1][end])
    # Check algebraic constraint: u1 + u2 = 1
    @test abs(sol.u[1][end][1] + sol.u[1][end][2] - 1.0f0) < 0.01f0
end

# Test with GPURodas4
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

# Test with GPURodas5P
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

# Test with GPUKvaerno3
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

# Test with GPUKvaerno5
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
