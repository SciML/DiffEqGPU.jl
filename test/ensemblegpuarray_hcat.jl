using DiffEqGPU
using KernelAbstractions: CPU
using OrdinaryDiffEq: Tsit5
using SciMLBase: EnsembleProblem, EnsembleSerial, ODEProblem, remake, solve
using StaticArrays: SVector
using Test

function lorenz(u, p, t)
    T = eltype(u)
    du1 = T(10) * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - T(2.666) * u[3]
    return SVector{3, T}(du1, du2, du3)
end

function lorenz_ensemble(n)
    T = Float32
    u0 = SVector{3, T}(1, 0, 0)
    tspan = (zero(T), one(T))
    p = SVector{1, T}(21)
    plist = range(zero(T), T(21); length = max(n, 2))[1:n]
    prob = ODEProblem{false}(lorenz, u0, tspan, p)
    prob_func = (prob, ctx) -> remake(prob, p = SVector{1, T}(plist[ctx.sim_id]))
    return EnsembleProblem(prob; prob_func, safetycopy = false)
end

function solve_bytes(n)
    ens = lorenz_ensemble(n)
    return @allocated solve(
        ens, Tsit5(), EnsembleGPUArray(CPU(), 0.0);
        trajectories = n, save_everystep = false, dense = false,
        dt = 0.1f0, adaptive = false,
    )
end

@testset "EnsembleGPUArray concatenation allocation" begin
    ens = lorenz_ensemble(4)
    kwargs = (;
        trajectories = 4, save_everystep = false, dense = false,
        dt = 0.01f0, adaptive = false,
    )
    gpu = solve(ens, Tsit5(), EnsembleGPUArray(CPU(), 0.0); kwargs...)
    cpu = solve(ens, Tsit5(), EnsembleSerial(); kwargs...)
    for i in 1:4
        @test gpu.u[i].u[end] ≈ cpu.u[i].u[end] atol = 1.0f-4 rtol = 1.0f-3
    end

    one = solve(
        ens, Tsit5(), EnsembleGPUArray(CPU(), 0.0);
        trajectories = 2, batch_size = 1, save_everystep = false, dense = false,
        dt = 0.01f0, adaptive = false,
    )
    @test length(one.u) == 2

    solve_bytes(32)
    b10 = solve_bytes(2^10)
    b12 = solve_bytes(2^12)
    # Four times as many trajectories stays under an 8× allocation increase.
    @test b12 / b10 < 8
end
