using DiffEqGPU, Enzyme, KernelAbstractions, SciMLBase, StaticArrays, Test

using CUDA
const backend = CUDA.CUDABackend()

exponential_rhs(u, p, t) = p[1] * u

function ensemble_loss(p::Vector{T}, backend, adaptive) where {T}
    prob = ODEProblem{false}(
        exponential_rhs, SVector(one(T)), (zero(T), one(T)), SVector(p[1])
    )
    prob_func = (prob, ctx) -> remake(prob; p = SVector(p[ctx.sim_id]))
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = length(p), adaptive, dt = T(0.05),
        abstol = T(1.0e-7), reltol = T(1.0e-6), save_everystep = false
    )
    return sum(s -> sum(s.u[end]), sol.u)
end

@testset "Enzyme ensemble gradients ($T, adaptive=$adaptive)" for
    T in (Float32, Float64), adaptive in (false, true)
    p = T[0.2, -0.3, 0.1]
    dp = zero(p)
    Enzyme.autodiff(
        Reverse, ensemble_loss, Active, Duplicated(p, dp), Const(backend), Const(adaptive)
    )
    @test dp ≈ exp.(p) rtol = (T === Float32 ? 2.0e-5 : 1.0e-7)
end

function lorenz_rhs(u, p, t)
    x, y, z = u
    return SVector(10 * (y - x), x * (p[1] - z) - y, x * y - (8 / 3) * z)
end

function lorenz_sensitivity_rhs(u, p, t)
    x, y, z, sx, sy, sz = u
    return SVector(
        10 * (y - x), x * (p[1] - z) - y, x * y - (8 / 3) * z,
        10 * (sy - sx), (p[1] - z) * sx - sy - x * sz + x,
        y * sx + x * sy - (8 / 3) * sz
    )
end

function lorenz_loss(p, backend, adaptive)
    prob = ODEProblem{false}(lorenz_rhs, SVector(1.0, 0.0, 0.0), (0.0, 1.0), SVector(p[1]))
    prob_func = (prob, ctx) -> remake(prob; p = SVector(p[ctx.sim_id]))
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(
        ensemble, GPUTsit5(), EnsembleGPUKernel(backend, 0.0);
        trajectories = length(p), adaptive, dt = 0.001,
        abstol = 1.0e-10, reltol = 1.0e-9, save_everystep = false
    )
    return sum(s -> sum(abs2, s.u[end]), sol.u)
end

@testset "Lorenz parameter gradients" begin
    p = [28.0, 29.0, 30.0]
    prob = ODEProblem{false}(
        lorenz_sensitivity_rhs, SVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0), SVector(p[1])
    )
    prob_func = (prob, ctx) -> remake(prob; p = SVector(p[ctx.sim_id]))
    reference = solve(
        EnsembleProblem(prob; prob_func, safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(CPU(), 0.0); trajectories = length(p),
        adaptive = false, dt = 0.0001, save_everystep = false
    )
    expected = map(reference.u) do sol
        u = sol.u[end]
        2 * sum(u[i] * u[i + 3] for i in 1:3)
    end
    for adaptive in (false, true)
        dp = zero(p)
        Enzyme.autodiff(
            Reverse, lorenz_loss, Active, Duplicated(p, dp), Const(backend), Const(adaptive)
        )
        @test dp ≈ expected rtol = 1.0e-6
    end
end
