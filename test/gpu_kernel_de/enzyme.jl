using DiffEqGPU, Enzyme, KernelAbstractions, SciMLBase, StaticArrays, Test

const backend = if get(ENV, "GROUP", "Enzyme") == "CUDA"
    using CUDA
    CUDA.CUDABackend()
else
    CPU()
end

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
