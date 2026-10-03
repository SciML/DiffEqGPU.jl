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

function time_loss(p, backend)
    prob = ODEProblem{false}((u, p, t) -> SVector(t), SVector(0.0), (p[1], p[2]))
    sol = solve(
        EnsembleProblem(prob; safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(backend, 0.0); trajectories = 3,
        adaptive = false, dt = p[3], save_everystep = false
    )
    return sum(s -> only(s.u[end]), sol.u)
end

@testset "Initial time, final time, and step size gradients" begin
    p = [0.2, 0.93, 0.07]
    @test time_loss(p, backend) ≈ 1.5 * (p[2]^2 - p[1]^2) rtol = 1.0e-10
    dp = zero(p)
    Enzyme.autodiff(Reverse, time_loss, Active, Duplicated(p, dp), Const(backend))
    @test dp ≈ 3 .* [-p[1], p[2], 0.0] rtol = 1.0e-9 atol = 1.0e-10
end

function initial_state_loss(u0, backend)
    prob = ODEProblem{false}(exponential_rhs, SVector(u0[1]), (0.0, 1.0), SVector(0.2))
    prob_func = (prob, ctx) -> remake(prob; u0 = SVector(u0[ctx.sim_id]))
    sol = solve(
        EnsembleProblem(prob; prob_func, safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(backend, 0.0); trajectories = length(u0),
        adaptive = false, dt = 0.05, save_everystep = false
    )
    return sum(s -> only(s.u[end]), sol.u)
end

@testset "Initial state gradients" begin
    u0 = [0.4, 0.5, 0.7]
    @test initial_state_loss(u0, backend) ≈ sum(u0) * exp(0.2) rtol = 1.0e-7
    du0 = zero(u0)
    Enzyme.autodiff(Reverse, initial_state_loss, Active, Duplicated(u0, du0), Const(backend))
    @test du0 ≈ fill(exp(0.2), length(u0)) rtol = 1.0e-7
end

linear_event(u, t, integrator) = u[1] - integrator.p[1]
function curved_event(u, t, integrator)
    d = u[1] - integrator.p[1]
    return d + eltype(u)(1.0e4) * d^3
end
reset_half!(integrator) = (integrator.u = SVector(oftype(integrator.u[1], 0.5)))

function event_time_loss(p::Vector{T}, rhs, condition, u0, backend) where {T}
    prob = ODEProblem{false}(rhs, SVector(T(u0)), (zero(T), one(T)), SVector(p[1]))
    prob_func = (prob, ctx) -> remake(prob; p = SVector(p[ctx.sim_id]))
    cb = ContinuousCallback(condition, reset_half!; save_positions = (false, false))
    sol = solve(
        EnsembleProblem(prob; prob_func, safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(backend, 0.0); trajectories = length(p), adaptive = false,
        dt = T(0.05), callback = cb, merge_callbacks = true, save_everystep = false
    )
    return sum(s -> only(s.u[end]), sol.u)
end

function event_root(p::T, g, bracket, rootfind) where {T}
    return DiffEqGPU.gpu_find_root(t -> g(t, p), T.(bracket), rootfind)
end

@testset "Continuous event root sensitivities ($T, p = $p, $rootfind)" for (T, g, p, bracket) in (
            (Float32, (t, p) -> (t - p) + 1.0f4 * (t - p)^3, 0.71f0, (0.7, 0.75)),
            (Float64, (t, p) -> (t - p) + 1.0e4 * (t - p)^3, 0.71, (0.7, 0.75)),
            (Float64, (t, p) -> (t - p) + (t - p)^3, 1.0e9, (1.0e9 - 1, 1.0e9 + 1)),
            (Float32, (t, p) -> expm1(25000.0f0 * (t - p)), 0.71f0, (0.7065, 0.7135)),
        ), rootfind in (SciMLBase.LeftRootFind, SciMLBase.RightRootFind)
    # Each condition crosses zero transversally at t = p, so dt/dp = 1 exactly.
    dp, root = Enzyme.autodiff(
        ReverseWithPrimal, event_root, Active, Active(p), Const(g), Const(bracket),
        Const(rootfind)
    )
    @test root === event_root(p, g, bracket, rootfind)
    @test first(dp) ≈ 1 rtol = 10 * eps(T)
end

@testset "Event root sensitivities reject degenerate crossings" begin
    # Zero time derivative at the exact root t = 0, a slope that overflows Float32, and
    # a condition whose time derivative cancels. The slopes depend on `p` at run time,
    # as kernel conditions always do through the integrator state.
    for (p, g, bracket) in (
            (0.0, (t, p) -> t^3 - p, (-1.0, 1.0)),
            (0.71f0, (t, p) -> ((t - p) * p * 1.0f38) * 10.0f0, (0.7, 0.75)),
            (0.5, (t, p) -> p * t - p * t + (one(t) - p), (-1.0, 1.0)),
        )
        @test_throws r"time derivative at the event is zero or non-finite" Enzyme.autodiff(
            Reverse, event_root, Active, Active(p), Const(g), Const(bracket),
            Const(SciMLBase.LeftRootFind)
        )
    end
end

@testset "Parameter-dependent continuous event time gradients ($T)" for T in (Float32, Float64)
    rtol = T === Float32 ? 2.0e-5 : 1.0e-8
    # u′ = 1 from 0 hits u = p > 0.75 once, at t = p, so u(1) = 1.5 - p.
    p = T[0.78, 0.84, 0.93]
    rhs = (u, p, t) -> SVector(one(eltype(u)))
    for condition in (linear_event, curved_event)
        @test event_time_loss(p, rhs, condition, 0, backend) ≈ sum(T(1.5) .- p) rtol = rtol
        dp = zero(p)
        Enzyme.autodiff(
            Reverse, event_time_loss, Active, Duplicated(p, dp),
            Const(rhs), Const(condition), Const(0), Const(backend)
        )
        @test dp ≈ -ones(T, 3) rtol = rtol
    end
end

@testset "Nonlinear continuous event time gradients" begin
    # u′ = u from 1 hits u = p at t = log(p), so u(1) = 0.5e / p.
    p = [1.3, 1.7, 2.2]
    rhs = (u, p, t) -> u
    @test event_time_loss(p, rhs, linear_event, 1, backend) ≈ sum(0.5 * ℯ ./ p) rtol = 1.0e-7
    dp = zero(p)
    Enzyme.autodiff(
        Reverse, event_time_loss, Active, Duplicated(p, dp),
        Const(rhs), Const(linear_event), Const(1), Const(backend)
    )
    @test dp ≈ -0.5 * ℯ ./ p .^ 2 rtol = 1.0e-6
end

time_event(u, t, integrator) = t - integrator.p[1]
reset_zero!(integrator) = (integrator.u = zero(integrator.u))
double_state!(integrator) = (integrator.u = 2 * integrator.u)

function endpoint_event_loss(p, affect!, backend)
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(1.0), SVector(0.0), (0.0, 1.0), SVector(p[1])
    )
    cb = ContinuousCallback(time_event, affect!; save_positions = (false, false))
    sol = solve(
        EnsembleProblem(prob; safetycopy = false), GPUTsit5(),
        EnsembleGPUKernel(backend, 0.0); trajectories = 3, adaptive = false,
        dt = 0.25, callback = cb, merge_callbacks = true, save_everystep = false
    )
    return sum(s -> only(s.u[end]), sol.u) / 3
end

@testset "Event time gradients at step endpoints ($(nameof(affect!)))" for (affect!, sign) in (
        (reset_zero!, -1), (double_state!, 1),
    )
    # u′ = 1 from 0 with the event at t = p: resetting gives u(1) = 1 - p and doubling
    # gives u(1) = 2p + (1 - p) = 1 + p. p = 0.5 and 0.75 land exactly on step endpoints.
    for p in (0.5, 0.74, 0.75, 0.76)
        @test endpoint_event_loss([p], affect!, backend) ≈ 1 + sign * p rtol = 1.0e-12
        dp = [0.0]
        Enzyme.autodiff(
            Reverse, endpoint_event_loss, Active, Duplicated([p], dp),
            Const(affect!), Const(backend)
        )
        @test only(dp) ≈ sign rtol = 1.0e-10
    end
end
