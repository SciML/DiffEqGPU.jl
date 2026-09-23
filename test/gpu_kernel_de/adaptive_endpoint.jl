using DiffEqGPU, SciMLBase, StaticArrays, Test

# Drives `step!` directly with an iteration cap so that an integrator that stops
# advancing before `tf` fails the test instead of hanging the kernel loop.
function reaches_endpoint(alg, ::Type{T}, tf, λ, reltol) where {T}
    f = ODEFunction{false}((u, p, t) -> SVector(-u[1], -p[1] * u[2]))
    integ = DiffEqGPU.init(
        alg, f, false, SVector{2, T}(1, 1), zero(T), T(tf), T(tf), SVector{1, T}(λ),
        T(reltol / 1000), T(reltol), DiffEqGPU.DiffEqBase.ODE_DEFAULT_NORM, nothing,
        CallbackSet(nothing), nothing
    )
    ts = zeros(T, 2)
    us = zeros(SVector{2, T}, 2)
    for _ in 1:10_000
        integ.t < T(tf) || return integ.t == T(tf)
        DiffEqGPU.step!(integ, ts, us)
    end
    return false
end

@testset "Adaptive endpoint termination ($(nameof(typeof(alg))), $T)" for
    alg in (
            GPUTsit5(), GPUTsit5IController(), GPUVern7(), GPUVern9(),
            GPURosenbrock23(), GPURodas4(), GPURodas5P(), GPUKvaerno3(), GPUKvaerno5(),
        ),
        T in (Float32, Float64)
    @test all(
        reaches_endpoint(alg, T, tf, λ, reltol)
            for tf in range(0.05, 2.0; length = 200), λ in (1, 10, 100),
            reltol in (1.0e-3, 1.0e-4, 1.0e-6)
    )
end
