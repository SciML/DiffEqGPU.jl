using DiffEqGPU, JLArrays, Test

@testset "continuous callback direction dispatch" begin
    affect!(integrator) = (integrator.u[1] += 10)
    affect_neg!(integrator) = (integrator.u[1] -= 10)
    callback = DiffEqGPU.ContinuousCallback(
        (u, t, integrator) -> u[1], affect!, affect_neg!
    )
    vector_callback = DiffEqGPU.generate_callback(callback, 3, nothing)
    u = JLArray(reshape(Float32[1, 2, 3], 1, :))
    integrator = (; u, t = 0.0f0, p = zero(u))

    vector_callback.affect!(integrator, Int8[1, -1, 0])
    @test Array(u) == reshape(Float32[11, -8, 3], 1, :)
end
