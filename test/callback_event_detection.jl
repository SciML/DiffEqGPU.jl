using DiffEqGPU, JLArrays, GPUArraysCore, Test

GPUArraysCore.allowscalar(false)

@testset "continuous callback direction dispatch" begin
    affect!(integrator) = (integrator.u[1] += 10)
    affect_neg!(integrator) = (integrator.u[1] -= 10)
    callback = DiffEqGPU.ContinuousCallback(
        (u, t, integrator) -> u[1], affect!, affect_neg!
    )
    vector_callback = DiffEqGPU.generate_callback(callback, 3, nothing)

    # DiffEqBase hands the affect a `@view` of its `Vector{Int8}` mask buffer, so the
    # view form is the one real solves exercise.
    masks = (Int8[1, -1, 0], @view(Int8[1, -1, 0, 0][1:3]))
    @testset "mask::$(typeof(mask))" for mask in masks
        u = JLArray(reshape(Float32[1, 2, 3], 1, :))
        integrator = (; u, t = 0.0f0, p = zero(u))

        vector_callback.affect!(integrator, mask)
        @test Array(u) == reshape(Float32[11, -8, 3], 1, :)
    end
end
