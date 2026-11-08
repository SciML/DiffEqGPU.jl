using DiffEqGPU, Enzyme, KernelAbstractions, SciMLBase, StaticArrays, Test

const adaptation_backends = if get(ENV, "GROUP", "Enzyme") == "CUDA"
    using CUDA
    (CUDA.CUDABackend(),)
else
    using JLArrays
    (CPU(), JLArrays.JLBackend())
end

function adaptation_field_loss(x::Vector{T}, backend) where {T}
    f = let rate = x[5]
        (u, p, t) -> rate * u
    end
    prob = SciMLBase.ImmutableODEProblem{false}(
        f, SVector(x[1]), (x[2], x[3]), SVector(x[4]); weight = x[6]
    )
    adapted = DiffEqGPU._adapt_kernel_problem(backend, prob)
    if Enzyme.within_autodiff()
        @assert adapted isa DiffEqGPU.KernelODEProblem
    end
    return adapted.u0[1]^2 + adapted.tspan[1] * adapted.tspan[2] +
        adapted.p[1]^2 + only(adapted.f(SVector(T(3)), adapted.p, zero(T))) +
        adapted.kwargs[:weight]^2
end

@testset "Problem adaptation preserves active fields ($backend)" for backend in adaptation_backends
    for T in (Float32, Float64)
        x = T[2, 0.25, 1.25, 0.5, 0.75, 7]
        @test adaptation_field_loss(x, backend) == T(55.8125)
        dx = ones(T, 6)
        for repetitions in 1:2
            Enzyme.autodiff(Reverse, adaptation_field_loss, Active, Duplicated(x, dx), Const(backend))
            @test dx == ones(T, 6) + repetitions * T[4, 1.25, 0.25, 1, 3, 14]
        end
    end
end
