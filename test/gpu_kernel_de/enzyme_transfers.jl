using DiffEqGPU, Enzyme, KernelAbstractions, StaticArrays, Test

struct TransferRecord{F}
    p::SVector{2, Float64}
    tspan::Tuple{Float64, Float64}
    flag::Bool
    f::F
end

function transfer_loss(p, backend)
    f = let a = p[5]
        x -> a * x
    end
    records = [TransferRecord(SVector(p[1], p[2]), (p[3], p[4]), true, f)]
    device = DiffEqGPU._kernel_transfer(backend, records)
    result = only(DiffEqGPU._kernel_transfer(CPU(), device))
    return result.flag ?
        sum(abs2, result.p) + result.tspan[1] * result.tspan[2] + result.f(3.0) : -1.0
end

const transfer_backends = if get(ENV, "GROUP", "Enzyme") == "CUDA"
    using CUDA
    (CUDA.CUDABackend(),)
else
    using JLArrays
    (CPU(), JLArrays.JLBackend())
end

@testset "Aggregate transfers preserve all active fields ($backend)" for backend in transfer_backends
    @test Base.get_extension(DiffEqGPU, :EnzymeExt) !== nothing
    p = [2.0, 3.0, 4.0, 5.0, 6.0]
    @test transfer_loss(p, backend) == 51.0
    dp = ones(5)
    for repetitions in 1:2
        Enzyme.autodiff(Reverse, transfer_loss, Active, Duplicated(p, dp), Const(backend))
        @test dp == ones(5) + repetitions * [4, 6, 5, 4, 3]
    end
end

function packed_scalar_loss(p, backend)
    arg = DiffEqGPU._pack_kernel_scalar(backend, p[1])
    value = arg isa AbstractArray ? only(DiffEqGPU._kernel_transfer(CPU(), arg)) : arg
    return value^2
end

@testset "Scalar storage is only needed during differentiation ($backend)" for backend in transfer_backends
    for T in (Float32, Float64)
        p = T[2]
        @test DiffEqGPU._pack_kernel_scalar(backend, p[1]) === p[1]
        dp = zero(p)
        Enzyme.autodiff(Reverse, packed_scalar_loss, Active, Duplicated(p, dp), Const(backend))
        @test dp == T[4]
    end
end
