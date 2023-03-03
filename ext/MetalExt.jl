module MetalExt
    using KernelAbstractions
    using Metal, MetalKernels, Adapt
    import DiffEqGPU

    DiffEqGPU.maxthreads(::MetalDevice) = 256
    DiffEqGPU.maybe_prefer_blocks(::MetalDevice) = MetalDevice()

    # TODO move to KA
    Adapt.adapt_storage(::CPU, a::MtlArray) = adapt(Array, a)
    Adapt.adapt_storage(::MetalDevice, a::MtlArray) = a
    Adapt.adapt_storage(::MetalDevice, a::Array) = adapt(MtlArray, a)

    function DiffEqGPU.allocate(::MetalDevice, ::Type{T}, init, dims) where {T}
        MtlArray{T}(init, dims)
    end
end