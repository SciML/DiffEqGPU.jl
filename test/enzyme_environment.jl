let
    cuda = get(ENV, "GROUP", "Enzyme") == "CUDA"
    code = """
    using Pkg
    Pkg.instantiate()
    include($(repr(joinpath(@__DIR__, "qa", "enzyme.jl"))))
    if $cuda
        include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme_cuda_records.jl"))))
    end
    include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme_transfers.jl"))))
    include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme.jl"))))
    """
    run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "enzyme")) -e $code`)
end
