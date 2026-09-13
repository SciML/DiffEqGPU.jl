let
    cuda = get(ENV, "GROUP", "Enzyme") == "CUDA"
    envdir = joinpath(@__DIR__, "enzyme")
    code = """
    using Pkg
    Pkg.instantiate()
    using SciMLTesting
    SciMLTesting.activate_group_env($(repr(envdir)))
    include($(repr(joinpath(envdir, "imports.jl"))))
    if $cuda
        include($(repr(joinpath(envdir, "cuda_records.jl"))))
    end
    include($(repr(joinpath(envdir, "transfers.jl"))))
    include($(repr(joinpath(envdir, "gradients.jl"))))
    """
    run(`$(Base.julia_cmd()) --project=$envdir -e $code`)
end
