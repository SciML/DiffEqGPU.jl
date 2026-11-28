using Test

@testset "Enzyme environments" begin
    group = get(ENV, "GROUP", "CUDA")
    envdir = joinpath(@__DIR__, "enzyme")
    if group == "CUDA"
        record_env = joinpath(envdir, "cuda_records")
        record_code = """
        using Pkg
        Pkg.instantiate()
        include($(repr(joinpath(envdir, "cuda_records.jl"))))
        """
        @testset "CUDACore record allocations" begin
            run(`$(Base.julia_cmd()) --project=$record_env -e $record_code`)
        end
    end
    code = """
    using Pkg
    Pkg.instantiate()
    using SciMLTesting
    SciMLTesting.activate_group_env($(repr(envdir)))
    include($(repr(joinpath(envdir, "imports.jl"))))
    include($(repr(joinpath(envdir, "problem_adaptation.jl"))))
    include($(repr(joinpath(envdir, "transfers.jl"))))
    include($(repr(joinpath(envdir, "gradients.jl"))))
    """
    @testset "DiffEqGPU integration" begin
        run(addenv(`$(Base.julia_cmd()) --project=$envdir -e $code`, "GROUP" => group))
    end
end
