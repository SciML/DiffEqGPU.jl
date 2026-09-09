using TOML

let
    root = dirname(@__DIR__)
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    # Enzyme GPU gradients are CUDA-only; isolate from OpenCL's GPUCompiler major.
    target = "enzyme_cuda"
    names = project["targets"][target]
    packages = merge(project["deps"], project["weakdeps"], project["extras"])
    deps = Dict(name => packages[name] for name in names)
    compat = Dict(name => project["compat"][name] for name in [names; "julia"])
    mktempdir() do envdir
        open(joinpath(envdir, "Project.toml"), "w") do io
            TOML.print(io, Dict("deps" => deps, "compat" => compat))
        end
        code = """
        using Pkg
        Pkg.develop(path = $(repr(root)))
        Pkg.add(PackageSpec(
            url = "https://github.com/ChrisRackauckas-Claude/CUDA.jl.git",
            subdir = "CUDACore",
            rev = "081de781a6f81a63cb8d1d88c77eb7f5243a163a"
        ))
        Pkg.instantiate()
        include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme_cuda_records.jl"))))
        include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme_cuda_copy_rules.jl"))))
        include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme.jl"))))
        """
        run(`$(Base.julia_cmd()) --project=$envdir -e $code`)
    end
end
