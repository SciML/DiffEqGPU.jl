using TOML

let
    root = dirname(@__DIR__)
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    target = get(ENV, "GROUP", "Enzyme") == "CUDA" ? "enzyme_cuda" : "enzyme"
    names = project["targets"][target]
    packages = merge(project["deps"], project["weakdeps"], project["extras"])
    deps = Dict(name => packages[name] for name in names)
    compat = Dict(name => project["compat"][name] for name in [names; "julia"])
    # Enzyme and current OpenCL require different GPUCompiler major versions.
    mktempdir() do envdir
        open(joinpath(envdir, "Project.toml"), "w") do io
            TOML.print(io, Dict("deps" => deps, "compat" => compat))
        end
        code = """
        using Pkg
        Pkg.develop(path = $(repr(root)))
        if $(target == "enzyme_cuda")
            Pkg.add(PackageSpec(
                url = "https://github.com/ChrisRackauckas-Claude/CUDA.jl.git",
                subdir = "CUDACore",
                rev = "081de781a6f81a63cb8d1d88c77eb7f5243a163a"
            ))
        end
        Pkg.instantiate()
        $(target == "enzyme_cuda") && include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme_cuda_records.jl"))))
        include($(repr(joinpath(@__DIR__, "gpu_kernel_de", "enzyme.jl"))))
        """
        run(`$(Base.julia_cmd()) --project=$envdir -e $code`)
    end
end
