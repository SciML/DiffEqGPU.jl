using Pkg

@show pwd()
Pkg.activate(".")

@info "Add Metal from main"

Pkg.add(PackageSpec(url = "https://github.com/JuliaGPU/Metal.jl.git",
                    rev = "main"))

Pkg.update()

@show Pkg.status()
Pkg.build()
Pkg.precompile()
Pkg.test()
