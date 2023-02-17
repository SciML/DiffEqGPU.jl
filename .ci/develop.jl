using Pkg

@show pwd()
Pkg.activate(".")

@info "Add MetalKernels for KernelAbstractions "

Pkg.add(PackageSpec(url = "https://github.com/tgymnich/KernelAbstractions.jl.git",
                    rev = "metal", subdir = "lib/MetalKernels"))

Pkg.add(PackageSpec(url = "https://github.com/JuliaGPU/Metal.jl.git",
                    rev = "main"))

Pkg.update()

@show Pkg.status()
Pkg.build()
Pkg.precompile()
Pkg.test()
