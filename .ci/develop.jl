using Pkg

@show pwd()
Pkg.activate(".")

@info "Add oneAPIKernels"

Pkg.add(PackageSpec(url = "https://github.com/utkarsh530/KernelAbstractions.jl.git",
                    rev = "patch-1", subdir = "lib/oneAPIKernels"))

Pkg.update()

@show Pkg.status()
Pkg.build()
Pkg.precompile()
Pkg.test()
