# DiffEqGPU: Massively Data-Parallel GPU Solving of ODEs

This library is a component package of the DifferentialEquations.jl ecosystem. It includes
functionality for making use of GPUs in the differential equation solvers.

## Installation

To install DiffEqGPU.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("DiffEqGPU")
```

This will also install all the dependencies, including the
[CUDA.jl](https://cuda.juliagpu.org/stable/), which will also install all the required
versions of CUDA and CuDNN required by these libraries. Note that the same requirements
of CUDA.jl apply to DiffEqGPU, such as requiring a GPU with CUDA v11 compatibility. For
more information on these requirements, see
[the requirements of CUDA.jl](https://cuda.juliagpu.org/stable/installation/overview/).

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - There are a few community forums:
    
      + the #diffeq-bridged channel in the [Julia Slack](https://julialang.org/slack/)
      + [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
      + on the [Julia Discourse forums](https://discourse.julialang.org)
      + see also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
