using Documenter, DiffEqGPU

include("pages.jl")

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

makedocs(
    sitename = "DiffEqGPU.jl",
    authors = "Chris Rackauckas",
    modules = [DiffEqGPU],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqGPU/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/DiffEqGPU.jl.git";
    push_preview = true
)
