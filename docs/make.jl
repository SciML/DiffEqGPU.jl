using Documenter, DiffEqGPU

include("pages.jl")

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

makedocs(;
    sitename = "DiffEqGPU.jl",
    authors = "Chris Rackauckas",
    modules = [DiffEqGPU],
    clean = true, linkcheck = true,
    linkcheck_ignore = [
        # SciML's hosted docs reject Documenter's linkcheck crawler with HTTP 403,
        # though the cross-doc links resolve fine in a browser.
        r"^https://docs\.sciml\.ai/.*",
    ],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqGPU/stable/"
    ),
    pages
)

deploydocs(
    repo = "github.com/SciML/DiffEqGPU.jl.git";
    push_preview = true
)
