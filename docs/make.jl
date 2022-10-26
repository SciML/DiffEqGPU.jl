cd("./docs")
using Pkg;
Pkg.activate(".");
Pkg.instantiate();
using Documenter, DiffEqGPU

include("pages.jl")

makedocs(sitename = "DiffEqGPU.jl",
         authors = "Chris Rackauckas",
         modules = [DiffEqGPU],
         clean = true, doctest = false,
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/DiffEqGPU/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/DiffEqGPU.jl.git";
           push_preview = true)
