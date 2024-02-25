# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "getting_started.md",
    "Tutorials" => Any[
        "GPU Ensembles" => Any["tutorials/gpu_ensemble_basic.md",
            "tutorials/parallel_callbacks.md",
            "tutorials/multigpu.md",
            "tutorials/lower_level_api.md",
            "tutorials/weak_order_conv_sde.md"],
        "Within-Method GPU" => Any["tutorials/within_method_gpu.md"]],
    "Examples" => Any[
        "GPU Ensembles" => Any["examples/sde.md",
            "examples/ad.md",
            "examples/reductions.md"],
        "Within-Method GPU" => Any["examples/reaction_diffusion.md",
            "examples/bruss.md"]],
    "Manual" => Any["manual/ensemblegpukernel.md",
        "manual/ensemblegpuarray.md",
        "manual/backends.md",
        "manual/optimal_trajectories.md",
        "manual/choosing_ensembler.md"]
]
