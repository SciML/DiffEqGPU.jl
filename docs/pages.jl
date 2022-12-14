# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
        "Tutorials" => Any[
            "tutorials/gpu_ensemble_basic.md",
            "tutorials/parallel_callbacks.md",
            "tutorials/multigpu.md"
            ],
        "Examples" => Any[
            "examples/sde.md",
            "examples/ad.md",
            "examples/reductions.md",
        ],
        "Manual" => Any[
            "manual/ensemblegpukernel.md",
            "manual/ensemblegpuarray.md",
            "manual/optimal_trajectories.md"
            "manual/choosing_ensembler.md"
        ]
        ]
