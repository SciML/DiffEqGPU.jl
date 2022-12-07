# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
        "Tutorials" => Any[
            "gpu_ensemble_basic.md",
            "parallel_callbacks.md",
            "multigpu.md"
            ],
        "Manual" => Any[
            "ensemblegpukernel.md",
            "ensemblegpuarray.md",
            "optimal_trajectories.md"
        ]
        ]
