using DiffEqGPU, Test

function docs_entries(root)
    entries = Set{String}()
    declarations = Set{String}()
    in_docs_block = false

    md_files = String[]
    for (dir, _, files) in walkdir(root)
        append!(md_files, joinpath.(dir, filter(endswith(".md"), files)))
    end

    for file in md_files
        for line in eachline(file)
            stripped = strip(line)
            if stripped == "```@docs"
                in_docs_block = true
                continue
            elseif startswith(stripped, "```")
                in_docs_block = false
                continue
            end

            for m in eachmatch(r"`([A-Za-z_][A-Za-z_0-9!]*)(?:\([^`]*)?`", line)
                push!(declarations, m.captures[1])
            end

            if in_docs_block && !isempty(stripped)
                push!(entries, stripped)
                push!(entries, last(split(stripped, ".")))
                push!(declarations, stripped)
                push!(declarations, last(split(stripped, ".")))
            end
        end
    end

    return entries, declarations
end

@testset "Public API docs coverage" begin
    exported_names = setdiff(names(DiffEqGPU; all = false), (:DiffEqGPU,))
    missing_docstrings = Symbol[]

    for name in exported_names
        Docs.hasdoc(DiffEqGPU, name) || push!(missing_docstrings, name)
    end

    @test isempty(missing_docstrings)
end

@testset "Rendered API docs coverage" begin
    root = joinpath(pkgdir(DiffEqGPU), "docs", "src")
    entries, declarations = docs_entries(root)
    exported_names = setdiff(names(DiffEqGPU; all = false), (:DiffEqGPU,))

    missing_export_declarations = Symbol[
        name for name in exported_names if String(name) ∉ declarations
    ]
    missing_owned_exports = Symbol[]

    for name in exported_names
        binding = getproperty(DiffEqGPU, name)
        parentmodule(binding) === DiffEqGPU || continue
        String(name) ∈ entries || push!(missing_owned_exports, name)
    end

    developer_interfaces = (
        :EnsembleArrayAlgorithm,
        :EnsembleKernelAlgorithm,
        :GPUODEAlgorithm,
        :GPUSDEAlgorithm,
        :GPUODEImplicitAlgorithm,
        :AbstractNLSolver,
        :AbstractNLSolverCache,
        :NLSolver,
        :vectorized_solve,
        :vectorized_asolve,
        :vectorized_map_solve,
    )
    missing_developer_interfaces = Symbol[
        name for name in developer_interfaces if String(name) ∉ entries
    ]

    @test isempty(missing_export_declarations)
    @test isempty(missing_owned_exports)
    @test isempty(missing_developer_interfaces)
end
