using SciMLTesting, DiffEqGPU, Test
using JET

run_qa(
    DiffEqGPU;
    explicit_imports = true,
    ei_kwargs = (;
        # StaticVecOrMat is re-exported by StaticArrays but owned by StaticArraysCore,
        # which is not a direct dependency.
        all_explicit_imports_via_owners = (; ignore = (:StaticVecOrMat,)),
        # SciMLBase abstract types/traits re-exported through DiffEqBase, accessed as
        # DiffEqBase.<name>; DiffEqBase is the natural import point here.
        all_qualified_accesses_via_owners = (;
            ignore = (
                :AbstractODEAlgorithm, :AbstractODEIntegrator, :AbstractSDEAlgorithm,
                :has_jac, :has_tgrad,
            ),
        ),
        # Non-public names from upstream packages; these become public as those base
        # libraries declare `public` (verified flagged against the registered releases
        # on Julia 1.12, where these checks run).
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase (owned here, but not yet declared public)
                :AbstractContinuousCallback, :AbstractDiscreteCallback,
                :AbstractODEIntegrator, :DEFAULT_REDUCTION, :EnsembleAlgorithm,
                :FINALIZE_DEFAULT, :INITIALIZE_DEFAULT, :LeftRootFind, :NoRootFind,
                :RootfindOpt, :__solve, :_unwrap_val, :build_linear_solution,
                :default_rng_func, :generate_sim_seeds, :has_initialization_data,
                :has_tgrad, :solve_batch, :specialization, :tighten_container_eltype,
                # DiffEqBase (the RecursiveArrayTools re-export module)
                :RecursiveArrayTools,
                # ForwardDiff
                :Chunk, :Dual, :Partials, :construct_seeds, :derivative, :jacobian,
                :npartials, :partials, :value,
                # LinearSolve
                :LinearCache, :SciMLLinearSolveAlgorithm, :init_cacheval,
                :needs_concrete_A,
                # SimpleDiffEq
                :_build_atsit5_caches, :_build_tsit5_caches, :bθs,
                # LinearAlgebra
                :HermOrSym,
                # Core
                :Compiler, :return_type,
            ),
        ),
        # Non-public names imported from upstream packages (StaticArrays internals).
        all_explicit_imports_are_public = (;
            ignore = (
                :var"@_inline_meta", :LU, :StaticLUMatrix, :StaticVecOrMat,
            ),
        ),
    ),
    # ~75 names implicitly imported via heavy `using` of KernelAbstractions/SciMLBase/
    # DiffEqBase/StaticArrays/... ; making them explicit is a large refactor.
    # https://github.com/SciML/DiffEqGPU.jl/issues/466
    ei_broken = (:no_implicit_imports,),
)
