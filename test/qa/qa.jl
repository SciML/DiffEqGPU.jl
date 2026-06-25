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
        # Non-public names from upstream packages (SciMLBase, DiffEqBase, ForwardDiff,
        # LinearSolve, SimpleDiffEq, LinearAlgebra, Adapt, Random, Base/Core); these
        # become public as those base libraries declare `public`.
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase
                :AbstractContinuousCallback, :AbstractDEAlgorithm,
                :AbstractDiscreteCallback, :AbstractEnsembleProblem,
                :AbstractJumpProblem, :AbstractODEProblem, :DEFAULT_REDUCTION,
                :Default, :EnsembleAlgorithm, :FINALIZE_DEFAULT, :INITIALIZE_DEFAULT,
                :LeftRootFind, :NoRootFind, :RootfindOpt, :Success, :T, :Terminated,
                :__solve, :_unwrap_val, :build_linear_solution, :build_solution,
                :default_rng_func, :generate_sim_seeds, :has_initialization_data,
                :is_diagonal_noise, :solve_batch, :specialization,
                :successful_retcode, :tighten_container_eltype,
                # SciMLBase + DiffEqBase
                :AbstractODEAlgorithm, :AbstractODEIntegrator, :AbstractSDEAlgorithm,
                :has_jac, :has_tgrad,
                # DiffEqBase
                :ODE_DEFAULT_NORM, :RecursiveArrayTools, :find_callback_time,
                :find_first_continuous_callback, :get_condition,
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
                # Adapt
                :adapt_structure,
                # Random
                :seed!,
                # Base / Core
                :Compiler, :Fix2, :return_type,
            ),
        ),
        # Non-public names imported from upstream packages (StaticArrays internals,
        # SciMLBase.ImmutableODEProblem).
        all_explicit_imports_are_public = (;
            ignore = (
                :var"@_inline_meta", :ImmutableODEProblem, :LU, :StaticLUMatrix,
                :StaticVecOrMat,
            ),
        ),
    ),
    # ~75 names implicitly imported via heavy `using` of KernelAbstractions/SciMLBase/
    # DiffEqBase/StaticArrays/... ; making them explicit is a large refactor.
    # https://github.com/SciML/DiffEqGPU.jl/issues/466
    ei_broken = (:no_implicit_imports,),
)
