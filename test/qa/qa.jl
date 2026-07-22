using SciMLTesting, DiffEqGPU
using JET

const REEXPORTED_API = (
    :BrownFullBasicInit,
    :CheckInit,
    :EnsembleDistributed,
    :EnsembleProblem,
    :EnsembleSerial,
    :EnsembleSolution,
    :EnsembleThreads,
    :terminate!,
)

run_qa(
    DiffEqGPU;
    reexports_allow = REEXPORTED_API,
    api_docs_kwargs = (; rendered_ignore = REEXPORTED_API),
    ei_kwargs = (;
        # StaticVecOrMat is re-exported by StaticArrays but owned by StaticArraysCore.
        # It is a non-public type alias used only in method-signature dispatch for the
        # vendored GPU linear-solve kernels; importing it from its true owner would
        # still leave it non-public, so the via-owners exception is the natural place.
        all_explicit_imports_via_owners = (; ignore = (:StaticVecOrMat,)),
        # Non-public names accessed qualified from upstream packages. These are genuine
        # internal/extension APIs; they will drop out of this list as those packages
        # declare `public` (verified flagged against the registered releases on Julia
        # 1.12, where these checks run).
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase callback/rootfind/ensemble internals (not yet `public`)
                :AbstractContinuousCallback, :AbstractDiscreteCallback,
                :DEFAULT_REDUCTION, :FINALIZE_DEFAULT, :INITIALIZE_DEFAULT,
                :LeftRootFind, :NoRootFind, :RootfindOpt, :build_linear_solution,
                :default_rng_func, :generate_sim_seeds, :solve_batch,
                :specialization, :tighten_container_eltype,
                # ForwardDiff differentiation API (documented but not `public`)
                :Chunk, :Dual, :Partials, :construct_seeds, :derivative, :jacobian,
                :npartials, :partials, :value,
                # LinearSolve cache/algorithm extension interface (not `public`)
                :LinearCache, :SciMLLinearSolveAlgorithm, :init_cacheval,
                :needs_concrete_A,
                # SimpleDiffEq Tsit5 tableau-cache internals (not `public`)
                :_build_atsit5_caches, :_build_tsit5_caches, :bθs,
                # LinearAlgebra Hermitian/Symmetric union (stdlib, not `public`)
                :HermOrSym,
                # Core compiler inference used to size a Channel/Vector eltype; no
                # public cross-version replacement (Base.infer_return_type is 1.11+,
                # and the LTS floor is Julia 1.10).
                :Compiler, :return_type,
            ),
        ),
        # Non-public names imported explicitly from upstream packages. The
        # StaticArrays/StaticArraysCore internals back the vendored GPU LU/linsolve
        # kernels; `setindex` is the immutable Base helper used by the GPU LU pivot.
        all_explicit_imports_are_public = (;
            ignore = (
                :var"@_inline_meta", :LU, :StaticLUMatrix, :StaticVecOrMat,
                :StaticMatrix, :StaticVector, :similar_type, :setindex,
            ),
        ),
    ),
)
