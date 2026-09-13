using DiffEqGPU, Enzyme, ExplicitImports, Test

@testset "Enzyme extension imports" begin
    ext = Base.get_extension(DiffEqGPU, :EnzymeExt)
    @test ext !== nothing
    for check in (
            check_no_implicit_imports, check_no_stale_explicit_imports,
            check_all_explicit_imports_are_public, check_all_explicit_imports_via_owners,
            check_all_qualified_accesses_via_owners, check_no_self_qualified_accesses,
        )
        @test check(ext) === nothing
    end
    # _kernel_transfer belongs to this package. The other two names are Enzyme's
    # documented, mandatory custom-rule entry points, which lack `public` declarations.
    @test check_all_qualified_accesses_are_public(
        ext; ignore = (:_kernel_transfer, :augmented_primal, :reverse)
    ) === nothing
end
