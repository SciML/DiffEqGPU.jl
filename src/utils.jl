diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

make_prob_compatible(prob) = prob

function make_prob_compatible(prob::T) where {T <: ODEProblem}
    # Strip function wrappers to make the problem GPU-compatible
    # This is necessary for ModelingToolkit-generated problems with RuntimeGeneratedFunctions
    f_unwrapped = SciMLBase.unwrapped_f(prob.f)

    # Create a new ODEFunction with the unwrapped function
    # Remove the symbolic system (sys) which contains non-isbits types
    # Keep other properties like mass_matrix, jac, etc. that may be GPU-compatible
    f_stripped = if prob.f isa SciMLBase.ODEFunction
        SciMLBase.ODEFunction{SciMLBase.isinplace(prob)}(
            f_unwrapped;
            mass_matrix = prob.f.mass_matrix,
            analytic = prob.f.analytic,
            tgrad = prob.f.tgrad,
            jac = prob.f.jac,
            jvp = prob.f.jvp,
            vjp = prob.f.vjp,
            jac_prototype = prob.f.jac_prototype,
            sparsity = prob.f.sparsity,
            Wfact = prob.f.Wfact,
            Wfact_t = prob.f.Wfact_t,
            paramjac = prob.f.paramjac,
            syms = prob.f.syms,
            indepsym = prob.f.indepsym,
            colorvec = prob.f.colorvec,
            sys = nothing  # Remove symbolic system for GPU compatibility
        )
    else
        f_unwrapped
    end

    # Remake the problem with stripped function
    # Keep parameters as-is; they will be handled by Adapt rules
    prob_stripped = remake(prob; f = f_stripped)

    # Convert to ImmutableODEProblem
    convert(ImmutableODEProblem, prob_stripped)
end
