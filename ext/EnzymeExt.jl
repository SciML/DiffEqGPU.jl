module EnzymeExt

using DiffEqGPU: DiffEqGPU
using Enzyme: Enzyme, EnzymeRules, Const, Duplicated, Reverse

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(DiffEqGPU._kernel_transfer)},
        ::Type{RT}, backend::Const, x::Duplicated
    ) where {RT}
    y = func.val(backend.val, x.val)
    # Zero the elements on the host; device-array wrapper metadata can be inactive.
    dy = func.val(backend.val, Enzyme.make_zero(Array(x.val)))
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? y : nothing,
        EnzymeRules.needs_shadow(config) ? dy : nothing, dy
    )
end

_copy_payload!(dest, src) = (copyto!(dest, src); nothing)

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(DiffEqGPU._kernel_transfer)},
        ::Type{RT}, dy, backend::Const, x::Duplicated
    ) where {RT}
    # Let Enzyme accumulate every active field, including captured functions and times.
    primal = Array(x.val)
    dx = Array(x.dval)
    Enzyme.autodiff(
        Reverse, _copy_payload!, Const,
        Duplicated(copy(primal), Array(dy)), Duplicated(primal, dx)
    )
    copyto!(x.dval, dx)
    return (nothing, nothing)
end

end
