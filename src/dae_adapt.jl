# Override SciMLBase adapt functions to allow DAEs for GPU kernels
import SciMLBase: adapt_structure
import Adapt

# Allow DAE adaptation for GPU kernels
function adapt_structure(to, f::SciMLBase.ODEFunction{iip}) where {iip}
    # For GPU kernels, we now support DAEs with mass matrices and initialization
    SciMLBase.ODEFunction{iip, SciMLBase.FullSpecialize}(
        f.f,
        jac = f.jac,
        mass_matrix = f.mass_matrix,
        initialization_data = f.initialization_data
    )
end
