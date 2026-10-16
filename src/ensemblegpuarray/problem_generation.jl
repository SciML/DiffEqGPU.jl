# A single column is returned unchanged. Array columns are concatenated; other columns form a row.
# Ordinary scalar parameters must not use this helper: keep them as a 1-D vector so kernels can
# pass `p[i]` (see `ensemble_param` and `pack_ordinary_parameters`).
function _hcat_batch(cols)
    length(cols) == 1 && return cols[1]
    return cols[1] isa AbstractArray ? reduce(hcat, cols) : reshape(cols, 1, :)
end

# Pack per-trajectory `prob.p` values for `EnsembleGPUArray` kernels.
#
# Scalar parameters become a 1-D vector so `ensemble_param` returns `p[i]` as a `Number`.
# Array parameters (including a singleton trajectory or a length-1 vector) become an
# `nparam × ntraj` matrix so `ensemble_param` returns the full column `p[:, i]`.
function pack_ordinary_parameters(probs)
    cols = [prob.p isa AbstractArray ? Array(prob.p) : prob.p for prob in probs]
    if cols[1] isa AbstractArray
        # Always a matrix — including n = 1 — so singleton array packs are not mistaken
        # for scalar batches by `ensemble_param`'s 1-D branch.
        return length(cols) == 1 ? hcat(cols[1]) : reduce(hcat, cols)
    else
        return cols
    end
end

function generate_problem(
        prob::SciMLBase.AbstractODEProblem,
        u0,
        p,
        jac_prototype,
        colorvec
    )
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t; ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
                kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(
                    jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
                return lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
                kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(
                    jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
                return lufact!(version, W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
                kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                return kernel(version)(
                    tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
            end
        end
    else
        _tgrad = nothing
    end

    f_func = ODEFunction(
        _f; Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec,
        jac_prototype,
        sparsity = nothing,
        tgrad = _tgrad
    )
    return prob = ODEProblem(
        f_func, u0, prob.tspan, p;
        prob.kwargs...
    )
end

function generate_problem(prob::SDEProblem, u0, p, jac_prototype, colorvec)
    if prob.noise_rate_prototype !== nothing
        error("Incompatible problem detected. EnsembleGPUArray currently requires `prob.noise_rate_prototype === nothing`, i.e. only diagonal noise is currently supported. Track https://github.com/SciML/DiffEqGPU.jl/issues/331 for more information.")
    end

    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    _g = let f = prob.f.g, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
                kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(
                    jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
                return lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
                kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(
                    jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
                return lufact!(version, W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
                kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                return kernel(version)(
                    tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
            end
        end
    else
        _tgrad = nothing
    end

    f_func = SDEFunction(
        _f, _g; Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec,
        jac_prototype,
        sparsity = nothing,
        tgrad = _tgrad
    )
    return prob = SDEProblem(
        f_func, _g, u0, prob.tspan, p;
        prob.kwargs...
    )
end
