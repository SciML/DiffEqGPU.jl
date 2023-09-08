function generate_problem(prob::ODEProblem, u0, p, jac_prototype, colorvec)
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t; ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
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
                kernel(version)(tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
            end
        end
    else
        _tgrad = nothing
    end

    f_func = ODEFunction(_f, Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec=colorvec,
        jac_prototype = jac_prototype,
        tgrad = _tgrad)
    prob = ODEProblem(f_func, u0, prob.tspan, p;
        prob.kwargs...)
end

function generate_problem(prob::SDEProblem, u0, p, jac_prototype, colorvec)
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    _g = let f = prob.f.g, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
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
                kernel(version)(tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
            end
        end
    else
        _tgrad = nothing
    end

    f_func = SDEFunction(_f, _g, Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec=colorvec,
        jac_prototype = jac_prototype,
        tgrad = _tgrad)
    prob = SDEProblem(f_func, _g, u0, prob.tspan, p;
        prob.kwargs...)
end
