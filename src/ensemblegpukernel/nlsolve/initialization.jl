@inline function gpu_initialization_solve(prob, nlsolve_alg, abstol, reltol)
    f = prob.f
    u0 = prob.u0
    p = prob.p
    
    # Check if initialization is actually needed
    if !SciMLBase.has_initialization_data(f) || f.initialization_data === nothing
        return u0, p, true
    end
    
    initdata = f.initialization_data
    if initdata.initializeprob === nothing
        return u0, p, true
    end
    
    # Use SimpleNonlinearSolve directly - it's GPU compatible
    try
        # Default to SimpleTrustRegion if no algorithm specified
        alg = nlsolve_alg === nothing ? SimpleTrustRegion() : nlsolve_alg
        
        # Create initialization problem
        initprob = initdata.initializeprob
        
        # Update the problem if needed
        if initdata.update_initializeprob! !== nothing
            if initdata.is_update_oop === Val(true)
                initprob = initdata.update_initializeprob!(initprob, (u=u0, p=p))
            else
                initdata.update_initializeprob!(initprob, (u=u0, p=p))
            end
        end
        
        # Solve initialization problem using SimpleNonlinearSolve
        sol = solve(initprob, alg; abstol, reltol)
        
        # Extract results
        if SciMLBase.successful_retcode(sol)
            # Apply result mappings if they exist
            u_init = if initdata.initializeprobmap !== nothing
                initdata.initializeprobmap(sol)
            else
                u0
            end
            
            p_init = if initdata.initializeprobpmap !== nothing  
                initdata.initializeprobpmap((u=u0, p=p), sol)
            else
                p
            end
            
            return u_init, p_init, true
        else
            # If initialization fails, use original values
            return u0, p, false
        end
    catch
        # If anything goes wrong, fall back to original values
        return u0, p, false
    end
end