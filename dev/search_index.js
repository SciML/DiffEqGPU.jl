var documenterSearchIndex = {"docs":
[{"location":"diffeqgpu/#API","page":"API","title":"API","text":"","category":"section"},{"location":"diffeqgpu/","page":"API","title":"API","text":"Modules = [DiffEqGPU]","category":"page"},{"location":"diffeqgpu/#DiffEqGPU.DiffEqGPU","page":"API","title":"DiffEqGPU.DiffEqGPU","text":"DiffEqGPU\n\n(Image: Join the chat at https://julialang.zulipchat.com #sciml-bridged) (Image: Global Docs) (Image: Build status)\n\n(Image: codecov)\n\n(Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages) (Image: SciML Code Style)\n\nThis library is a component package of the DifferentialEquations.jl ecosystem. It includes functionality for making use of GPUs in the differential equation solvers.\n\nWithin-Method GPU Parallelism with Direct CuArray Usage\n\nThe native Julia libraries, including (but not limited to) OrdinaryDiffEq, StochasticDiffEq, and DelayDiffEq, are compatible with u0 being a CuArray. When this occurs, all array operations take place on the GPU, including any implicit solves. This is independent of the DiffEqGPU library. These speedup the solution of a differential equation which is sufficiently large or expensive. This does not require DiffEqGPU.jl.\n\nFor example, the following is a GPU-accelerated solve with Tsit5:\n\nusing OrdinaryDiffEq, CUDA, LinearAlgebra\nu0 = cu(rand(1000))\nA  = cu(randn(1000,1000))\nf(du,u,p,t)  = mul!(du,A,u)\nprob = ODEProblem(f,u0,(0.0f0,1.0f0)) # Float32 is better on GPUs!\nsol = solve(prob,Tsit5())\n\nParameter-Parallelism with GPU Ensemble Methods\n\nParameter-parallel GPU methods are provided for the case where a single solve is too cheap to benefit from within-method parallelism, but the solution of the same structure (same f) is required for very many different choices of u0 or p. For these cases, DiffEqGPU exports the following ensemble algorithms:\n\nEnsembleGPUArray: Utilizes the CuArray setup to parallelize ODE solves across the GPU.\nEnsembleGPUKernel: Utilizes the GPU kernels to parallelize each ODE solve with their separate ODE integrator on each kernel. \nEnsembleCPUArray: A test version for analyzing the overhead of the array-based parallelism setup.\n\nFor more information on using the ensemble interface, see the DiffEqDocs page on ensembles\n\nFor example, the following solves the Lorenz equation with 10,000 separate random parameters on the GPU:\n\nusing DiffEqGPU, OrdinaryDiffEq\nfunction lorenz(du,u,p,t)\n    du[1] = p[1]*(u[2]-u[1])\n    du[2] = u[1]*(p[2]-u[3]) - u[2]\n    du[3] = u[1]*u[2] - p[3]*u[3]\nend\n\nu0 = Float32[1.0;0.0;0.0]\ntspan = (0.0f0,100.0f0)\np = [10.0f0,28.0f0,8/3f0]\nprob = ODEProblem(lorenz,u0,tspan,p)\nprob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)\n@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)\n\nEnsembleGPUKernel\n\nThe EnsembleGPUKernel requires a specialized ODE algorithm which is written on the GPU kernel. These implementations are part of DiffEqGPU. These implementations do not allow mutation of arrays, hence use out-of-place (OOP) ODEProblem.\n\nSupport\n\nTsit5: The kernelized version can be called using GPUTsit5() with the EnsembleProblem. \n\nTaking the example above, we simulate the lorenz equation:\n\nusing DiffEqGPU, OrdinaryDiffEq, StaticArrays\n\nfunction lorenz(u, p, t)\n    σ = p[1]\n    ρ = p[2]\n    β = p[3]\n    du1 = σ * (u[2] - u[1])\n    du2 = u[1] * (ρ - u[3]) - u[2]\n    du3 = u[1] * u[2] - β * u[3]\n    return SVector{3}(du1, du2, du3)\nend\n\nu0 = @SVector [1.0f0; 0.0f0; 0.0f0]\ntspan = (0.0f0, 10.0f0)\np = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]\nprob = ODEProblem{false}(lorenz, u0, tspan, p)\nprob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)\n\n@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = false, dt = 0.1f0)\n\n@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = true, dt = 0.1f0, save_everystep = false)\n\nCallbacks with EnsembleGPUKernel\n\nUsing callbacks with EnsembleGPUKernel methods requires their own GPU-compatible callback implementations. MWE:\n\nusing DiffEqGPU, StaticArrays, OrdinaryDiffEq\nfunction f(u, p, t)\n    du1 = -u[1]\n    return SVector{1}(du1)\nend\n\nu0 = @SVector [10.0f0]\nprob = ODEProblem{false}(f, u0, (0.0f0, 10.0f0))\nprob_func = (prob, i, repeat) -> remake(prob, p = prob.p)\nmonteprob = EnsembleProblem(prob, safetycopy = false)\n\ncondition(u, t, integrator) = t == 4.0f0\naffect!(integrator) = integrator.u += @SVector[10.0f0]\n\ngpu_cb = DiscreteCallback(condition, affect!; save_positions = (false, false))\n\nsol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),\n            trajectories = 10,\n            adaptive = false, dt = 0.01f0, callback = gpu_cb, merge_callbacks = true,\n            tstops = [4.0f0])\n\nCurrent Support\n\nAutomated GPU parameter parallelism support is continuing to be improved, so there are currently a few limitations. Not everything is supported yet, but most of the standard features have support, including:\n\nExplicit Runge-Kutta methods\nImplicit Runge-Kutta methods\nRosenbrock methods\nDiscreteCallbacks and ContinuousCallbacks\nMultiple GPUs over clusters\n\nCurrent Limitations\n\nStiff ODEs require the analytical solution of every derivative function it requires. For example, Rosenbrock methods require the Jacobian and the gradient with respect to time, and so these two functions are required to be given. Note that they can be generated by the modelingtoolkitize approach. For example, 10,000 trajectories solved with Rodas5 and TRBDF2 is done via:\n\nfunction lorenz_jac(J,u,p,t)\n    σ = p[1]\n    ρ = p[2]\n    β = p[3]\n    x = u[1]\n    y = u[2]\n    z = u[3]\n    J[1,1] = -σ\n    J[2,1] = ρ - z\n    J[3,1] = y\n    J[1,2] = σ\n    J[2,2] = -1\n    J[3,2] = x\n    J[1,3] = 0\n    J[2,3] = -x\n    J[3,3] = -β\nend\n\nfunction lorenz_tgrad(J,u,p,t)\n    nothing\nend\n\nfunc = ODEFunction(lorenz,jac=lorenz_jac,tgrad=lorenz_tgrad)\nprob_jac = ODEProblem(func,u0,tspan,p)\nmonteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)\n\n@time solve(monteprob_jac,Rodas5(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)\n@time solve(monteprob_jac,TRBDF2(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)\n\nThese limitations are not fundamental and will be eased over time.\n\nSetting Up Multi-GPU\n\nTo setup a multi-GPU environment, first setup a processes such that every process has a different GPU. For example:\n\n# Setup processes with different CUDA devices\nusing Distributed\naddprocs(numgpus)\nimport CUDAdrv, CUDAnative\n\nlet gpuworkers = asyncmap(collect(zip(workers(), CUDAdrv.devices()))) do (p, d)\n  remotecall_wait(CUDAnative.device!, p, d)\n  p\nend\n\nThen setup the calls to work with distributed processes:\n\n@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random\n\n@everywhere begin\n    function lorenz_distributed(du,u,p,t)\n        du[1] = p[1]*(u[2]-u[1])\n        du[2] = u[1]*(p[2]-u[3]) - u[2]\n        du[3] = u[1]*u[2] - p[3]*u[3]\n    end\n    CuArrays.allowscalar(false)\n    u0 = Float32[1.0;0.0;0.0]\n    tspan = (0.0f0,100.0f0)\n    p = [10.0f0,28.0f0,8/3f0]\n    Random.seed!(1)\n    function prob_func_distributed(prob,i,repeat)\n        remake(prob,p=rand(3).*p)\n    end\nend\n\nNow each batch will run on separate GPUs. Thus we need to use the batch_size keyword argument from the Ensemble interface to ensure there are multiple batches. Let's solve 40,000 trajectories, batching 10,000 trajectories at a time:\n\nprob = ODEProblem(lorenz_distributed,u0,tspan,p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)\n\n@time sol2 = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=40_000,\n                                                 batch_size=10_000,saveat=1.0f0)\n\nThis will pmap over the batches, and thus if you have 4 processes each with a GPU, each batch of 10,000 trajectories will be run simultaneously. If you have two processes with two GPUs, this will do two sets of 10,000 at a time.\n\nExample Multi-GPU Script\n\nIn this example we know we have a 2-GPU system (1 eGPU), and we split the work across the two by directly defining the devices on the two worker processes:\n\nusing DiffEqGPU, CuArrays, OrdinaryDiffEq, Test\nCuArrays.device!(0)\n\nusing Distributed\naddprocs(2)\n@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random\n\n@everywhere begin\n    function lorenz_distributed(du,u,p,t)\n        du[1] = p[1]*(u[2]-u[1])\n        du[2] = u[1]*(p[2]-u[3]) - u[2]\n        du[3] = u[1]*u[2] - p[3]*u[3]\n    end\n    CuArrays.allowscalar(false)\n    u0 = Float32[1.0;0.0;0.0]\n    tspan = (0.0f0,100.0f0)\n    p = [10.0f0,28.0f0,8/3f0]\n    Random.seed!(1)\n    pre_p_distributed = [rand(Float32,3) for i in 1:100_000]\n    function prob_func_distributed(prob,i,repeat)\n        remake(prob,p=pre_p_distributed[i].*p)\n    end\nend\n\n@sync begin\n    @spawnat 2 begin\n        CuArrays.allowscalar(false)\n        CuArrays.device!(0)\n    end\n    @spawnat 3 begin\n        CuArrays.allowscalar(false)\n        CuArrays.device!(1)\n    end\nend\n\nCuArrays.allowscalar(false)\nprob = ODEProblem(lorenz_distributed,u0,tspan,p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)\n\n@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,batch_size=50_000,saveat=1.0f0)\n\nOptimal Numbers of Trajectories\n\nThere is a balance between two things for choosing the number of trajectories:\n\nThe number of trajectories needs to be high enough that the work per kernel is sufficient to overcome the kernel call cost.\nMore trajectories means that every trajectory will need more time steps since the adaptivity syncs all solves.\n\nFrom our testing, the balance is found at around 10,000 trajectories being optimal. Thus for larger sets of trajectories, use a batch size of 10,000. Of course, benchmark for yourself on your own setup!\n\n\n\n\n\n","category":"module"},{"location":"diffeqgpu/#DiffEqGPU.EnsembleCPUArray","page":"API","title":"DiffEqGPU.EnsembleCPUArray","text":"An EnsembleArrayAlgorithm for testing the overhead of the array-based parallelism setup.\n\n\n\n\n\n","category":"type"},{"location":"diffeqgpu/#DiffEqGPU.EnsembleGPUArray","page":"API","title":"DiffEqGPU.EnsembleGPUArray","text":"An EnsembleArrayAlgorithm which utilizes the GPU kernels to parallelize each ODE solve with their separate ODE integrator on each kernel.\n\nusing DiffEqGPU, OrdinaryDiffEq\nfunction lorenz(du,u,p,t)\n    du[1] = p[1]*(u[2]-u[1])\n    du[2] = u[1]*(p[2]-u[3]) - u[2]\n    du[3] = u[1]*u[2] - p[3]*u[3]\nend\n\nu0 = Float32[1.0;0.0;0.0]\ntspan = (0.0f0,100.0f0)\np = [10.0f0,28.0f0,8/3f0]\nprob = ODEProblem(lorenz,u0,tspan,p)\nprob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)\n@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)\n\n\n\n\n\n","category":"type"},{"location":"diffeqgpu/#DiffEqGPU.LinSolveGPUSplitFactorize","page":"API","title":"DiffEqGPU.LinSolveGPUSplitFactorize","text":"A parameter-parallel SciMLLinearSolveAlgorithm.\n\n\n\n\n\n","category":"type"},{"location":"diffeqgpu/#DiffEqGPU.Vern9Tableau","page":"API","title":"DiffEqGPU.Vern9Tableau","text":"From Verner's Webiste\n\n\n\n\n\n","category":"type"},{"location":"#DiffEqGPU","page":"Home","title":"DiffEqGPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This library is a component package of the DifferentialEquations.jl ecosystem. It includes functionality for making use of GPUs in the differential equation solvers.","category":"page"},{"location":"#Within-Method-GPU-Parallelism-with-Direct-CuArray-Usage","page":"Home","title":"Within-Method GPU Parallelism with Direct CuArray Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The native Julia libraries, including (but not limited to) OrdinaryDiffEq, StochasticDiffEq, and DelayDiffEq, are compatible with u0 being a CuArray. When this occurs, all array operations take place on the GPU, including any implicit solves. This is independent of the DiffEqGPU library. These speedup the solution of a differential equation which is sufficiently large or expensive. This does not require DiffEqGPU.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For example, the following is a GPU-accelerated solve with Tsit5:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using OrdinaryDiffEq, CUDA, LinearAlgebra\nu0 = cu(rand(1000))\nA  = cu(randn(1000,1000))\nf(du,u,p,t)  = mul!(du,A,u)\nprob = ODEProblem(f,u0,(0.0f0,1.0f0)) # Float32 is better on GPUs!\nsol = solve(prob,Tsit5())","category":"page"},{"location":"#Parameter-Parallelism-with-GPU-Ensemble-Methods","page":"Home","title":"Parameter-Parallelism with GPU Ensemble Methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Parameter-parallel GPU methods are provided for the case where a single solve is too cheap to benefit from within-method parallelism, but the solution of the same structure (same f) is required for very many different choices of u0 or p. For these cases, DiffEqGPU exports the following ensemble algorithms:","category":"page"},{"location":"","page":"Home","title":"Home","text":"EnsembleGPUArray: Utilizes the CuArray setup to parallelize ODE solves across the GPU.\nEnsembleGPUKernel: Utilizes the GPU kernels to parallelize each ODE solve with their separate ODE integrator on each kernel. \nEnsembleCPUArray: A test version for analyzing the overhead of the array-based parallelism setup.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more information on using the ensemble interface, see the DiffEqDocs page on ensembles","category":"page"},{"location":"","page":"Home","title":"Home","text":"For example, the following solves the Lorenz equation with 10,000 separate random parameters on the GPU:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using DiffEqGPU, OrdinaryDiffEq\nfunction lorenz(du,u,p,t)\n    du[1] = p[1]*(u[2]-u[1])\n    du[2] = u[1]*(p[2]-u[3]) - u[2]\n    du[3] = u[1]*u[2] - p[3]*u[3]\nend\n\nu0 = Float32[1.0;0.0;0.0]\ntspan = (0.0f0,100.0f0)\np = [10.0f0,28.0f0,8/3f0]\nprob = ODEProblem(lorenz,u0,tspan,p)\nprob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)\n@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)","category":"page"},{"location":"#EnsembleGPUKernel","page":"Home","title":"EnsembleGPUKernel","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The EnsembleGPUKernel requires a specialized ODE algorithm which is written on the GPU kernel. These implementations are part of DiffEqGPU. These implementations do not allow mutation of arrays, hence use out-of-place (OOP) ODEProblem.","category":"page"},{"location":"#Support","page":"Home","title":"Support","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tsit5: The kernelized version can be called using GPUTsit5() with the EnsembleProblem. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Taking the example above, we simulate the lorenz equation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using DiffEqGPU, OrdinaryDiffEq, StaticArrays\n\nfunction lorenz(u, p, t)\n    σ = p[1]\n    ρ = p[2]\n    β = p[3]\n    du1 = σ * (u[2] - u[1])\n    du2 = u[1] * (ρ - u[3]) - u[2]\n    du3 = u[1] * u[2] - β * u[3]\n    return SVector{3}(du1, du2, du3)\nend\n\nu0 = @SVector [1.0f0; 0.0f0; 0.0f0]\ntspan = (0.0f0, 10.0f0)\np = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]\nprob = ODEProblem{false}(lorenz, u0, tspan, p)\nprob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)\n\n@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = false, dt = 0.1f0)\n\n@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000, adaptive = true, dt = 0.1f0, save_everystep = false)","category":"page"},{"location":"#Callbacks-with-EnsembleGPUKernel","page":"Home","title":"Callbacks with EnsembleGPUKernel","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Using callbacks with EnsembleGPUKernel methods requires their own GPU-compatible callback implementations. MWE:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using DiffEqGPU, StaticArrays, OrdinaryDiffEq\nfunction f(u, p, t)\n    du1 = -u[1]\n    return SVector{1}(du1)\nend\n\nu0 = @SVector [10.0f0]\nprob = ODEProblem{false}(f, u0, (0.0f0, 10.0f0))\nprob_func = (prob, i, repeat) -> remake(prob, p = prob.p)\nmonteprob = EnsembleProblem(prob, safetycopy = false)\n\ncondition(u, t, integrator) = t == 4.0f0\naffect!(integrator) = integrator.u += @SVector[10.0f0]\n\ngpu_cb = DiscreteCallback(condition, affect!; save_positions = (false, false))\n\nsol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),\n            trajectories = 10,\n            adaptive = false, dt = 0.01f0, callback = gpu_cb, merge_callbacks = true,\n            tstops = [4.0f0])","category":"page"},{"location":"#Current-Support","page":"Home","title":"Current Support","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Automated GPU parameter parallelism support is continuing to be improved, so there are currently a few limitations. Not everything is supported yet, but most of the standard features have support, including:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Explicit Runge-Kutta methods\nImplicit Runge-Kutta methods\nRosenbrock methods\nDiscreteCallbacks and ContinuousCallbacks\nMultiple GPUs over clusters","category":"page"},{"location":"#Current-Limitations","page":"Home","title":"Current Limitations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Stiff ODEs require the analytical solution of every derivative function it requires. For example, Rosenbrock methods require the Jacobian and the gradient with respect to time, and so these two functions are required to be given. Note that they can be generated by the modelingtoolkitize approach. For example, 10,000 trajectories solved with Rodas5 and TRBDF2 is done via:","category":"page"},{"location":"","page":"Home","title":"Home","text":"function lorenz_jac(J,u,p,t)\n    σ = p[1]\n    ρ = p[2]\n    β = p[3]\n    x = u[1]\n    y = u[2]\n    z = u[3]\n    J[1,1] = -σ\n    J[2,1] = ρ - z\n    J[3,1] = y\n    J[1,2] = σ\n    J[2,2] = -1\n    J[3,2] = x\n    J[1,3] = 0\n    J[2,3] = -x\n    J[3,3] = -β\nend\n\nfunction lorenz_tgrad(J,u,p,t)\n    nothing\nend\n\nfunc = ODEFunction(lorenz,jac=lorenz_jac,tgrad=lorenz_tgrad)\nprob_jac = ODEProblem(func,u0,tspan,p)\nmonteprob_jac = EnsembleProblem(prob_jac, prob_func = prob_func)\n\n@time solve(monteprob_jac,Rodas5(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)\n@time solve(monteprob_jac,TRBDF2(),EnsembleGPUArray(),dt=0.1,trajectories=10_000,saveat=1.0f0)","category":"page"},{"location":"","page":"Home","title":"Home","text":"These limitations are not fundamental and will be eased over time.","category":"page"},{"location":"#Setting-Up-Multi-GPU","page":"Home","title":"Setting Up Multi-GPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To setup a multi-GPU environment, first setup a processes such that every process has a different GPU. For example:","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Setup processes with different CUDA devices\nusing Distributed\naddprocs(numgpus)\nimport CUDAdrv, CUDAnative\n\nlet gpuworkers = asyncmap(collect(zip(workers(), CUDAdrv.devices()))) do (p, d)\n  remotecall_wait(CUDAnative.device!, p, d)\n  p\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then setup the calls to work with distributed processes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random\n\n@everywhere begin\n    function lorenz_distributed(du,u,p,t)\n        du[1] = p[1]*(u[2]-u[1])\n        du[2] = u[1]*(p[2]-u[3]) - u[2]\n        du[3] = u[1]*u[2] - p[3]*u[3]\n    end\n    CuArrays.allowscalar(false)\n    u0 = Float32[1.0;0.0;0.0]\n    tspan = (0.0f0,100.0f0)\n    p = [10.0f0,28.0f0,8/3f0]\n    Random.seed!(1)\n    function prob_func_distributed(prob,i,repeat)\n        remake(prob,p=rand(3).*p)\n    end\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now each batch will run on separate GPUs. Thus we need to use the batch_size keyword argument from the Ensemble interface to ensure there are multiple batches. Let's solve 40,000 trajectories, batching 10,000 trajectories at a time:","category":"page"},{"location":"","page":"Home","title":"Home","text":"prob = ODEProblem(lorenz_distributed,u0,tspan,p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)\n\n@time sol2 = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=40_000,\n                                                 batch_size=10_000,saveat=1.0f0)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This will pmap over the batches, and thus if you have 4 processes each with a GPU, each batch of 10,000 trajectories will be run simultaneously. If you have two processes with two GPUs, this will do two sets of 10,000 at a time.","category":"page"},{"location":"#Example-Multi-GPU-Script","page":"Home","title":"Example Multi-GPU Script","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In this example we know we have a 2-GPU system (1 eGPU), and we split the work across the two by directly defining the devices on the two worker processes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test\nCuArrays.device!(0)\n\nusing Distributed\naddprocs(2)\n@everywhere using DiffEqGPU, CuArrays, OrdinaryDiffEq, Test, Random\n\n@everywhere begin\n    function lorenz_distributed(du,u,p,t)\n        du[1] = p[1]*(u[2]-u[1])\n        du[2] = u[1]*(p[2]-u[3]) - u[2]\n        du[3] = u[1]*u[2] - p[3]*u[3]\n    end\n    CuArrays.allowscalar(false)\n    u0 = Float32[1.0;0.0;0.0]\n    tspan = (0.0f0,100.0f0)\n    p = [10.0f0,28.0f0,8/3f0]\n    Random.seed!(1)\n    pre_p_distributed = [rand(Float32,3) for i in 1:100_000]\n    function prob_func_distributed(prob,i,repeat)\n        remake(prob,p=pre_p_distributed[i].*p)\n    end\nend\n\n@sync begin\n    @spawnat 2 begin\n        CuArrays.allowscalar(false)\n        CuArrays.device!(0)\n    end\n    @spawnat 3 begin\n        CuArrays.allowscalar(false)\n        CuArrays.device!(1)\n    end\nend\n\nCuArrays.allowscalar(false)\nprob = ODEProblem(lorenz_distributed,u0,tspan,p)\nmonteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)\n\n@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,batch_size=50_000,saveat=1.0f0)","category":"page"},{"location":"#Optimal-Numbers-of-Trajectories","page":"Home","title":"Optimal Numbers of Trajectories","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"There is a balance between two things for choosing the number of trajectories:","category":"page"},{"location":"","page":"Home","title":"Home","text":"The number of trajectories needs to be high enough that the work per kernel is sufficient to overcome the kernel call cost.\nMore trajectories means that every trajectory will need more time steps since the adaptivity syncs all solves.","category":"page"},{"location":"","page":"Home","title":"Home","text":"From our testing, the balance is found at around 10,000 trajectories being optimal. Thus for larger sets of trajectories, use a batch size of 10,000. Of course, benchmark for yourself on your own setup!","category":"page"},{"location":"#Reproducibility","page":"Home","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status(;mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can also download the \n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Manifest.toml\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">manifest</a> file and the\n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Project.toml\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">project</a> file.","category":"page"}]
}
