# Pkg.test runs with --check_bounds=1, forcing all bounds checks.
# This is incompatible with GPUifyLoops.
if Base.JLOptions().check_bounds == 1
  cmd = Cmd(filter(arg->!startswith(arg, "--check-bounds"), Base.julia_cmd().exec))
  code = """
    $(Base.load_path_setup_code(false))
    cd($(repr(@__DIR__)))
    include($(repr(@__FILE__)))
    """
  run(`$cmd --eval $code`)
  exit()
end
@assert Base.JLOptions().check_bounds == 0

using Safetestsets, Test

@time @safetestset begin include("ensemblegpuarray.jl") end end
@time @testset begin include("distributed_multi_gpu.jl") end end
