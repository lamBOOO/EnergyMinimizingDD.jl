# Driver for the full Section 4 numerical results suite.
#
#   julia --project=. examples/paper/run_all.jl          # full run (~30-60 min)
#   SMALL=1 julia --project=. examples/paper/run_all.jl  # smoke test (~2 min)
#   FORCE=1 ...                                          # ignore cached CSVs
#
# Studies write CSVs into data/; figures are then regenerated from the CSVs.

include("common.jl")
include("study1_setup.jl")
include("study2_evp.jl")
include("study3_robustness.jl")
include("study4_poisson.jl")
include("study5_plap.jl")
include("study6_heat.jl")
include("study7_local3d.jl")
include("study8_linear_cmp.jl")
include("study9_evp_cmp.jl")

t0 = time()
run_study1()
run_study2()
run_study3()
run_study4()
run_study5()
run_study6()
run_study7()
run_study8()
run_study9()
include("plots_all.jl")
make_all_figures()
@printf("run_all: finished in %.1f min\n", (time() - t0) / 60)
