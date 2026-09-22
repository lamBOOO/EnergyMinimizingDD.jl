# Driver for every numerical result in the final paper.
#
#   julia --project=. examples/paper/run_all.jl          # publication run
#   SMALL=1 julia --project=. examples/paper/run_all.jl  # smoke test
#   FORCE=1 ...                                          # ignore cached CSVs
#
# Studies write CSVs into data/; figures are then regenerated from the CSVs.

include("common.jl")
include("study8_linear_cmp.jl")
include("study8_linear_sensitivity.jl")
include("study9_evp_cmp.jl")
include("study10_gp.jl")
include("study10_gp_sensitivity.jl")
include("study11_semilinear.jl")
include("study11b_semilinear_l_shape.jl")

t0 = time()
run_study8()
run_study8_sensitivity()
run_study9()
run_study10()
run_study10_sensitivity()
run_study11()
run_study11b()
include("plots_all.jl")
make_all_figures()
if SMALL
  println("tables: skipped in SMALL mode (publication tables require the full parameter grids)")
else
  include("tables_all.jl")
  make_fig18_table()
end
@printf("run_all: finished in %.1f min\n", (time() - t0) / 60)
