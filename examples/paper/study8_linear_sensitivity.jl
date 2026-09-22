# Mesh and overlap sweeps used by the linear-source scaling table.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :gmres_ras) || include("study8_linear_cmp.jl")

const STUDY8_SCALING_METHODS = (
  "var_dd_additive",
  "var_dd_additive_history",
  "emdd_q3",
  "emdd_q4",
  "pcg_as",
  "gmres_ras",
)

function study8_scaling_history(
  method, K, b, dofs, schwarz; maxiter, relative_tolerance,
)
  initial = norm(b - K * ones(length(b)))
  tolerance = relative_tolerance * initial
  history = if method == "var_dd_additive"
    var_dd_linear_history(K, b, dofs; maxiter, tol=tolerance, history_depth=0)
  elseif method == "var_dd_additive_history"
    var_dd_linear_history(K, b, dofs; maxiter, tol=tolerance, history_depth=1)
  elseif method in ("emdd_q3", "emdd_q4")
    depth = parse(Int, string(last(method))) - 1
    var_dd_linear_history(K, b, dofs; maxiter, tol=tolerance, history_depth=depth)
  elseif method == "pcg_as"
    pcg_as(K, b, schwarz; maxiter, tol=tolerance)
  elseif method == "gmres_ras"
    gmres_ras(K, b, schwarz; maxiter, tol=tolerance)
  else
    error("unknown scaling method $method")
  end
  return [(solves, residual / initial) for (solves, residual) in history]
end

function run_study8_sensitivity()
  file = "study8_sensitivity.csv"
  if !needs_run(file)
    println("study8 sensitivity: cached, skipping")
    return
  end
  Random.seed!(1)
  tolerance = SMALL ? 1e-7 : 1e-10
  maxiterations = SMALL ? 35 : 300
  rows = (
    experiment=String[], regime=String[], method=String[], N=Int[], m=Int[],
    overlap=Int[], contrast=Float64[], history_depth=Int[],
    delta_over_H=Float64[], iteration=Int[], local_batches=Int[],
    local_solves=Int[], global_matvecs=Int[], relative_residual=Float64[],
  )

  function run_configuration(experiment, regime, N, m, overlap;
                             cell_owners=nothing)
    K, _, b, dofs, _, core = laplace_setup(
      N, m, overlap;
      diffusion=x -> 1.0,
      cell_owners,
      return_core_partition=true,
    )
    schwarz = schwarz_setup(K, dofs; core_dofs=core)
    for method in STUDY8_SCALING_METHODS
      history = study8_scaling_history(
        method, K, b, dofs, schwarz;
        maxiter=maxiterations, relative_tolerance=tolerance,
      )
      for (iteration, (solves, residual)) in enumerate(history)
        batches = solves ÷ m
        push!(rows.experiment, experiment)
        push!(rows.regime, regime)
        push!(rows.method, method)
        push!(rows.N, N)
        push!(rows.m, m)
        push!(rows.overlap, overlap)
        push!(rows.contrast, 1.0)
        push!(rows.history_depth,
          method == "var_dd_additive_history" ? 1 :
          method == "emdd_q3" ? 2 : method == "emdd_q4" ? 3 : 0)
        push!(rows.delta_over_H, overlap * sqrt(m) / N)
        push!(rows.iteration, iteration - 1)
        push!(rows.local_batches, batches)
        push!(rows.local_solves, solves)
        push!(rows.global_matvecs,
          method in ("pcg_as", "gmres_ras") ? 1 + batches : -1)
        push!(rows.relative_residual, residual)
      end
    end
    println("  $experiment/$regime: N=$N, overlap=$overlap")
  end

  mesh_sizes = SMALL ? [10, 20, 30] : collect(20:20:120)
  m = 4
  reference_owners = metis_cell_owners(first(mesh_sizes), m)
  for N in mesh_sizes
    owners = prolong_cell_owners(reference_owners, N)
    run_configuration("mesh", "fixed_layers", N, m, 2; cell_owners=owners)
    overlap = SMALL ? max(1, N ÷ 10) : N ÷ 20
    run_configuration(
      "mesh", "fixed_delta_over_H", N, m, overlap; cell_owners=owners
    )
  end
  N = SMALL ? 16 : 64
  for overlap in (SMALL ? [1, 2] : [1, 2, 4, 8])
    run_configuration("overlap", "layer_sweep", N, m, overlap)
  end
  savetable(file, rows)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_study8_sensitivity()
