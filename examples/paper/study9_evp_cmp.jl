# Study 9: smallest generalized eigenpair. All methods use the same initial
# vector, true relative residual, overlapping partitions, and one-level AS
# blocks. The local work categories are retained separately because a varDD
# local eigenproblem is not equivalent to an AS triangular solve.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :EVP_COMMON) || include("evp_common.jl")

function append_evp_history!(rows, method, m, lambda_reference, result)
  for entry in result.history
    push!(rows.method, string(method))
    push!(rows.m, m)
    push!(rows.iteration, entry.iteration)
    push!(rows.lambda, entry.lambda)
    push!(rows.eigenvalue_error, abs(entry.lambda - lambda_reference))
    push!(rows.residual, entry.residual)
    push!(rows.relative_residual, entry.relative_residual)
    push!(rows.local_eigen_batches, entry.local_eigen_batches)
    push!(rows.linear_as_batches, entry.linear_as_batches)
    push!(rows.local_iterations_critical, entry.local_iterations_critical)
    push!(rows.local_iterations_total, entry.local_iterations_total)
    push!(rows.global_k_products, entry.global_k_products)
    push!(rows.global_m_products, entry.global_m_products)
    push!(rows.inner_iterations, entry.inner_iterations)
  end
end

function append_evp_diagnostics!(local_rows, combination_rows, inner_rows, method, m, result)
  for stat in result.local_stats
    push!(local_rows.method, string(method))
    push!(local_rows.m, m)
    push!(local_rows.outer_iteration, stat.outer_iteration)
    push!(local_rows.subdomain, stat.subdomain)
    push!(local_rows.dimension, stat.dimension)
    push!(local_rows.iterations, stat.iterations)
    push!(local_rows.converged, Int(stat.converged))
    push!(local_rows.residual, stat.residual)
    push!(local_rows.k_nnz, stat.k_nnz)
    push!(local_rows.factor_nnz, stat.factor_nnz)
  end
  for stat in result.combination_stats
    push!(combination_rows.method, string(method))
    push!(combination_rows.m, m)
    push!(combination_rows.outer_iteration, stat.outer_iteration)
    push!(combination_rows.basis_columns, stat.basis_columns)
    push!(combination_rows.effective_rank, stat.effective_rank)
    push!(combination_rows.mass_condition, stat.mass_condition)
    push!(combination_rows.relative_gap, stat.relative_gap)
  end
  for stat in result.inner_stats
    push!(inner_rows.method, string(method))
    push!(inner_rows.m, m)
    push!(inner_rows.outer_iteration, stat.outer_iteration)
    push!(inner_rows.iterations, stat.iterations)
    push!(inner_rows.converged, Int(stat.converged))
    push!(inner_rows.relative_residual, stat.relative_residual)
  end
end

function run_study9()
  files = (
    "study9_evp_cmp.csv",
    "study9_partitions.csv",
    "study9_evp_local_stats.csv",
    "study9_evp_combination_stats.csv",
    "study9_evp_inner_stats.csv",
  )
  if !needs_run(files...)
    println("study9: cached, skipping")
    return
  end
  println("study9: EVP vs LOPSD, LOBPCG, JD, and shift-invert Lanczos with AS")
  Random.seed!(1)

  N = SMALL ? 10 : 40
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  relative_tolerance = SMALL ? 1e-5 : 1e-6
  maxiter = SMALL ? 12 : 100

  rows = (
    method=String[], m=Int[], iteration=Int[], lambda=Float64[],
    eigenvalue_error=Float64[], residual=Float64[], relative_residual=Float64[],
    local_eigen_batches=Int[], linear_as_batches=Int[],
    local_iterations_critical=Int[], local_iterations_total=Int[],
    global_k_products=Int[], global_m_products=Int[], inner_iterations=Int[],
  )
  local_rows = (
    method=String[], m=Int[], outer_iteration=Int[], subdomain=Int[],
    dimension=Int[], iterations=Int[], converged=Int[], residual=Float64[],
    k_nnz=Int[], factor_nnz=Int[],
  )
  combination_rows = (
    method=String[], m=Int[], outer_iteration=Int[], basis_columns=Int[],
    effective_rank=Int[], mass_condition=Float64[], relative_gap=Float64[],
  )
  inner_rows = (
    method=String[], m=Int[], outer_iteration=Int[], iterations=Int[],
    converged=Int[], relative_residual=Float64[],
  )
  part_rows = (m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[])

  for m in ms
    K, M, _, dofspar, _, core = schroedinger_setup(
      N, m, overlap; return_core_partition=true
    )
    schwarz = schwarz_setup(K, dofspar; core_dofs=core)
    lambda_reference = dense_reference_lambda(K, M)

    owner, mult = metis_cell_partition(N, m, overlap)
    for idx in eachindex(owner)
      push!(part_rows.m, m)
      push!(part_rows.N, N)
      push!(part_rows.idx, idx)
      push!(part_rows.owner, owner[idx])
      push!(part_rows.mult, mult[idx])
    end

    for method in EVP_COMPARISON_METHODS
      result = evp_method_result(
        method,
        K,
        M,
        dofspar,
        schwarz;
        maxiter,
        relative_tolerance,
        history_depth=1,
      )
      append_evp_history!(rows, method, m, lambda_reference, result)
      append_evp_diagnostics!(
        local_rows, combination_rows, inner_rows, method, m, result
      )
      terminal = last(result.history)
      @printf(
        "  m=%d, %-22s: %3d outer, relres %.2e, AS batches %d\n",
        m,
        string(method),
        terminal.iteration,
        terminal.relative_residual,
        terminal.linear_as_batches,
      )
    end
  end
  savetable("study9_evp_cmp.csv", rows)
  savetable("study9_partitions.csv", part_rows)
  savetable("study9_evp_local_stats.csv", local_rows)
  savetable("study9_evp_combination_stats.csv", combination_rows)
  savetable("study9_evp_inner_stats.csv", inner_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study9()
end
