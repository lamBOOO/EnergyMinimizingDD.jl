# Study 9 sensitivity: mesh/overlap, resolved oscillatory diffusion, varDD
# history depth, and local eigensolve/AS diagnostics.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :EVP_COMMON) || include("evp_common.jl")

const EVP_SENSITIVITY_METHODS = (
  :var_dd,
  :var_dd_history,
  :lobpcg_as,
  :jd_gmres_as,
  :si_lanczos_pcg_as_2,
  :si_lanczos_pcg_as_4,
  :si_lanczos_pcg_as_8,
  :si_lanczos_pcg_as_16,
  :si_lanczos_pcg_as_32,
)

function oscillatory_diffusion(frequency; contrast=1e3)
  return x -> 1 + (contrast - 1) * (
    1 + sin(2pi * frequency * x.data[1]) *
        sin(2pi * frequency * x.data[2])
  ) / 2
end

function run_study9_sensitivity()
  files = (
    "study9_evp_sensitivity.csv",
    "study9_evp_sensitivity_local.csv",
    "study9_evp_sensitivity_combination.csv",
    "study9_evp_sensitivity_inner.csv",
  )
  if !needs_run(files...)
    println("study9 sensitivity: cached, skipping")
    return
  end
  println("study9 sensitivity: mesh, overlap, oscillation, history, local work")
  Random.seed!(1)

  convergence = (
    experiment=String[], regime=String[], method=String[], N=Int[], m=Int[],
    overlap=Int[], frequency=Int[], contrast=Float64[], history_depth=Int[],
    delta_over_H=Float64[], iteration=Int[], lambda=Float64[],
    relative_residual=Float64[], local_eigen_batches=Int[],
    linear_as_batches=Int[], local_iterations_critical=Int[],
    local_iterations_total=Int[], global_k_products=Int[],
    global_m_products=Int[], inner_iterations=Int[], converged=Int[],
  )
  local_rows = (
    experiment=String[], regime=String[], method=String[], N=Int[], m=Int[],
    overlap=Int[], frequency=Int[], contrast=Float64[], history_depth=Int[],
    outer_iteration=Int[], subdomain=Int[], system=String[], dimension=Int[],
    iterations=Int[], converged=Int[], residual=Float64[], k_nnz=Int[],
    factor_nnz=Int[],
  )
  combination = (
    experiment=String[], regime=String[], method=String[], N=Int[], m=Int[],
    overlap=Int[], frequency=Int[], contrast=Float64[], history_depth=Int[],
    outer_iteration=Int[], basis_columns=Int[], effective_rank=Int[],
    mass_condition=Float64[], relative_gap=Float64[],
  )
  inner = (
    experiment=String[], regime=String[], method=String[], N=Int[], m=Int[],
    overlap=Int[], frequency=Int[], contrast=Float64[], history_depth=Int[],
    outer_iteration=Int[], iterations=Int[], converged=Int[],
    relative_residual=Float64[],
  )

  function run_configuration(
    experiment,
    regime,
    N,
    m,
    overlap;
    frequency=0,
    contrast=1.0,
    methods=EVP_SENSITIVITY_METHODS,
    history_depth=1,
    partitioning=:metis,
  )
    diffusion = frequency == 0 ? (x -> 1.0) :
                oscillatory_diffusion(frequency; contrast)
    K, M, _, dofspar, _, core = schroedinger_setup(
      N,
      m,
      overlap;
      diffusion,
      partitioning,
      return_core_partition=true,
    )
    schwarz = schwarz_setup(K, dofspar; core_dofs=core)
    delta_over_H = overlap * sqrt(m) / N
    maxiter = SMALL ? 15 : 120
    relative_tolerance = SMALL ? 1e-5 : 1e-6

    # Static local AS setup statistics are recorded once per configuration.
    for (subdomain, (dofs, factor)) in enumerate(zip(schwarz.dofs, schwarz.facts))
      block = sparse(K[dofs, dofs])
      push!(local_rows.experiment, String(experiment))
      push!(local_rows.regime, String(regime))
      push!(local_rows.method, "as_setup")
      push!(local_rows.N, N)
      push!(local_rows.m, m)
      push!(local_rows.overlap, overlap)
      push!(local_rows.frequency, frequency)
      push!(local_rows.contrast, contrast)
      push!(local_rows.history_depth, 0)
      push!(local_rows.outer_iteration, 0)
      push!(local_rows.subdomain, subdomain)
      push!(local_rows.system, "schwarz_block")
      push!(local_rows.dimension, length(dofs))
      push!(local_rows.iterations, 0)
      push!(local_rows.converged, 1)
      push!(local_rows.residual, 0.0)
      push!(local_rows.k_nnz, nnz(block))
      push!(local_rows.factor_nnz, nnz(sparse(factor.L)))
    end

    for method in methods
      result = evp_method_result(
        method,
        K,
        M,
        dofspar,
        schwarz;
        maxiter,
        relative_tolerance,
        history_depth,
      )
      terminal_converged = last(result.history).relative_residual <= relative_tolerance
      for entry in result.history
        push!(convergence.experiment, String(experiment))
        push!(convergence.regime, String(regime))
        push!(convergence.method, string(method))
        push!(convergence.N, N)
        push!(convergence.m, m)
        push!(convergence.overlap, overlap)
        push!(convergence.frequency, frequency)
        push!(convergence.contrast, contrast)
        push!(convergence.history_depth, method == :var_dd_history ? history_depth : 0)
        push!(convergence.delta_over_H, delta_over_H)
        push!(convergence.iteration, entry.iteration)
        push!(convergence.lambda, entry.lambda)
        push!(convergence.relative_residual, entry.relative_residual)
        push!(convergence.local_eigen_batches, entry.local_eigen_batches)
        push!(convergence.linear_as_batches, entry.linear_as_batches)
        push!(convergence.local_iterations_critical, entry.local_iterations_critical)
        push!(convergence.local_iterations_total, entry.local_iterations_total)
        push!(convergence.global_k_products, entry.global_k_products)
        push!(convergence.global_m_products, entry.global_m_products)
        push!(convergence.inner_iterations, entry.inner_iterations)
        push!(convergence.converged, Int(terminal_converged))
      end
      for stat in result.local_stats
        push!(local_rows.experiment, String(experiment))
        push!(local_rows.regime, String(regime))
        push!(local_rows.method, string(method))
        push!(local_rows.N, N)
        push!(local_rows.m, m)
        push!(local_rows.overlap, overlap)
        push!(local_rows.frequency, frequency)
        push!(local_rows.contrast, contrast)
        push!(local_rows.history_depth, method == :var_dd_history ? history_depth : 0)
        push!(local_rows.outer_iteration, stat.outer_iteration)
        push!(local_rows.subdomain, stat.subdomain)
        push!(local_rows.system, "vardd_augmented_pencil")
        push!(local_rows.dimension, stat.dimension)
        push!(local_rows.iterations, stat.iterations)
        push!(local_rows.converged, Int(stat.converged))
        push!(local_rows.residual, stat.residual)
        push!(local_rows.k_nnz, stat.k_nnz)
        push!(local_rows.factor_nnz, stat.factor_nnz)
      end
      for stat in result.combination_stats
        push!(combination.experiment, String(experiment))
        push!(combination.regime, String(regime))
        push!(combination.method, string(method))
        push!(combination.N, N)
        push!(combination.m, m)
        push!(combination.overlap, overlap)
        push!(combination.frequency, frequency)
        push!(combination.contrast, contrast)
        push!(combination.history_depth, method == :var_dd_history ? history_depth : 0)
        push!(combination.outer_iteration, stat.outer_iteration)
        push!(combination.basis_columns, stat.basis_columns)
        push!(combination.effective_rank, stat.effective_rank)
        push!(combination.mass_condition, stat.mass_condition)
        push!(combination.relative_gap, stat.relative_gap)
      end
      for stat in result.inner_stats
        push!(inner.experiment, String(experiment))
        push!(inner.regime, String(regime))
        push!(inner.method, string(method))
        push!(inner.N, N)
        push!(inner.m, m)
        push!(inner.overlap, overlap)
        push!(inner.frequency, frequency)
        push!(inner.contrast, contrast)
        push!(inner.history_depth, method == :var_dd_history ? history_depth : 0)
        push!(inner.outer_iteration, stat.outer_iteration)
        push!(inner.iterations, stat.iterations)
        push!(inner.converged, Int(stat.converged))
        push!(inner.relative_residual, stat.relative_residual)
      end
    end
    println(
      "  $experiment/$regime: N=$N, m=$m, overlap=$overlap, " *
      "frequency=$frequency, contrast=$contrast",
    )
  end

  # h, h/2, h/4. The second sequence holds delta/H fixed.
  mesh_sizes = SMALL ? [8, 12, 16] : [20, 40, 80]
  for N in mesh_sizes
    run_configuration(
      "mesh", "fixed_layers", N, 4, 2; partitioning=:cartesian
    )
    scaled_overlap = SMALL ? max(1, N ÷ 8) : N ÷ 20
    run_configuration(
      "mesh",
      "fixed_delta_over_H",
      N,
      4,
      scaled_overlap;
      partitioning=:cartesian,
    )
  end

  # Resolved oscillations: N/frequency >= 8 in the full experiment.
  oscillatory_N = SMALL ? 16 : 64
  frequencies = SMALL ? [1, 2] : [1, 2, 4, 8]
  oscillatory_overlap = SMALL ? 1 : 3
  for frequency in frequencies
    run_configuration(
      "oscillation",
      "frequency_sweep",
      oscillatory_N,
      4,
      oscillatory_overlap;
      frequency,
      contrast=1e3,
      partitioning=:cartesian,
    )
  end

  history_depths = SMALL ? [0, 1, 2] : [0, 1, 2, 4, 8]
  history_N = SMALL ? 12 : 40
  for depth in history_depths
    run_configuration(
      "history",
      "default_potential",
      history_N,
      4,
      2;
      methods=(:var_dd_history,),
      history_depth=depth,
    )
  end

  savetable("study9_evp_sensitivity.csv", convergence)
  savetable("study9_evp_sensitivity_local.csv", local_rows)
  savetable("study9_evp_sensitivity_combination.csv", combination)
  savetable("study9_evp_sensitivity_inner.csv", inner)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study9_sensitivity()
end
