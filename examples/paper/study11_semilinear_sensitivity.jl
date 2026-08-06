# Study 11 sensitivity experiments for the exponential semilinear source
# problem. The experiments vary one design choice at a time:
#   - h, h/2, h/4 with overlap layers 1, 2, 4 (fixed physical overlap),
#   - the number of PCG(AS) steps in each inexact Newton update,
#   - Anderson-RAS and varDD history depth.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :SEMILINEAR_SOURCE_METHODS) ||
  include("nonlinear_source_common.jl")
isdefined(Main, :SEMILINEAR_MODES) || include("study11_semilinear.jl")

function semilinear_sensitivity_setup(N, m, overlap)
  ea, ga, ha, dofs, _, ndofs, K, initial, core, M =
    FEMDiscretizations.FEM_SemilinearPoisson(
      N,
      m;
      potential=s -> exp(-s),
      potential_gradient=s -> -exp(-s),
      potential_hessian=s -> exp(-s),
      forcing=semilinear_forcing,
      overlap=overlap,
      quadrature_degree=8,
      initial_guess=x -> 0.0,
      return_mass_matrix=true,
    )
  energy = Energies.NonlinearEnergy(
    "exponential semilinear Poisson",
    ea,
    ga,
    ha,
    ndofs,
  )
  return (; energy, dofs, core, K, M, initial)
end

function run_semilinear_sensitivity_method(
  method,
  problem;
  maxiter,
  tolerance,
  parameter=0,
)
  if method == :nonlinear_ras
    return nonlinear_source_schwarz_baseline(
      problem.energy,
      problem.dofs;
      method=:nonlinear_ras,
      core_subdomains=problem.core,
      u0=problem.initial,
      maxiter,
      tolerance,
    )
  elseif method == :anderson_ras
    return nonlinear_source_anderson_ras(
      problem.energy,
      problem.dofs,
      problem.core;
      u0=problem.initial,
      maxiter,
      tolerance,
      history_depth=parameter,
    )
  elseif method == :newton_pcg_as
    accurate = parameter == 0
    return nonlinear_source_newton_pcg_as(
      problem.energy,
      problem.dofs;
      u0=problem.initial,
      maxiter,
      tolerance,
      inner_iterations=accurate ? 300 : parameter,
      inner_relative_tolerance=accurate ? 1e-10 : 0.0,
    )
  elseif method == :energy_imex_pcg_as
    return nonlinear_source_energy_imex_pcg_as(
      problem.energy,
      problem.K,
      problem.M,
      problem.dofs;
      u0=problem.initial,
      maxiter,
      tolerance,
      timestep=1.0,
      inner_maxiter=300,
      inner_relative_tolerance=1e-10,
    )
  elseif method == :raspen
    return nonlinear_source_raspen(
      problem.energy,
      problem.dofs,
      problem.core;
      u0=problem.initial,
      maxiter,
      tolerance,
      inner_maxiter=60,
      inner_relative_tolerance=1e-6,
    )
  elseif method == :aspin
    return nonlinear_source_aspin(
      problem.energy,
      problem.dofs;
      u0=problem.initial,
      maxiter,
      tolerance,
      inner_maxiter=60,
      inner_relative_tolerance=1e-6,
    )
  elseif method in (:var_dd, :var_dd_history)
    depth = method == :var_dd ? 0 : parameter
    return nonlinear_source_vardd(
      problem.energy,
      problem.dofs;
      u0=problem.initial,
      maxiter,
      tolerance,
      history_depth=depth,
    )
  end
  throw(ArgumentError("unsupported sensitivity method $method"))
end

function run_study11_sensitivity()
  output = "study11_semilinear_sensitivity.csv"
  if !needs_run(output)
    println("study11 sensitivity: cached, skipping")
    return
  end
  println("study11 sensitivity: mesh, Newton inner work, and history")

  tolerance = SMALL ? 1e-6 : 1e-7
  maxiter = SMALL ? 8 : 80
  rows = (
    experiment=String[],
    method=String[],
    N=Int[],
    m=Int[],
    overlap=Int[],
    parameter=Int[],
    outer_iterations=Int[],
    final_relative_residual=Float64[],
    nonlinear_local_batches=Int[],
    linear_as_batches=Int[],
    global_jacobian_products=Int[],
    converged=Int[],
  )

  function record(experiment, method, N, m, overlap, parameter, result)
    relative_residual = result.residual_history[end] / result.residual_history[1]
    push!(rows.experiment, experiment)
    push!(rows.method, string(method))
    push!(rows.N, N)
    push!(rows.m, m)
    push!(rows.overlap, overlap)
    push!(rows.parameter, parameter)
    push!(rows.outer_iterations, length(result.energy_history) - 1)
    push!(rows.final_relative_residual, relative_residual)
    push!(rows.nonlinear_local_batches, result.nonlinear_local_batches)
    push!(rows.linear_as_batches, result.linear_as_batches)
    push!(rows.global_jacobian_products, result.global_jacobian_products)
    push!(rows.converged, relative_residual <= tolerance)
    @printf(
      "  %-12s %-21s N=%-3d parameter=%-2d: %2d outer, relres %.2e\n",
      experiment,
      string(method),
      N,
      parameter,
      rows.outer_iterations[end],
      relative_residual,
    )
  end

  mesh_sizes = SMALL ? [8, 12] : [16, 32, 64]
  overlaps = SMALL ? [1, 2] : [1, 2, 4]
  mesh_methods = (
    :nonlinear_ras,
    :anderson_ras,
    :newton_pcg_as,
    :energy_imex_pcg_as,
    :aspin,
    :raspen,
    :var_dd,
    :var_dd_history,
  )
  for (N, overlap) in zip(mesh_sizes, overlaps)
    problem = semilinear_sensitivity_setup(N, 4, overlap)
    for method in mesh_methods
      parameter = method == :anderson_ras ? 4 :
                  method == :newton_pcg_as ? 4 :
                  method == :var_dd_history ? 1 : 0
      result = run_semilinear_sensitivity_method(
        method,
        problem;
        maxiter,
        tolerance,
        parameter,
      )
      record("mesh", method, N, 4, overlap, parameter, result)
    end
  end

  sensitivity_N = SMALL ? 8 : 32
  problem = semilinear_sensitivity_setup(sensitivity_N, 4, 2)
  for inner_iterations in (1, 2, 4, 8, 0)
    result = run_semilinear_sensitivity_method(
      :newton_pcg_as,
      problem;
      maxiter,
      tolerance,
      parameter=inner_iterations,
    )
    record(
      "newton_inner",
      :newton_pcg_as,
      sensitivity_N,
      4,
      2,
      inner_iterations,
      result,
    )
  end

  for method in (:anderson_ras, :var_dd_history), depth in (0, 1, 2, 4, 8)
    result = run_semilinear_sensitivity_method(
      method,
      problem;
      maxiter,
      tolerance,
      parameter=depth,
    )
    record("history", method, sensitivity_N, 4, 2, depth, result)
  end

  savetable(output, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study11_sensitivity()
end
