# Study 11: strictly convex semilinear Poisson problem
#
#   -Delta u = exp(-u) + f,  u = 0 on the boundary,
#
# with a manufactured sine solution. All four methods use the same triangular
# P1 mesh, METIS partition, overlap, and parallel local energy minimizers.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :NONLINEAR_SOURCE_METHODS) ||
  include("nonlinear_source_common.jl")

const SEMILINEAR_MODES = (
  (1.50, 1, 1),
  (0.55, 2, 3),
  (0.35, 3, 2),
)

semilinear_exact(x) = sum(
  coefficient * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in SEMILINEAR_MODES
)
semilinear_minus_laplacian(x) = pi^2 * sum(
  coefficient * (kx^2 + ky^2) * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in SEMILINEAR_MODES
)
semilinear_forcing(x) =
  semilinear_minus_laplacian(x) - exp(-semilinear_exact(x))

function run_study11()
  files = (
    "study11_semilinear_conv.csv",
    "study11_semilinear_summary.csv",
    "study11_semilinear_solution.csv",
    "study11_semilinear_partitions.csv",
  )
  if !needs_run(files...)
    println("study11: cached, skipping")
    return
  end
  println("study11: manufactured exponential semilinear Poisson problem")

  N = SMALL ? 8 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  maxiter = SMALL ? 8 : 30
  tolerance = SMALL ? 1e-6 : 1e-7

  convergence = (
    m=Int[],
    method=String[],
    outer=Int[],
    energy_gap=Float64[],
    relative_residual=Float64[],
  )
  summary = (
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    final_relative_residual=Float64[],
    relative_discrete_error=Float64[],
    relative_nodal_error=Float64[],
  )
  solutions = (N=Int[], idx=Int[], value=Float64[])

  energy_assembler,
  gradient_assembler,
  hessian_assembler,
  _,
  U,
  ndofs,
  _,
  initial = FEMDiscretizations.FEM_SemilinearPoisson(
    N,
    first(ms);
    potential=s -> exp(-s),
    potential_gradient=s -> -exp(-s),
    potential_hessian=s -> exp(-s),
    forcing=semilinear_forcing,
    overlap=overlap,
    quadrature_degree=8,
    initial_guess=x -> 0.0,
  )
  reference_energy = Energies.NonlinearEnergy(
    "exponential semilinear Poisson",
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    ndofs,
  )
  reference = Solvers.nonlinear_local_minimize(
    reference_energy,
    initial,
    collect(1:ndofs);
    relative_tolerance=1e-11,
    absolute_tolerance=1e-12,
    maxiter=100,
  ).u
  reference_value = Energies.energy(reference_energy, reference)
  reference_residual = norm(Energies.gradient(reference_energy, reference))
  reference_residual <= 1e-9 || @warn(
    "semilinear reference residual is $reference_residual"
  )

  exact_fe = interpolate_everywhere(semilinear_exact, U)
  exact_values = collect(get_free_dof_values(exact_fe))
  for (index, value) in enumerate(exact_values)
    push!(solutions.N, N)
    push!(solutions.idx, index)
    push!(solutions.value, value)
  end

  for m in ms
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    dofspar,
    _,
    _,
    _,
    initial,
    core_dofspar = FEMDiscretizations.FEM_SemilinearPoisson(
      N,
      m;
      potential=s -> exp(-s),
      potential_gradient=s -> -exp(-s),
      potential_hessian=s -> exp(-s),
      forcing=semilinear_forcing,
      overlap=overlap,
      quadrature_degree=8,
      initial_guess=x -> 0.0,
    )
    energy = Energies.NonlinearEnergy(
      "exponential semilinear Poisson",
      energy_assembler,
      gradient_assembler,
      hessian_assembler,
      ndofs,
    )

    for method in NONLINEAR_SOURCE_METHODS
      result = if method in (:nonlinear_as, :nonlinear_ras)
        nonlinear_source_schwarz_baseline(
          energy,
          dofspar;
          method=method,
          core_subdomains=core_dofspar,
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
        )
      else
        nonlinear_source_vardd(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=(method == :var_dd_history ? 1 : 0),
        )
      end

      initial_residual = result.residual_history[1]
      for outer in eachindex(result.energy_history)
        push!(convergence.m, m)
        push!(convergence.method, string(method))
        push!(convergence.outer, outer - 1)
        push!(
          convergence.energy_gap,
          max(result.energy_history[outer] - reference_value, 0.0),
        )
        push!(
          convergence.relative_residual,
          result.residual_history[outer] / initial_residual,
        )
      end

      push!(summary.m, m)
      push!(summary.method, string(method))
      push!(summary.outer_iterations, length(result.energy_history) - 1)
      push!(
        summary.final_relative_residual,
        result.residual_history[end] / initial_residual,
      )
      push!(
        summary.relative_discrete_error,
        norm(result.u - reference) / norm(reference),
      )
      push!(
        summary.relative_nodal_error,
        norm(result.u - exact_values) / norm(exact_values),
      )
      @printf(
        "  m = %d, %-21s: %2d outer, relres %.2e, nodal error %.2e\n",
        m,
        string(method),
        summary.outer_iterations[end],
        summary.final_relative_residual[end],
        summary.relative_nodal_error[end],
      )
    end
  end

  savetable("study11_semilinear_conv.csv", convergence)
  savetable("study11_semilinear_summary.csv", summary)
  savetable("study11_semilinear_solution.csv", solutions)
  savetable(
    "study11_semilinear_partitions.csv",
    triangle_partition_rows(N, ms, overlap),
  )
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study11()
end
