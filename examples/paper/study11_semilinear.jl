# Study 11: strictly convex semilinear Poisson problem
#
#   -Delta u + beta*u^3 = f,  u = 0 on the boundary,
#
# with a manufactured sine solution. All methods use the same triangular P1
# mesh, METIS partition, overlap, initial iterate, and true residual test.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :SEMILINEAR_SOURCE_METHODS) ||
  include("nonlinear_source_common.jl")

const SEMILINEAR_MODES = (
  (1.50, 1, 1),
  (0.55, 2, 3),
  (0.35, 3, 2),
)
const SEMILINEAR_BETA = 1.0
const SEMILINEAR_BETAS = (1.0, 10.0, 100.0)

semilinear_exact(x) = sum(
  coefficient * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in SEMILINEAR_MODES
)
semilinear_minus_laplacian(x) = pi^2 * sum(
  coefficient * (kx^2 + ky^2) * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in SEMILINEAR_MODES
)
semilinear_forcing(x) =
  semilinear_minus_laplacian(x) + SEMILINEAR_BETA * semilinear_exact(x)^3

function run_study11()
  files = (
    "study11_semilinear_conv.csv",
    "study11_semilinear_summary.csv",
    "study11_semilinear_solution.csv",
    "study11_semilinear_partitions.csv",
    "study11_semilinear_work.csv",
  )
  if !needs_run(files...)
    cached = loadtable(files[1])
    cached_cases = if hasproperty(cached, :beta) &&
                      hasproperty(cached, :method)
      Set(zip(cached.beta, cached.method))
    else
      Set()
    end
    expected_cases = Set(
      (beta, string(method)) for beta in SEMILINEAR_BETAS,
      method in SEMILINEAR_SOURCE_METHODS
    )
    if cached_cases == expected_cases
      println("study11: cached, skipping")
      return
    end
  end
  println("study11: manufactured cubic semilinear Poisson problem")

  N = SMALL ? 8 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  maxiter = SMALL ? 6 : 40
  tolerance = SMALL ? 1e-6 : 1e-7

  convergence = (
    beta=Float64[],
    m=Int[],
    method=String[],
    outer=Int[],
    energy_gap=Float64[],
    relative_residual=Float64[],
  )
  summary = (
    beta=Float64[],
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    final_relative_residual=Float64[],
    relative_discrete_error=Float64[],
    relative_nodal_error=Float64[],
  )
  work = (
    beta=Float64[],
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    nonlinear_local_batches=Int[],
    linear_as_batches=Int[],
    global_jacobian_products=Int[],
  )
  solutions = (N=Int[], idx=Int[], value=Float64[])

  for beta in SEMILINEAR_BETAS
    forcing(x) = semilinear_minus_laplacian(x) + beta * semilinear_exact(x)^3
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
    potential=s -> beta * s^4 / 4,
    potential_gradient=s -> beta * s^3,
    potential_hessian=s -> 3 * beta * s^2,
    forcing=forcing,
    overlap=overlap,
    quadrature_degree=8,
    initial_guess=x -> 0.0,
  )
  reference_energy = Energies.NonlinearEnergy(
    "cubic semilinear Poisson",
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
  if beta == first(SEMILINEAR_BETAS)
    for (index, value) in enumerate(exact_values)
      push!(solutions.N, N)
      push!(solutions.idx, index)
      push!(solutions.value, value)
    end
  end

  for m in ms
    @info m
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    dofspar,
    _,
    _,
    stiffness,
    initial,
    core_dofspar,
    mass = FEMDiscretizations.FEM_SemilinearPoisson(
      N,
      m;
      potential=s -> beta * s^4 / 4,
      potential_gradient=s -> beta * s^3,
      potential_hessian=s -> 3 * beta * s^2,
      forcing=forcing,
      overlap=overlap,
      quadrature_degree=8,
      initial_guess=x -> 0.0,
      return_mass_matrix=true,
    )
    @info "  ndofs = $ndofs, core_ndofs = $(length(core_dofspar))"
    energy = Energies.NonlinearEnergy(
      "cubic semilinear Poisson",
      energy_assembler,
      gradient_assembler,
      hessian_assembler,
      ndofs,
    )

    for method in SEMILINEAR_SOURCE_METHODS
      @time result = if method == :nonlinear_as
        nonlinear_source_schwarz_baseline(
          energy,
          dofspar;
          method=method,
          core_subdomains=core_dofspar,
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
        )
      elseif method == :anderson_ras
        nonlinear_source_anderson_ras(
          energy,
          dofspar,
          core_dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=4,
        )
      elseif method == :nonlinear_cg_optim_as
        nonlinear_source_optim_ncg_as(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
        )
      elseif method == :newton_pcg_as_8
        nonlinear_source_newton_pcg_as(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_iterations=8,
        )
      elseif method == :newton_pcg_as_4
        nonlinear_source_newton_pcg_as(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_iterations=4,
        )
      elseif method == :newton_pcg_as_2
        nonlinear_source_newton_pcg_as(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_iterations=2,
        )
      elseif method == :newton_pcg_as_1
        nonlinear_source_newton_pcg_as(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_iterations=1,
        )
      elseif method == :energy_imex_pcg_as
        nonlinear_source_energy_imex_pcg_as(
          energy,
          stiffness,
          mass,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          timestep=1.0,
          inner_maxiter=SMALL ? 30 : 200,
          inner_relative_tolerance=1e-10,
        )
      elseif method == :raspen
        nonlinear_source_raspen(
          energy,
          dofspar,
          core_dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_maxiter=SMALL ? 8 : 40,
          inner_relative_tolerance=1e-6,
        )
      elseif method == :aspin
        nonlinear_source_aspin(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          inner_maxiter=SMALL ? 8 : 40,
          inner_relative_tolerance=1e-6,
        )
      elseif method == :var_dd
        nonlinear_source_vardd(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=0,
        )
      elseif method == :var_dd_history
        nonlinear_source_vardd(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=1,
        )
      elseif method == :var_dd_quadratic
        nonlinear_source_vardd(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=0,
          quadratic_model=true,
        )
      elseif method == :var_dd_quadratic_history
        nonlinear_source_vardd(
          energy,
          dofspar;
          u0=initial,
          maxiter=maxiter,
          tolerance=tolerance,
          history_depth=1,
          quadratic_model=true,
        )
      end

      initial_residual = result.residual_history[1]
      for outer in eachindex(result.energy_history)
        push!(convergence.beta, beta)
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

      push!(summary.beta, beta)
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
      push!(work.beta, beta)
      push!(work.m, m)
      push!(work.method, string(method))
      push!(work.outer_iterations, length(result.energy_history) - 1)
      push!(work.nonlinear_local_batches, result.nonlinear_local_batches)
      push!(work.linear_as_batches, result.linear_as_batches)
      push!(work.global_jacobian_products, result.global_jacobian_products)
      @printf(
        "  beta = %.0f, m = %d, %-21s: %2d outer, relres %.2e, nodal error %.2e\n",
        beta,
        m,
        string(method),
        summary.outer_iterations[end],
        summary.final_relative_residual[end],
        summary.relative_nodal_error[end],
      )
    end
  end
  end

  savetable("study11_semilinear_conv.csv", convergence)
  savetable("study11_semilinear_summary.csv", summary)
  savetable("study11_semilinear_solution.csv", solutions)
  savetable("study11_semilinear_work.csv", work)
  savetable(
    "study11_semilinear_partitions.csv",
    triangle_partition_rows(N, ms, overlap),
  )
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study11()
end
