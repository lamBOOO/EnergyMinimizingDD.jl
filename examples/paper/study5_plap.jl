# Study 5: homogeneous p-Laplacian on a triangular P1 mesh.
#
# varDD uses the package's common inf_step/combine_step interface. The AS/RAS
# comparison implementations live in this study and reuse the same inf_step.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :NONLINEAR_SOURCE_METHODS) ||
  include("nonlinear_source_common.jl")

function run_study5()
  files = (
    "study5_conv.csv",
    "study5_summary.csv",
    "study5_solutions.csv",
    "study5_partitions.csv",
  )
  if !needs_run(files...)
    println("study5: cached, skipping")
    return
  end
  println("study5: homogeneous p-Laplacian vs one-level nonlinear Schwarz")

  N = SMALL ? 8 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  ps = [2.0, 3.0, 4.0]
  maxiter = SMALL ? 8 : 30
  tolerance = SMALL ? 1e-6 : 1e-7

  convergence = (
    p=Float64[],
    m=Int[],
    method=String[],
    outer=Int[],
    energy_gap=Float64[],
    relative_residual=Float64[],
  )
  summary = (
    p=Float64[],
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    final_relative_residual=Float64[],
    relative_solution_error=Float64[],
  )
  solutions = (p=Float64[], N=Int[], idx=Int[], value=Float64[])

  for p in ps
    reference_energy_assembler,
    reference_gradient_assembler,
    reference_hessian_assembler,
    _,
    _,
    ndofs,
    _,
    reference_initial = FEMDiscretizations.FEM_PLaplacian(
      N, first(ms), p, x -> 1.0, overlap; epsilon=1e-8
    )
    reference_problem = Energies.NonlinearEnergy(
      "p-Laplacian p=$p",
      reference_energy_assembler,
      reference_gradient_assembler,
      reference_hessian_assembler,
      ndofs,
    )

    reference = Solvers.nonlinear_local_minimize(
      reference_problem,
      reference_initial,
      collect(1:ndofs);
      relative_tolerance=1e-11,
      absolute_tolerance=1e-12,
      maxiter=100,
    ).u
    reference_energy = Energies.energy(reference_problem, reference)
    reference_residual = norm(Energies.gradient(reference_problem, reference))
    reference_residual <= 1e-8 || @warn(
      "p=$p reference residual is $(reference_residual)"
    )

    for (index, value) in enumerate(reference)
      push!(solutions.p, p)
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
      core_dofspar = FEMDiscretizations.FEM_PLaplacian(
        N, m, p, x -> 1.0, overlap; epsilon=1e-8
      )
      energy = Energies.NonlinearEnergy(
        "p-Laplacian p=$p",
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
          push!(convergence.p, p)
          push!(convergence.m, m)
          push!(convergence.method, string(method))
          push!(convergence.outer, outer - 1)
          push!(
            convergence.energy_gap,
            max(result.energy_history[outer] - reference_energy, 0.0),
          )
          push!(
            convergence.relative_residual,
            result.residual_history[outer] / initial_residual,
          )
        end

        push!(summary.p, p)
        push!(summary.m, m)
        push!(summary.method, string(method))
        push!(summary.outer_iterations, length(result.energy_history) - 1)
        push!(
          summary.final_relative_residual,
          result.residual_history[end] / initial_residual,
        )
        push!(
          summary.relative_solution_error,
          norm(result.u - reference) / norm(reference),
        )
        @printf(
          "  p = %.0f, m = %d, %-21s: %2d outer, relres %.2e\n",
          p,
          m,
          string(method),
          summary.outer_iterations[end],
          summary.final_relative_residual[end],
        )
      end
    end
  end

  savetable("study5_conv.csv", convergence)
  savetable("study5_summary.csv", summary)
  savetable("study5_solutions.csv", solutions)
  savetable("study5_partitions.csv", triangle_partition_rows(N, ms, overlap))
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study5()
end
