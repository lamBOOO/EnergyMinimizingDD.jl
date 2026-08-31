using LinearAlgebra
using Printf
using Statistics
using VariationalDD.Energies
using VariationalDD.FEMDiscretizations
using VariationalDD.Solvers

"Run one Poisson solve and return iteration count, residual, error, and time."
function run_method(energy, subdomains, reference, tolerance, restriction)
  result = nothing
  elapsed = @elapsed result = Solvers.var_dd(
    energy,
    subdomains;
    maxiter=200,
    tol=tolerance,
    restriction=restriction,
    verbose=false,
  )
  solution, _, _, _, residuals = result
  return (
    iterations=length(residuals),
    residual=last(residuals),
    relative_error=norm(solution - reference) / norm(reference),
    elapsed=elapsed,
  )
end

function compare_restricted_overlap(; N=24, m=4, overlaps=(1, 2, 4), samples=3)
  println("Poisson comparison: N=$N, m=$m, $samples timed samples")
  println("times exclude finite-element setup and include all varDD work")
  @printf(
    "%-8s  %-11s  %5s  %12s  %12s  %10s\n",
    "overlap",
    "method",
    "iters",
    "residual",
    "relative err",
    "median (s)"
  )

  for overlap in overlaps
    K, _, b, subdomains, _ = FEMDiscretizations.FEM_Schroedinger(
      N, m; P=x -> 0.0, f=x -> 1.0, overlap=overlap, partitioning=:cartesian
    )
    energy = Energies.QuadraticEnergy(K, b)
    reference = K \ b
    tolerance = 1e-8 * norm(b - K * ones(length(b)))

    # Compile both paths before collecting timings.
    run_method(energy, subdomains, reference, tolerance, :none)
    run_method(energy, subdomains, reference, tolerance, :partition_of_unity)

    for (name, restriction) in (("EMDD", :none), ("REMDD", :partition_of_unity))
      runs = [
        run_method(energy, subdomains, reference, tolerance, restriction) for
        _ in 1:samples
      ]
      representative = runs[1]
      @printf(
        "%-8d  %-11s  %5d  %12.4e  %12.4e  %10.4f\n",
        overlap,
        name,
        representative.iterations,
        representative.residual,
        representative.relative_error,
        median(getproperty.(runs, :elapsed)),
      )
    end
  end
end

compare_restricted_overlap()
