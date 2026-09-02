# Two-level EMDD and REMDD for a scalar Poisson problem.
# Run with: julia --project=. examples/nicolaides_poisson.jl

using LinearAlgebra
using EnergyMinimizingDD.Energies
using EnergyMinimizingDD.FEMDiscretizations
using EnergyMinimizingDD.Solvers

function main()
  N = 20
  m = 4
  overlap = 2
  K, _, b, subdomains, _, cores = FEMDiscretizations.FEM_Schroedinger(
    N,
    m;
    P=x -> 0.0,
    f=x -> 1.0,
    overlap,
    partitioning=:cartesian,
    return_core_partition=true,
  )
  energy = Energies.QuadraticEnergy(K, b)
  multiplicity_basis = Solvers.partition_of_unity_weights(subdomains, length(b))
  harmonic_basis = Solvers.nicolaides_coarse_basis(K, cores, subdomains)
  coarse_cases = (
    ("plain", nothing),
    ("multiplicity PoU", multiplicity_basis),
    ("harmonic Nicolaides", harmonic_basis),
  )
  tolerance = 1e-10 * norm(b - K * ones(length(b)))

  for (name, restriction) in (("EMDD", :none), ("REMDD", :partition_of_unity)),
    q in (1, 2),
    (coarse_name, coarse_basis) in coarse_cases

    _, _, _, _, residuals = Solvers.var_dd(
      energy,
      subdomains;
      maxiter=100,
      tol=tolerance,
      history_depth=q - 1,
      restriction,
      coarse_basis,
      verbose=false,
    )
    println(
      "$name q=$q, $coarse_name: $(length(residuals)) batches, " *
      "residual $(last(residuals))",
    )
  end
end

main()
