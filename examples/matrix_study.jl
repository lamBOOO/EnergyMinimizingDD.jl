
using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.Solvers
using Plots
using LinearAlgebra
using Tables
using CSV

overlap1 = zeros(8, 3)
overlap2 = zeros(8, 3)
overlap3 = zeros(8, 3)
#collect data for matrix study
for k = 1:8
  N = 10 * k

  ms = collect(2:2:6)
  for (i, m) in enumerate(ms)
    println("Doing olap 1 for N=$N and m=$m")
    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap = 1)
    energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
    result = Solvers.var_dd(
      energy_eigen_fem,
      part,
      maxiter = 100,
      tol = 1E-4,
      save_local_updates = true,
    )
    e_hist = result[3]
    overlap1[k, i] = size(e_hist, 1)
  end
end

for k = 1:8
  N = 10 * k
  ms = collect(2:2:6)
  for (i, m) in enumerate(ms)
    println("Doing olap 2 for N=$N and m=$m")

    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap = 2)
    energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
    result = Solvers.var_dd(
      energy_eigen_fem,
      part,
      maxiter = 100,
      tol = 1E-4,
      save_local_updates = true,
    )
    e_hist = result[3]
    overlap2[k, i] = size(e_hist, 1)

  end
end

for k = 1:8
  N = 10 * k
  ms = collect(2:2:6)
  for (i, m) in enumerate(ms)
    println("Doing olap 3 for N=$N and m=$m")

    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap = 3)
    energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
    result = Solvers.var_dd(
      energy_eigen_fem,
      part,
      maxiter = 100,
      tol = 1E-4,
      save_local_updates = true,
    )
    e_hist = result[3]
    overlap3[k, i] = size(e_hist, 1)

  end
end
CSV.write("Overlap1.csv", Tables.table(overlap1), writeheader = false)
CSV.write("Overlap2.csv", Tables.table(overlap2), writeheader = false)
CSV.write("Overlap3.csv", Tables.table(overlap3), writeheader = false)
