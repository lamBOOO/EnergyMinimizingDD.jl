
using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.Solvers
using Plots
using LinearAlgebra
using Tables
using CSV

overlap1 = zeros(4, 3)
overlap2 = zeros(4, 3)
overlap4 = zeros(4, 3)
f(x) = 0
#collect data for matrix study
for k = 1:4
  N = 10 * (2^k)

  for i = 1:3
    m = 2^i
    println("Doing olap 1 for N=$N and m=$m")
    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(
      N,
      m,
      P = f,
      overlap = 1,
      maxiter = 100,
    )
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

for k = 1:4
  N = 10 * (2^k)

  for i = 1:3
    m = 2^i
    println("Doing olap 2 for N=$N and m=$m")

    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(
      N,
      m,
      P = f,
      overlap = 2,
      maxiter = 100,
    )
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

for k = 1:4
  N = 10 * (2^k)

  for i = 1:3
    m = 2^i
    println("Doing olap 4 for N=$N and m=$m")

    K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(
      N,
      m,
      P = f,
      overlap = 4,
      maxiter = 100,
    )
    energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
    result = Solvers.var_dd(
      energy_eigen_fem,
      part,
      maxiter = 100,
      tol = 1E-4,
      save_local_updates = true,
    )
    e_hist = result[3]
    overlap4[k, i] = size(e_hist, 1)

  end
end
CSV.write("Overlap1.csv", Tables.table(overlap1), writeheader = false)
CSV.write("Overlap2.csv", Tables.table(overlap2), writeheader = false)
CSV.write("Overlap4.csv", Tables.table(overlap4), writeheader = false)
