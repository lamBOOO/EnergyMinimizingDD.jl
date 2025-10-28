using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Solvers

N = 20
m = 9
maxiter = 200
tol = 1e-5
overlap = 2

println("\n=== Schrödinger EVP FEM Example (N=20,30) ===")
K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap=overlap)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
result = Solvers.var_dd(
  energy_eigen_fem,
  part,
  maxiter=maxiter,
  tol=tol,
  save_local_updates=true
)
u_approx, lambda_approx, lambda_history, solutions, local_updates_history = result
@assert length(lambda_history)<=26
println("✓ Iteration test for N=20 passed")

N=30
K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap=overlap)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
result = Solvers.var_dd(
  energy_eigen_fem,
  part,
  maxiter=maxiter,
  tol=tol,
  save_local_updates=true
)
u_approx, lambda_approx, lambda_history, solutions, local_updates_history = result
@assert length(lambda_history)<=37
println("✓ Iteration test for N=30 passed")
