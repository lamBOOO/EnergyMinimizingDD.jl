using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Solvers

N = 45
m = 9
maxiter = 200
tol = 1e-5
lambda_exact = 2 * pi^2
f1(x) = 0

println("\n=== Laplace EVP FEM analytical convergence Example (N=45) ===")
K, M, b, part, U =
  FEMDiscretizations.FEM_Schroedinger(N, m, P = f1, overlap = 2)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
res= Solvers.var_dd(
  energy_eigen_fem,
  part,
  maxiter = 100,
  tol = 1E-4,
  save_local_updates = false,
)
println(typeof(res))
abs_err = abs(lambda_exact -info.e )
@assert abs_err < 0.01
println("✓ Laplace convergence test passed")
println("Analytical EV=$lambda_exact")
println("Computed EV=$lambda_approx")
println("approximation error=$abs_err")
