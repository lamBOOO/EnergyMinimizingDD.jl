using VariationalDD.Energies
using VariationalDD.FEMDiscretizations
using VariationalDD.Solvers
using Test

function test_iteration_count(result, expected_iterations)
  _, _, energy_history, solution_history, residual_history, local_update_history = result
  actual_iterations = length(residual_history)
  allowed_iterations =
    expected_iterations isa Integer ? (expected_iterations,) : expected_iterations

  # Residuals and local updates are recorded once per completed iteration.
  # Energy and solution histories additionally contain the initial state.
  @test actual_iterations in allowed_iterations
  @test length(local_update_history) == actual_iterations
  @test length(energy_history) == actual_iterations + 1
  @test length(solution_history) == actual_iterations + 1
end

N = 20
m = 9
maxiter = 200
tol = 1e-5
overlap = 2

println("\n=== Schrödinger EVP FEM Example (N=20,30) ===")
K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap = overlap)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
result = Solvers.var_dd(
  energy_eigen_fem,
  part,
  maxiter = maxiter,
  tol = tol,
  save_local_updates = true,
)
test_iteration_count(result, 24:25)
println("✓ Iteration test for N=20 passed")

N = 30
K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap = overlap)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
result = Solvers.var_dd(
  energy_eigen_fem,
  part,
  maxiter = maxiter,
  tol = tol,
  save_local_updates = true,
)
test_iteration_count(result, 35:36)
println("✓ Iteration test for N=30 passed")
