using VariationalDD.Energies
using VariationalDD.FEMDiscretizations
using VariationalDD.Solvers
using VariationalDD
using LinearAlgebra
using FiniteDiff
using Gridap
using GridapDistributed
using Metis
using IterativeSolvers
using Arpack
using Printf
using Random
using LineSearches

vtk_output_dir = normpath(joinpath(@__DIR__, "..", "output", "vtk"))
mkpath(vtk_output_dir)

N = 80
m = 2
maxiter = 100
tol = 1e-4
overlap = 4

# Schroedinger EVP FEM
println("\n=== Schrödinger EVP FEM Example ===")
K, M, b, part, U = VariationalDD.FEMDiscretizations.FEM_Schroedinger(
  N, m; overlap=overlap
)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
result = VariationalDD.Solvers.var_dd(
  energy_eigen_fem, part; maxiter=maxiter, tol=tol, save_local_updates=true
)

# Handle different return values based on save_local_updates
u_approx,
lambda_approx, lambda_history, solutions, resnorm_history,
local_updates_history = result
println("Final approximate eigenvalue = $lambda_approx")
exact_sol = eigs(K, M; nev=1, which=:SM, maxiter=1000)
println("Exact eigenvalue = $(exact_sol[1][1])")

# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  joinpath(vtk_output_dir, "eigen_solution");
  cellfields=["u_approx" => FEFunction(U, u_approx)],
)
# write all sols to vtk file for visualization
for (i, sol) in enumerate(solutions)
  writevtk(
    U.space.fe_basis.trian,
    joinpath(vtk_output_dir, "schroedinger_solution_iter$(i-1)");
    cellfields=["u" => FEFunction(U, sol)],
  )
end
for (iter, local_updates) in enumerate(local_updates_history)
  cellfields = Dict{String,FEFunction}()
  for (subdomain_idx, local_sol) in enumerate(local_updates)
    cellfields["u_subdomain_$(subdomain_idx)"] = FEFunction(U, local_sol)
  end
  writevtk(
    U.space.fe_basis.trian,
    joinpath(vtk_output_dir, "schroedinger_local_updates_iter$(iter-1)");
    cellfields=cellfields,
  )
end
@assert abs(lambda_approx - exact_sol[1][1]) < 1e-6
println("✓ passed.")

#inverse iteaertion
println("\n=== Inverse Iteration Schrödinger EVP FEM Example ===")
x = ones(size(K, 1))
ii_resnorms = Float64[]
for i in 1:100
  global x
  x = (K) \ (M*x)
  x = x / sqrt(dot(x, M*x))
  resnorm = norm(K*x - (dot(x, K*x)/dot(x, M*x))*M*x)
  push!(ii_resnorms, resnorm)
  @printf("Iteration %3d: Residual norm ≈ %12.6e\n", i, resnorm)
  if resnorm < tol
    break
  end
end
lambda_approx_inv = dot(x, K*x)/dot(x, M*x)
println("Final approximate eigenvalue (inverse iteration) = $lambda_approx_inv")

# Poisson problem FEM: -Δu = f with f(x) = 1
println("\n=== Poisson Linear FEM Example ===")
K, M, b, part, U = VariationalDD.FEMDiscretizations.FEM_Schroedinger(
  N, m; P=(x -> 0.0), f=(x -> 1.0), overlap=overlap
)
energy_poisson_fem = Energies.QuadraticEnergy(K, b, 0.0)
# direct solve
u_poisson_direct = K \ b
# Solvers.var_dd solve with local update visualization
result = VariationalDD.Solvers.var_dd(
  energy_poisson_fem, part; maxiter=maxiter, tol=1e-5, save_local_updates=true
)
u_poisson,
E_poisson, E_hist, sols, resnorm_hist_poisson,
local_updates_history = result
E_poisson = Energies.energy(energy_poisson_fem, u_poisson)
println("Poisson energy = $E_poisson")
# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  joinpath(vtk_output_dir, "poisson_solution");
  cellfields=[
    "u_poisson" => FEFunction(U, u_poisson),
    "u_poisson_direct" => FEFunction(U, u_poisson_direct),
  ],
)
for (iter, local_updates) in enumerate(local_updates_history)
  cellfields = Dict{String,FEFunction}()
  for (subdomain_idx, local_sol) in enumerate(local_updates)
    cellfields["u_subdomain_$(subdomain_idx)"] = FEFunction(U, local_sol)
  end
  writevtk(
    U.space.fe_basis.trian,
    joinpath(vtk_output_dir, "poisson_local_updates_iter$(iter-1)");
    cellfields=cellfields,
  )
end
# check difference
println(
  "norm(u_poisson - u_poisson_direct) = ", norm(u_poisson - u_poisson_direct)
)
@assert norm(u_poisson - u_poisson_direct) < 1e-4
println("✓ passed.")

# write all sols to vtk file for visualization
for (i, sol) in enumerate(sols)
  writevtk(
    U.space.fe_basis.trian,
    joinpath(vtk_output_dir, "poisson_solution_iter$(i-1)");
    cellfields=["u" => FEFunction(U, sol)],
  )
end

# Linear Regression problem example
println("\n=== Linear Regression Example ===")
# Create a synthetic linear regression problem: y = Ax + ε
n_params = 20    # number of parameters to estimate
n_obs = 100      # number of observations (overdetermined system)
Random.seed!(42) # for reproducibility

# Generate random design matrix and true parameters
A_lr = randn(n_obs, n_params)
x_true = randn(n_params)
b_lr = A_lr * x_true + 0.1 * randn(n_obs)  # add some noise

# Create linear regression energy functional
energy_lr = Energies.LinearRegressionEnergy(A_lr, b_lr)

# Create simple uniform partition for demonstration
# In practice, this would be more sophisticated domain decomposition
part_lr = [Vector{Int32}() for _ in 1:m]
for i in 1:n_params
  push!(part_lr[((i - 1) % m) + 1], i)
end

# Direct solution via normal equations
x_direct = (A_lr' * A_lr) \ (A_lr' * b_lr)

# Domain decomposition solution
x_dd, E_lr, E_hist_lr, sols_lr, resnorm_hist_lr = Solvers.var_dd(
  energy_lr, part_lr; maxiter=maxiter, tol=tol
)

# Compare solutions
println("Direct least squares energy: $(Energies.energy(energy_lr, x_direct))")
println("DD least squares energy: $E_lr")
println("Relative error in solution: $(norm(x_dd - x_direct) / norm(x_direct))")
println("Residual norm (direct): $(norm(A_lr * x_direct - b_lr))")
println("Residual norm (DD): $(norm(A_lr * x_dd - b_lr))")

# Verify that we're solving the normal equations
normal_residual_direct = (A_lr' * A_lr) * x_direct - (A_lr' * b_lr)
normal_residual_dd = (A_lr' * A_lr) * x_dd - (A_lr' * b_lr)
println("Normal equation residual (direct): $(norm(normal_residual_direct))")
println("Normal equation residual (DD): $(norm(normal_residual_dd))")

@assert norm(x_dd - x_direct) / norm(x_direct) < 1e-3
@assert abs(E_lr - Energies.energy(energy_lr, x_direct)) < 1e-6
println("✓ passed.")


# Nonlinear algebraic system example: Circle-Cubic intersection
println("\n=== Nonlinear Algebraic System Example ===")
println("Solving: F[1] = x[1]² + x[2]² - 1 = 0  (circle)")
println("         F[2] = x[1]³ - x[2] = 0        (cubic)")

# Define the system F(x) = 0 as an energy minimization: E(x) = ½||F(x)||²
function circle_cubic_system(x::Vector{Float64})
  F = zeros(2)
  F[1] = x[1]^2 + x[2]^2 - 1.0    # circle: x² + y² = 1
  F[2] = x[1]^3 - x[2]             # cubic: y = x³
  return F
end

# Energy functional: E(x) = ½||F(x)||²
function circle_cubic_energy(x::Vector{Float64})
  F = circle_cubic_system(x)
  return 0.5 * dot(F, F)
end

# Gradient: ∇E(x) = J(x)ᵀ F(x) where J is Jacobian of F
function circle_cubic_gradient(x::Vector{Float64})
  F = circle_cubic_system(x)

  # Jacobian matrix J = [∂F₁/∂x₁  ∂F₁/∂x₂]
  #                     [∂F₂/∂x₁  ∂F₂/∂x₂]
  J = zeros(2, 2)
  J[1, 1] = 2 * x[1]        # ∂F₁/∂x₁ = 2x₁
  J[1, 2] = 2 * x[2]        # ∂F₁/∂x₂ = 2x₂
  J[2, 1] = 3 * x[1]^2      # ∂F₂/∂x₁ = 3x₁²
  J[2, 2] = -1.0          # ∂F₂/∂x₂ = -1

  return J' * F  # ∇E = Jᵀ F
end

# Create the nonlinear energy for the algebraic system
energy_circle_cubic = Energies.NonlinearEnergy(
  "Circle-Cubic System",
  circle_cubic_energy,
  circle_cubic_gradient,
  2, # 2D problem
)

# Create simple partition for 2D problem (each subdomain gets one variable)
part_cc = [Vector{Int32}([1]), Vector{Int32}([2])]

# Initial guess (near one of the expected solutions)
x_init = [0.8, 0.5]  # Should converge to intersection point

try
  println("Initial guess: x = $(x_init)")
  println("Initial F(x) = $(circle_cubic_system(x_init))")
  println("Initial ||F(x)|| = $(norm(circle_cubic_system(x_init)))")
  println(
    "Initial energy E(x) = $(Energies.energy(energy_circle_cubic, x_init))"
  )

  # Solve using domain decomposition
  x_sol, E_sol, E_hist_cc, sols_cc, resnorm_hist_cc = Solvers.var_dd(
    energy_circle_cubic, part_cc; maxiter=50, tol=1e-5
  )

  println("\nSolution found: x = $(x_sol)")
  F_sol = circle_cubic_system(x_sol)
  println("Final F(x) = $(F_sol)")
  println("Final ||F(x)|| = $(norm(F_sol))")
  println("Final energy E(x) = $E_sol")

  # Verify the solution
  circle_error = abs(x_sol[1]^2 + x_sol[2]^2 - 1.0)
  cubic_error = abs(x_sol[1]^3 - x_sol[2])

  println("\nVerification:")
  println("Circle equation error: |x² + y² - 1| = $(circle_error)")
  println("Cubic equation error: |x³ - y| = $(cubic_error)")

  if norm(F_sol) < 1e-6
    println("✓ Nonlinear algebraic system solved successfully!")
    println(
      "  Solution represents intersection of unit circle and cubic curve."
    )
  else
    println("⚠ System not fully converged, residual = $(norm(F_sol))")
  end

catch e
  println("Error in algebraic system example: $e")
end

# Nonlinear algebraic synthetic system example
println("\n=== Nonlinear Synthetic Algebraic System Example ===")

# Define the system F(x) = 0 as an energy minimization: E(x) = ½||F(x)||²
function synth_system(x::Vector{Float64})
  F = zeros(5)
  F[1] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + (x[5]^2) - 5
  F[2] = exp(x[1]) + x[2] * x[3] - exp(1) - 1
  F[3] = sin(x[3]) + x[4] * x[5] - sin(1) - 1
  F[4] = x[1] * x[2] + x[3] * x[4] + (x[5]^3) - 3
  F[5] = x[1] + x[2] + x[3] + x[4] + x[5] - 5
  return F
end

# Energy functional: E(x) = ½||F(x)||²
function synthetic_sys_energy(x::Vector{Float64})
  F = synth_system(x)
  return 0.5 * dot(F, F)
end

# Gradient: ∇E(x) = J(x)ᵀ F(x) where J is Jacobian of F
function synthetic_sys_gradient(x::Vector{Float64})
  F = synth_system(x)
  J = zeros(5, 5)
  for i in 1:5
    J[1, i] = 2 * x[i]
  end
  J[2, 1] = exp(x[1])
  J[2, 2] = x[3]
  J[2, 3] = x[2]
  J[3, 3] = cos(x[3])
  J[3, 4] = x[5]
  J[3, 5] = x[4]
  J[4, 1] = x[2]
  J[4, 2] = x[1]
  J[4, 3] = x[4]
  J[4, 4] = x[3]
  J[4, 5] = 3 * x[5]^2
  for i in 1:5
    J[5, i] = 1
  end
  return J' * F  # ∇E = Jᵀ F
end
energy_synthetic_system = Energies.NonlinearEnergy(
  "Synthetic 5D System",
  synthetic_sys_energy,
  synthetic_sys_gradient,
  5, # 5D problem
)

part_synth = [Vector{Int32}([1, 2, 3]), Vector{Int32}([3, 4, 5])]

# Initial guess (near one of the expected solutions)
x_init2 = [1.2, 1.3, 1.4, 1.1, 1.2]  # Should converge to analytical solution

try
  println("Initial guess: x = $(x_init2)")
  println("Initial F(x) = $(synth_system(x_init2))")
  println("Initial ||F(x)|| = $(norm(synth_system(x_init2)))")
  println(
    "Initial energy E(x) = $(Energies.energy(energy_synthetic_system, x_init2))"
  )

  # Solve using domain decomposition
  x_sol2, E_sol2, E_hist_synth, sols_synth = Solvers.var_dd(
    energy_synthetic_system, part_synth; maxiter=10000, tol=1e-5
  )

  println("\nSolution found: x = $(x_sol2)")
  F_sol2 = synth_system(x_sol2)
  println("Final F(x) = $(F_sol2)")
  println("Final ||F(x)|| = $(norm(F_sol2))")
  println("Final energy E(x) = $E_sol2")

  # Verify the solution
  err = ones(5) - x_sol2
  synth_error = norm(err)

  println("\nVerification:")
  println("System error = $(synth_error)")

  if norm(F_sol2) < 1e-6
    println("✓ Nonlinear algebraic system solved successfully!")
    println(
      "  Solution represents intersection of unit circle and cubic curve."
    )
  else
    println("⚠ System not fully converged, residual = $(norm(F_sol2))")
  end
catch e
  println("Error in algebraic system example: $e")
end
