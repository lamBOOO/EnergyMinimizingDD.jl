using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.Solvers
using Plots
using LinearAlgebra

# TODO: Improve and write data to CSV

N = 20
ms = collect(2:2:10)
errs = zeros(length(Ns))
errors = []
for (i, m) in enumerate(ms)
  K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap=1)
  energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
  # energy_eigen_fem = Energies.RayleighQuotient(K)
  result = Solvers.var_dd(
    energy_eigen_fem,
    part,
    maxiter=100,
    tol=1E-4,
    save_local_updates=true
  )
  e_hist = result[3]
  # true lowest eigenvalue
  lowest_eval = minimum(eigen(Matrix(K), Matrix(M)).values)
  push!(errors, e_hist .- lowest_eval)
end

plot(
  errors, yaxis=:log,
  labels=permutedims(["m=$(m)" for m in ms]),
  title="Convergence of lowest eigenvalue for different m",
  xlabel="Iteration", ylabel="Error in lowest eigenvalue",
  markershape=:auto
)
