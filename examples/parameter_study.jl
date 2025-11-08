using VariationalDomainDecomposition.FEMDiscretizations
using VariationalDomainDecomposition.Energies
using VariationalDomainDecomposition.Solvers
using Plots
using LinearAlgebra

# TODO: Improve and write data to CSV

N = 20
ms = collect(2:2:10)
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

plt1=plot(
  errors, yaxis=:log,
  labels=permutedims(["m=$(m)" for m in ms]),
  title="Convergence of lowest eigenvalue for different m",
  xlabel="Iteration", ylabel="Error in lowest eigenvalue",
  markershape=:auto
)

err2=[]
m=6
Ns=collect(10:10:50)
for (i, N) in enumerate(Ns)
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
  push!(err2, e_hist .- lowest_eval)
end

plt2=plot(
  err2, yaxis=:log,
  labels=permutedims(["N=$(N)" for N in Ns]),
  title="Convergence of lowest eigenvalue for different N",
  xlabel="Iteration", ylabel="Error in lowest eigenvalue",
  markershape=:auto
)

err3=[]
m=6
N=35
olaps=collect(2:2:10)
for (i, o) in enumerate(olaps)
  K, M, b, part, U = FEMDiscretizations.FEM_Schroedinger(N, m, overlap=o)
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
  push!(err3, e_hist .- lowest_eval)
end

plt3=plot(
  err3, yaxis=:log,
  labels=permutedims(["Overlap=$(o)" for o in olaps]),
  title="Convergence of lowest eigenvalue for different overlaps",
  xlabel="Iteration", ylabel="Error in lowest eigenvalue",
  markershape=:auto
)
display(plt1)
display(plt2)
display(plt3)