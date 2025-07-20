using Plots
include("../solve.jl")
using Test
@testset "Plot tests" begin
N=500
m       = 9
maxiter = 200
tol     = 1e-10


u_approx, lambda_approx, lambda_history, solutions = ddm_eigen_solver(
    N=N,
    m=m,
    maxiter=maxiter,
    tol=tol
)
#  -- Plot 1: Final approximate eigenfunction --
x = range(0, 1, length = N+2)
u_plot = vcat(0.0, u_approx, 0.0)

plt1 = plot(
   x, u_plot,
   marker    = :o,
    xlabel    = "x",
    ylabel    = "u(x)",
    title     = "DDM Approx. Eigenfunction (λ ≈ $lambda_approx)",
    label     = "Final Eigenfunction for N= $N"
)

#  -- Plot 2: Convergence of the Rayleigh quotient --
iters = 0:length(lambda_history)-1

plt2 = plot(
   iters, lambda_history,
  marker = :o,
    xlabel = "Iteration",
    ylabel = "Rayleigh Quotient",
    title  = "Convergence of Eigenvalue (m=$m, N=$N)"
)

#  -- Plot 3: Convergence in the eigenvector (M-norm) --
#  We measure the distance of each solution from the final solution
K, M = laplace_eig_matrices(N)
final_sol = solutions[end]
exact_sol = eigen(K)
exact_val = exact_sol.values[1]
exact_vec = exact_sol.vectors[:, 1]
distances = [
   M_norm_distance(solutions[i], exact_vec, M)
    for i in 1:length(solutions)
]
plt3 = plot(
    iters, distances,
   marker = :o,
   xlabel = "Iteration",
    ylabel = "||u^(k) - u^(exact)||_M",
    title  = "Convergence of the Eigenvector in M-norm",
   yaxis = :log
)

#  -- Plot 4: Overlay all iteration solutions --
plt4 = plot(title="All Iteration Solutions (m=$m, N=$N)")
for i in 1:length(solutions)
   u_iter_withBC = vcat(0.0, solutions[i], 0.0)
    plot!(x, u_iter_withBC,
     label="Iter $(i-1)",
    legend=false,
    marker=:none)
end

# Display them in sequence
savefig(plt1, "Final_fct.png")
savefig(plt2, "R_convergence.png")
savefig(plt3, "Convergence_vec.png")
savefig(plt4, "all_its.png")
end