#!/usr/bin/env julia
# Example: How to use the nonoverlapping domain decomposition feature

using DDEigen

# Example 1: Default overlapping behavior (nodes at interface belong to both subdomains)
println("Example 1: DEFAULT (Overlapping) Mode")
println("="^60)

# With overlap=2 (default) and nonoverlapping=false (default)
# K, M, dofspar, VV = Setup_FEM(100, 9)
# Interface nodes will appear in multiple subdomains

# Example 2: Nonoverlapping behavior (nodes at interface belong to only one subdomain)
println("\nExample 2: NONOVERLAPPING Mode")
println("="^60)

# With overlap=0 and nonoverlapping=true
# K, M, dofspar, VV = Setup_FEM(100, 9; overlap=0, nonoverlapping=true)
# Interface nodes will appear in only one subdomain

# The nonoverlapping feature is useful when you want:
# - ids = [1,2,3,4]
# - domains = [[1,2],[3,4]]  (without overlap)
#
# Instead of the default overlapping:
# - domains = [[1,2,3],[2,3,4]]  (with overlap)

# Using ddm_eigen_solver with nonoverlapping:
# u, λ, history, solutions = ddm_eigen_solver(
#     N=100,
#     m=9,
#     maxiter=200,
#     tol=1e-10,
#     overlap=0,
#     nonoverlapping=true
# )

println("\nUsage:")
println("  Setup_FEM(N, m; overlap=0, nonoverlapping=true)")
println("  ddm_eigen_solver(N=100, m=9, overlap=0, nonoverlapping=true)")
println("\nParameters:")
println("  overlap:        Number of element layers to add as overlap (default: 2)")
println("  nonoverlapping: If true, removes duplicate DOFs at subdomain interfaces (default: false)")
