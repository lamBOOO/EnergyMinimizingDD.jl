using VarDD

for m in [4, 8, 16, 32]
  println("\n=== Parameter study with m = $m ===")

  # Problem setup
  N = 2^7          # Number of FEM nodes
  overlap = 2      # Overlap size
  maxiter = 50     # Maximum number of DD iterations
  tol = 1e-8       # Tolerance for convergence
  # setup
  VarDD.var_dd(m, ...)
  iters |> CSV
  println("Wrote results to examples/parameter_study_m$(m).csv")
end
