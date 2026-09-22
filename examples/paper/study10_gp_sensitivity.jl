# Study 10 sensitivity: how much does the varDD iterate history help on the
# Gross--Pitaevskii nonlinear eigenproblem? The sweep mirrors the history
# experiments of study8 (linear source), study9 (EVP), and study11
# (semilinear): the same problem is solved with history depths q = 0, 1, ...
# and we record the iterations / local-solve batches needed to reach the
# tolerance.
#   - gp_additive: independent local nonlinear Rayleigh-quotient minimizations
#   - gp_projected_qemdd: projected Riemannian-Newton local models
# One outer sweep costs m local solves that run in parallel, so the critical
# path is one local solve per sweep for every depth; a smaller iteration count
# at larger q is therefore a genuine speedup.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :HJ_GP_KAPPA) || include("study10_gp.jl")

const GP_SENSITIVITY_METHODS = ("gp_additive", "gp_projected_qemdd")

function run_study10_sensitivity()
  file = "study10_gp_sensitivity.csv"
  if !needs_run(file)
    println("study10 sensitivity: cached, skipping")
    return nothing
  end
  println("study10 sensitivity: Gross--Pitaevskii varDD history depth")

  N = SMALL ? 16 : 32
  m = SMALL ? 2 : 4
  overlap = 2
  beta = HJ_GP_KAPPA
  depths = SMALL ? [0, 1, 2] : [0, 1, 2, 3, 4]
  tol = 1e-6
  maxiter = SMALL ? 10 : 30

  rows = (
    experiment = String[],
    method = String[],
    N = Int[],
    m = Int[],
    overlap = Int[],
    beta = Float64[],
    history_depth = Int[],
    q = Int[],
    iteration = Int[],
    solves = Int[],
    energy = Float64[],
    resnorm = Float64[],
    mass_error = Float64[],
    lambda = Float64[],
    outer_iterations = Int[],
    converged = Int[],
  )

  K, M, quartic, cubic, density_matrix, dofspar, U =
    hj_gp_discretization(N, m; overlap = overlap)
  e = Energies.GrossPitaevskiiRayleighQuotient(
    K,
    M,
    beta,
    quartic,
    cubic;
    density_matrix,
  )
  u0 = hj_gp_initial_vector(U, M)

  for method in GP_SENSITIVITY_METHODS
    for depth in depths
      _, history = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        history_depth = depth,
        projected_gp_model = method == "gp_projected_qemdd",
      )
      outer_iterations = length(history) - 1
      converged = last(history)[3] <= tol
      for (iteration, entry) in enumerate(history)
        solves, energy, residual, mass_error, lambda = entry
        push!(rows.experiment, "history")
        push!(rows.method, method)
        push!(rows.N, N)
        push!(rows.m, m)
        push!(rows.overlap, overlap)
        push!(rows.beta, beta)
        push!(rows.history_depth, depth)
        # The paper counts the local subspace dimension q = depth + 1.
        push!(rows.q, depth + 1)
        push!(rows.iteration, iteration - 1)
        push!(rows.solves, solves)
        push!(rows.energy, energy)
        push!(rows.resnorm, residual)
        push!(rows.mass_error, mass_error)
        push!(rows.lambda, lambda)
        push!(rows.outer_iterations, outer_iterations)
        push!(rows.converged, Int(converged))
      end
      @printf(
        "  %s: q = %d, outer iterations = %d, residual = %.2e%s\n",
        method,
        depth + 1,
        outer_iterations,
        last(history)[3],
        converged ? "" : " (budget reached)",
      )
    end
  end

  savetable(file, rows)
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study10_sensitivity()
end
