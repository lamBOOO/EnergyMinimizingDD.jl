# Study 2: eigenvalue problem.
# (a) convergence histories: sweep over number of subdomains m and overlap
# (b) comparison against an inverse-iteration baseline (Schroedinger potential)
# (c) FEM accuracy: converged eigenvalue error vs exact 2*pi^2 over mesh sizes

isdefined(Main, :PAPER_COMMON) || include("common.jl")

# One var_dd run on the Laplace/Schroedinger EVP; returns eigenvalue history.
function evp_history(K, M, dofspar; maxiter, tol = 1e-12)
  e = Energies.GeneralizedRayleighQuotient(K, M)
  _, _, e_hist, _, resnorm_hist =
    Solvers.var_dd(e, dofspar; maxiter = maxiter, tol = tol, verbose = false)
  return e_hist, length(resnorm_hist)
end

function run_study2()
  files = (
    "study2_sweep_m.csv",
    "study2_sweep_olap.csv",
    "study2_baseline.csv",
    "study2_haccuracy.csv",
  )
  if !needs_run(files...)
    println("study2: cached, skipping")
    return
  end
  println("study2: eigenvalue problem")
  Random.seed!(1)

  N = SMALL ? 20 : 40
  maxiter = SMALL ? 20 : 60

  # (a1) sweep number of subdomains m (overlap = 2)
  ms = SMALL ? [2, 4] : [2, 4, 6, 8, 10]
  rows_m = (param = Int[], iter = Int[], err = Float64[])
  lambda_ref = NaN
  for m in ms
    K, M, b, dofspar, U = laplace_setup(N, m, 2)
    isnan(lambda_ref) && (lambda_ref = reference_lambda(K, M))
    e_hist, _ = evp_history(K, M, dofspar; maxiter = maxiter)
    for (k, ev) in enumerate(e_hist)
      push!(rows_m.param, m)
      push!(rows_m.iter, k - 1)
      push!(rows_m.err, ev - lambda_ref)
    end
    println("  sweep m = $m done ($(length(e_hist) - 1) iters)")
  end
  savetable("study2_sweep_m.csv", rows_m)

  # (a2) sweep overlap (m = 6)
  olaps = SMALL ? [1, 2] : [1, 2, 4, 8]
  m_fix = SMALL ? 2 : 6
  rows_o = (param = Int[], iter = Int[], err = Float64[])
  for olap in olaps
    K, M, b, dofspar, U = laplace_setup(N, m_fix, olap)
    e_hist, _ = evp_history(K, M, dofspar; maxiter = maxiter)
    for (k, ev) in enumerate(e_hist)
      push!(rows_o.param, olap)
      push!(rows_o.iter, k - 1)
      push!(rows_o.err, ev - lambda_ref)
    end
    println("  sweep overlap = $olap done ($(length(e_hist) - 1) iters)")
  end
  savetable("study2_sweep_olap.csv", rows_o)

  # (b) baseline: var_dd vs inverse iteration, Schroedinger potential
  K, M, b, dofspar, U = schroedinger_setup(N, m_fix, 2)
  lambda_s = reference_lambda(K, M)
  e_hist, _ = evp_history(K, M, dofspar; maxiter = maxiter)

  Kf = factorize(K)
  x = ones(size(K, 1))
  ii_hist = [dot(x, K * x) / dot(x, M * x)]
  for _ = 1:maxiter
    x = Kf \ (M * x)
    x ./= sqrt(dot(x, M * x))
    push!(ii_hist, dot(x, K * x) / dot(x, M * x))
  end

  rows_b = (method = String[], iter = Int[], err = Float64[])
  for (k, ev) in enumerate(e_hist)
    push!(rows_b.method, "var_dd")
    push!(rows_b.iter, k - 1)
    push!(rows_b.err, ev - lambda_s)
  end
  for (k, ev) in enumerate(ii_hist)
    push!(rows_b.method, "inverse_iteration")
    push!(rows_b.iter, k - 1)
    push!(rows_b.err, ev - lambda_s)
  end
  savetable("study2_baseline.csv", rows_b)
  println("  baseline comparison done")

  # (c) FEM accuracy: converged eigenvalue vs exact 2*pi^2 (Laplace)
  Ns = SMALL ? [10, 20] : [10, 20, 40, 80]
  lambda_exact = 2 * pi^2
  rows_h = (N = Int[], h = Float64[], err = Float64[], iters = Int[])
  for Nh in Ns
    K, M, b, dofspar, U = laplace_setup(Nh, 4, 2)
    e_hist, iters = evp_history(K, M, dofspar; maxiter = SMALL ? 50 : 150)
    push!(rows_h.N, Nh)
    push!(rows_h.h, 1 / Nh)
    push!(rows_h.err, e_hist[end] - lambda_exact)
    push!(rows_h.iters, iters)
    println("  h-accuracy N = $Nh done ($(iters) iters)")
  end
  savetable("study2_haccuracy.csv", rows_h)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study2()
end
