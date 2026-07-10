# Study 9: EVP -- comparison against a one-level additive-Schwarz-preconditioned
# steepest-descent and LOBPCG baselines on the SAME overlapping partitions.
#   - var_dd (energy-optimal recombination of the m local solves)
#   - LOPSD + additive Schwarz preconditioner, memoryless
#   - LOBPCG + additive Schwarz preconditioner
# No shift-invert ARPACK curve is included.
# Cost unit: subdomain solves (one var_dd sweep / AS preconditioner application = m).

isdefined(Main, :PAPER_COMMON) || include("common.jl")

using IterativeSolvers

rayleigh(K, M, x) = dot(x, K * x) / dot(x, M * x)
evp_resnorm(K, M, x, lambda) = norm(K * x - lambda .* (M * x))

function normalize_M!(x, M)
  x ./= sqrt(dot(x, M * x))
  return x
end

function lopsd_as_history(K, M, S; maxiter, tol)
  m = nsub(S)
  P = ASPreconditioner(S)
  x = normalize_M!(ones(size(K, 1)), M)
  lambda = rayleigh(K, M, x)
  hist = Tuple{Int,Float64,Float64}[(0, lambda, evp_resnorm(K, M, x, lambda))]

  for k = 1:maxiter
    r = K * x .- lambda .* (M * x)
    z = similar(r)
    ldiv!(z, P, r)
    # Memoryless locally optimal preconditioned steepest descent:
    # Rayleigh-Ritz in span{x, T r}, with no previous search direction.
    x_new = Solvers.combine_step(
      Energies.GeneralizedRayleighQuotient(K, M),
      hcat(x, z),
    )
    lambda = rayleigh(K, M, x_new)
    rn = evp_resnorm(K, M, x_new, lambda)
    push!(hist, (k * m, lambda, rn))
    rn < tol && return hist
    x = x_new
  end
  return hist
end

function lobpcg_as_history(K, M, S; maxiter, tol)
  m = nsub(S)
  x0 = ones(size(K, 1), 1)
  hist = Tuple{Int,Float64,Float64}[(0, rayleigh(K, M, view(x0, :, 1)), NaN)]
  result = lobpcg(
    K,
    M,
    false,
    x0;
    P = ASPreconditioner(S),
    maxiter = maxiter,
    tol = tol,
    log = true,
  )
  for state in result.trace
    push!(
      hist,
      (
        state.iteration * m,
        state.ritz_values[1],
        state.residual_norms[1],
      ),
    )
  end
  return hist
end

function run_study9()
  if !needs_run("study9_evp_cmp.csv")
    println("study9: cached, skipping")
    return
  end
  println("study9: EVP vs LOPSD+AS/LOBPCG+AS baselines (no ARPACK curve)")
  Random.seed!(1)

  N = SMALL ? 20 : 40
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  tol = 1e-8
  maxiter = SMALL ? 25 : 100

  rows = (
    method = String[],
    m = Int[],
    solves = Int[],
    lambda = Float64[],
    err = Float64[],
    resnorm = Float64[],
  )
  record(method, m, lambda_ref, hist) = for (s, lambda, rn) in hist
    push!(rows.method, method)
    push!(rows.m, m)
    push!(rows.solves, s)
    push!(rows.lambda, lambda)
    push!(rows.err, abs(lambda - lambda_ref))
    push!(rows.resnorm, rn)
  end

  for m in ms
    K, M, b, dofspar, U = schroedinger_setup(N, m, overlap)
    S = schwarz_setup(K, dofspar)
    lambda_ref = dense_reference_lambda(K, M)

    _, _, e_hist, _, resnorm_hist = Solvers.var_dd(
      Energies.GeneralizedRayleighQuotient(K, M),
      dofspar;
      maxiter = maxiter,
      tol = tol,
      verbose = false,
    )
    var_hist = Tuple{Int,Float64,Float64}[]
    for (k, lambda) in enumerate(e_hist)
      rn = k == 1 ? NaN : resnorm_hist[k-1]
      push!(var_hist, ((k - 1) * m, lambda, rn))
    end
    record("var_dd", m, lambda_ref, var_hist)

    record(
      "lopsd_as",
      m,
      lambda_ref,
      lopsd_as_history(K, M, S; maxiter = maxiter, tol = tol),
    )
    record(
      "lobpcg_as",
      m,
      lambda_ref,
      lobpcg_as_history(K, M, S; maxiter = maxiter, tol = tol),
    )
    println("  m = $m done")
  end
  savetable("study9_evp_cmp.csv", rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study9()
end
