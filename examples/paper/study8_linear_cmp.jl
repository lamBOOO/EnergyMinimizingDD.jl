# Study 8: Poisson — comparison against one-level Schwarz baselines on the
# SAME overlapping partition:
#   - var_dd (energy-optimal recombination of the m local solves)
#   - damped additive Schwarz (theta = 1/max_multiplicity; undamped AS
#     diverges as a stationary iteration)
#   - restricted additive Schwarz (RAS, Cai-Sarkis), stationary
#   - CG preconditioned with additive Schwarz
# Cost unit: subdomain solves (one sweep / preconditioner application = m).

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function as_stationary(K, b, S; theta, maxsweeps, tol)
  x = ones(size(K, 1))
  hist = Tuple{Int,Float64}[]
  for k = 1:maxsweeps
    r = b - K * x
    push!(hist, ((k - 1) * nsub(S), norm(r)))
    norm(r) < tol && return hist
    x .+= theta .* apply_AS(S, r)
  end
  push!(hist, (maxsweeps * nsub(S), norm(b - K * x)))
  return hist
end

function ras_stationary(K, b, S; maxsweeps, tol)
  x = ones(size(K, 1))
  hist = Tuple{Int,Float64}[]
  for k = 1:maxsweeps
    r = b - K * x
    push!(hist, ((k - 1) * nsub(S), norm(r)))
    norm(r) < tol && return hist
    x .+= apply_RAS(S, r)
  end
  push!(hist, (maxsweeps * nsub(S), norm(b - K * x)))
  return hist
end

function pcg_as(K, b, S; maxiter, tol)
  m = nsub(S)
  x = ones(size(K, 1))
  r = b - K * x
  hist = [(0, norm(r))]
  z = apply_AS(S, r)
  solves = m
  p = copy(z)
  rz = dot(r, z)
  for k = 1:maxiter
    Kp = K * p
    alpha = rz / dot(p, Kp)
    x .+= alpha .* p
    r .-= alpha .* Kp
    push!(hist, (solves, norm(r)))
    norm(r) < tol && return hist
    z = apply_AS(S, r)
    solves += m
    rz_new = dot(r, z)
    p .= z .+ (rz_new / rz) .* p
    rz = rz_new
  end
  return hist
end

function run_study8()
  if !needs_run("study8_linear_cmp.csv")
    println("study8: cached, skipping")
    return
  end
  println("study8: Poisson vs Schwarz baselines")
  Random.seed!(1)

  N = SMALL ? 20 : 40
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  tol = 1e-10
  maxsweeps = SMALL ? 50 : 400

  rows = (
    method = String[],
    m = Int[],
    solves = Int[],
    resnorm = Float64[],
  )
  record(method, m, hist) = for (s, rn) in hist
    push!(rows.method, method)
    push!(rows.m, m)
    push!(rows.solves, s)
    push!(rows.resnorm, rn)
  end

  for m in ms
    K, M, b, dofspar, U = laplace_setup(N, m, overlap)
    S = schwarz_setup(K, dofspar)

    # var_dd: resnorm_hist[k] = ||K u - b|| after sweep k (m solves per sweep)
    _, _, _, _, resnorm_hist = Solvers.var_dd(
      Energies.QuadraticEnergy(K, b),
      dofspar;
      maxiter = maxsweeps,
      tol = tol,
      verbose = false,
    )
    record("var_dd", m, [(k * m, rn) for (k, rn) in enumerate(resnorm_hist)])

    theta = 1 / S.max_mult
    record("as", m, as_stationary(K, b, S; theta, maxsweeps, tol))
    record("ras", m, ras_stationary(K, b, S; maxsweeps, tol))
    record("pcg_as", m, pcg_as(K, b, S; maxiter = maxsweeps, tol))
    println("  m = $m done (AS damping theta = $(round(theta; digits = 3)))")
  end
  savetable("study8_linear_cmp.csv", rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study8()
end
