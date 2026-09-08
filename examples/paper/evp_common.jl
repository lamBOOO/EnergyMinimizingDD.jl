# Shared, study-local solvers and work accounting for the linear generalized
# eigenvalue experiments. These comparison methods deliberately do not extend
# the package solver API.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

using IterativeSolvers

const EVP_COMMON = true

const EVP_COMPARISON_METHODS = (
  :var_dd,
  :var_dd_history,
  :lopsd_as,
  :lobpcg_as,
  :jd_gmres_as_1,
  :jd_gmres_as_2,
  :jd_gmres_as_4,
  :si_lanczos_pcg_as_2,
  :si_lanczos_pcg_as_4,
  :si_lanczos_pcg_as_8,
  :si_lanczos_pcg_as_16,
  :si_lanczos_pcg_as_32,
)

const EVP_JD_GMRES_ITERATIONS = Dict(
  :jd_gmres_as_1 => 1,
  :jd_gmres_as_2 => 2,
  :jd_gmres_as_4 => 4,
)

const EVP_LANCZOS_PCG_ITERATIONS = Dict(
  :si_lanczos_pcg_as_2 => 2,
  :si_lanczos_pcg_as_4 => 4,
  :si_lanczos_pcg_as_8 => 8,
  :si_lanczos_pcg_as_16 => 16,
  :si_lanczos_pcg_as_32 => 32,
)

rayleigh(K, M, x) = dot(x, K * x) / dot(x, M * x)
evp_residual(K, M, x, lambda) = K * x .- lambda .* (M * x)

function normalize_M!(x, M)
  x ./= sqrt(dot(x, M * x))
  return x
end

function evp_entry(
  iteration,
  lambda,
  residual,
  initial_residual;
  local_eigen_batches=0,
  linear_as_batches=0,
  local_iterations_critical=0,
  local_iterations_total=0,
  global_k_products=0,
  global_m_products=0,
  inner_iterations=0,
)
  return (
    iteration=Int(iteration),
    lambda=Float64(lambda),
    residual=Float64(residual),
    relative_residual=Float64(residual / initial_residual),
    local_eigen_batches=Int(local_eigen_batches),
    linear_as_batches=Int(linear_as_batches),
    local_iterations_critical=Int(local_iterations_critical),
    local_iterations_total=Int(local_iterations_total),
    global_k_products=Int(global_k_products),
    global_m_products=Int(global_m_products),
    inner_iterations=Int(inner_iterations),
  )
end

function empty_evp_result(u, history; kwargs...)
  return (
    u=u,
    history=history,
    local_stats=get(kwargs, :local_stats, NamedTuple[]),
    combination_stats=get(kwargs, :combination_stats, NamedTuple[]),
    inner_stats=get(kwargs, :inner_stats, NamedTuple[]),
  )
end

"Memoryless locally optimal preconditioned steepest descent with AS."
function evp_lopsd_as(K, M, schwarz; maxiter, relative_tolerance)
  x = normalize_M!(ones(size(K, 1)), M)
  lambda = rayleigh(K, M, x)
  initial = norm(evp_residual(K, M, x, lambda))
  history = [evp_entry(0, lambda, initial, initial; global_k_products=1, global_m_products=1)]
  k_products = 1
  m_products = 1
  for iteration = 1:maxiter
    r = evp_residual(K, M, x, lambda)
    z = apply_AS(schwarz, r)
    # combine_step applies K and M to both basis vectors, and the explicit
    # residual evaluation below applies each operator once more.
    x = Solvers.combine_step(
      Energies.GeneralizedRayleighQuotient(K, M),
      hcat(x, z),
    )
    lambda = rayleigh(K, M, x)
    residual = norm(evp_residual(K, M, x, lambda))
    k_products += 3
    m_products += 3
    push!(
      history,
      evp_entry(
        iteration,
        lambda,
        residual,
        initial;
        linear_as_batches=iteration,
        global_k_products=k_products,
        global_m_products=m_products,
      ),
    )
    residual <= relative_tolerance * initial && break
  end
  return empty_evp_result(x, history)
end

"Single-vector LOBPCG with the shared one-level AS preconditioner."
function evp_lobpcg_as(K, M, schwarz; maxiter, relative_tolerance)
  x0 = normalize_M!(ones(size(K, 1)), M)
  lambda0 = rayleigh(K, M, x0)
  initial = norm(evp_residual(K, M, x0, lambda0))
  result = lobpcg(
    K,
    M,
    false,
    reshape(x0, :, 1);
    P=ASPreconditioner(schwarz),
    maxiter=maxiter,
    tol=relative_tolerance * initial,
    log=true,
  )
  history = [evp_entry(0, lambda0, initial, initial; global_k_products=1, global_m_products=1)]
  for state in result.trace
    iteration = state.iteration
    push!(
      history,
      evp_entry(
        iteration,
        state.ritz_values[1],
        state.residual_norms[1],
        initial;
        linear_as_batches=iteration,
        global_k_products=1 + iteration,
        global_m_products=1 + iteration,
      ),
    )
  end
  return empty_evp_result(vec(result.X[:, 1]), history)
end

"PCG solve with explicit AS and operator-application counters."
function evp_pcg_as(K, rhs, schwarz; relative_tolerance, maxiter)
  x = zeros(length(rhs))
  r = copy(rhs)
  initial = norm(r)
  initial == 0 && return (
    x=x, Kx=zeros(length(rhs)), iterations=0, as_batches=0,
    k_products=0, converged=true, relative_residual=0.0,
  )
  z = apply_AS(schwarz, r)
  as_batches = 1
  p = copy(z)
  rz = dot(r, z)
  iterations = 0
  converged = false
  for iteration = 1:maxiter
    Kp = K * p
    denominator = dot(p, Kp)
    (!isfinite(denominator) || denominator <= 0) && break
    alpha = rz / denominator
    x .+= alpha .* p
    r .-= alpha .* Kp
    iterations = iteration
    if norm(r) <= relative_tolerance * initial
      converged = true
      break
    end
    iteration == maxiter && break
    z = apply_AS(schwarz, r)
    as_batches += 1
    rz_new = dot(r, z)
    (!isfinite(rz_new) || rz_new <= 0) && break
    p .= z .+ (rz_new / rz) .* p
    rz = rz_new
  end
  return (
    x=x,
    Kx=rhs - r,
    iterations=iterations,
    as_batches=as_batches,
    k_products=iterations,
    converged=converged,
    relative_residual=norm(r) / initial,
  )
end

"Inexact shift-and-invert Lanczos with full K-orthogonalization."
function evp_si_lanczos_pcg_as(
  K,
  M,
  schwarz;
  maxiter,
  relative_tolerance,
  inner_relative_tolerance=1e-10,
  inner_maxiter=500,
  restart_dimension=20,
)
  x = ones(size(K, 1))
  Kx = K * x
  Mx = M * x
  scale = sqrt(dot(x, Kx))
  x ./= scale
  Kx ./= scale
  Mx ./= scale
  V = reshape(copy(x), :, 1)
  KV = reshape(copy(Kx), :, 1)
  MV = reshape(copy(Mx), :, 1)
  lambda = dot(x, Kx) / dot(x, Mx)
  initial = norm(Kx - lambda .* Mx)
  history = [evp_entry(0, lambda, initial, initial; global_k_products=1, global_m_products=1)]
  inner_stats = NamedTuple[]
  linear_batches = 0
  inner_iterations = 0
  k_products = 1
  m_products = 1

  for outer = 1:maxiter
    if size(V, 2) >= restart_dimension
      scale = sqrt(dot(x, Kx))
      V = reshape(x ./ scale, :, 1)
      KV = reshape(Kx ./ scale, :, 1)
      MV = reshape(Mx ./ scale, :, 1)
    end

    inner = evp_pcg_as(
      K,
      view(MV, :, size(MV, 2)),
      schwarz;
      relative_tolerance=inner_relative_tolerance,
      maxiter=inner_maxiter,
    )
    linear_batches += inner.as_batches
    inner_iterations += inner.iterations
    k_products += inner.k_products
    z = copy(inner.x)
    # Form K*z explicitly before orthogonalization. For a tightly converged
    # inner solve, recovering it as rhs-r is accurate enough; after only a few
    # PCG steps, however, the resulting recurrence-level roundoff can be
    # amplified when the new direction is nearly in the current subspace.
    Kz = K * z
    k_products += 1

    # Full two-pass K-orthogonalization preserves the symmetric
    # shift-and-invert subspace despite finite-precision inner solves.
    for _ = 1:2
      coefficients = V' * Kz
      z .-= V * coefficients
      Kz .-= KV * coefficients
    end
    z_norm = sqrt(max(dot(z, Kz), 0.0))
    z_norm <= 100 * eps(Float64) && break
    z ./= z_norm
    Kz ./= z_norm
    Mz = M * z
    m_products += 1
    V = hcat(V, z)
    KV = hcat(KV, Kz)
    MV = hcat(MV, Mz)

    reduced_stiffness = Symmetric(V' * KV)
    reduced_mass = Symmetric(V' * MV)
    # Fixed, low PCG budgets may make the inverse application almost
    # collinear with the retained Ritz vector after a restart. Stop at the
    # last valid Ritz pair instead of passing a numerically dependent basis
    # to the definite generalized eigensolver.
    (!isposdef(reduced_stiffness) || !isposdef(reduced_mass)) && break
    reduced = eigen(reduced_stiffness, reduced_mass)
    coefficient = reduced.vectors[:, 1]
    x = V * coefficient
    Kx = KV * coefficient
    Mx = MV * coefficient
    scale = sqrt(dot(x, Mx))
    x ./= scale
    Kx ./= scale
    Mx ./= scale
    lambda = reduced.values[1]
    residual = norm(Kx - lambda .* Mx)
    push!(
      inner_stats,
      (
        outer_iteration=outer,
        iterations=inner.iterations,
        converged=inner.converged,
        relative_residual=inner.relative_residual,
      ),
    )
    push!(
      history,
      evp_entry(
        outer,
        lambda,
        residual,
        initial;
        linear_as_batches=linear_batches,
        global_k_products=k_products,
        global_m_products=m_products,
        inner_iterations=inner_iterations,
      ),
    )
    residual <= relative_tolerance * initial && break
  end
  return empty_evp_result(x, history; inner_stats)
end

"Flexible GMRES solution of the projected Jacobi--Davidson correction equation."
function evp_jd_correction(
  K,
  M,
  schwarz,
  u,
  Mu,
  theta,
  residual;
  relative_tolerance,
  maxiter,
)
  beta = norm(residual)
  beta == 0 && return (
    correction=zeros(length(u)), Kcorrection=zeros(length(u)),
    Mcorrection=zeros(length(u)), iterations=0, as_batches=0,
    k_products=0, m_products=0, converged=true,
    relative_residual=0.0,
  )
  dimension = min(maxiter, length(u))
  V = zeros(length(u), dimension + 1)
  Z = zeros(length(u), dimension)
  KZ = zeros(length(u), dimension)
  MZ = zeros(length(u), dimension)
  H = zeros(dimension + 1, dimension)
  V[:, 1] .= -residual ./ beta
  rhs = zeros(dimension + 1)
  rhs[1] = beta
  coefficients = zeros(0)
  used = 0
  converged = false

  for iteration = 1:dimension
    raw = apply_AS(schwarz, view(V, :, iteration))
    Mraw = M * raw
    projection = dot(u, Mraw)
    z = raw .- projection .* u
    Mz = Mraw .- projection .* Mu
    Kz = K * z
    w = Kz .- theta .* Mz
    w .-= Mu .* dot(u, w)
    Z[:, iteration] .= z
    KZ[:, iteration] .= Kz
    MZ[:, iteration] .= Mz

    for _ = 1:2, j = 1:iteration
      alpha = dot(view(V, :, j), w)
      H[j, iteration] += alpha
      w .-= alpha .* view(V, :, j)
    end
    H[iteration+1, iteration] = norm(w)
    H[iteration+1, iteration] > eps(beta) &&
      (V[:, iteration+1] .= w ./ H[iteration+1, iteration])
    coefficients = view(H, 1:iteration+1, 1:iteration) \
                   view(rhs, 1:iteration+1)
    projected_residual = norm(
      view(rhs, 1:iteration+1) -
      view(H, 1:iteration+1, 1:iteration) * coefficients,
    )
    used = iteration
    if projected_residual <= relative_tolerance * beta ||
       H[iteration+1, iteration] <= eps(beta)
      converged = projected_residual <= relative_tolerance * beta
      break
    end
  end
  correction = view(Z, :, 1:used) * coefficients
  Kcorrection = view(KZ, :, 1:used) * coefficients
  Mcorrection = view(MZ, :, 1:used) * coefficients
  equation_residual = -residual - (
    Kcorrection - theta .* Mcorrection .-
    Mu .* dot(u, Kcorrection - theta .* Mcorrection)
  )
  return (
    correction=correction,
    Kcorrection=Kcorrection,
    Mcorrection=Mcorrection,
    iterations=used,
    as_batches=used,
    k_products=used,
    m_products=used,
    converged=converged,
    relative_residual=norm(equation_residual) / beta,
  )
end

"Restarted single-vector Jacobi--Davidson with AS-preconditioned FGMRES."
function evp_jd_gmres_as(
  K,
  M,
  schwarz;
  maxiter,
  relative_tolerance,
  inner_relative_tolerance=1e-2,
  inner_maxiter=20,
  restart_dimension=20,
)
  u = normalize_M!(ones(size(K, 1)), M)
  Ku = K * u
  Mu = M * u
  V = reshape(copy(u), :, 1)
  KV = reshape(copy(Ku), :, 1)
  MV = reshape(copy(Mu), :, 1)
  lambda = dot(u, Ku)
  initial = norm(Ku - lambda .* Mu)
  history = [evp_entry(0, lambda, initial, initial; global_k_products=1, global_m_products=1)]
  inner_stats = NamedTuple[]
  linear_batches = 0
  inner_iterations = 0
  k_products = 1
  m_products = 1

  for outer = 1:maxiter
    reduced = eigen(Symmetric(V' * KV), Symmetric(V' * MV))
    coefficient = reduced.vectors[:, 1]
    u = V * coefficient
    Ku = KV * coefficient
    Mu = MV * coefficient
    scale = sqrt(dot(u, Mu))
    u ./= scale
    Ku ./= scale
    Mu ./= scale
    lambda = reduced.values[1]
    residual_vector = Ku - lambda .* Mu
    residual = norm(residual_vector)
    residual <= relative_tolerance * initial && break

    correction = evp_jd_correction(
      K,
      M,
      schwarz,
      u,
      Mu,
      lambda,
      residual_vector;
      relative_tolerance=inner_relative_tolerance,
      maxiter=inner_maxiter,
    )
    linear_batches += correction.as_batches
    inner_iterations += correction.iterations
    k_products += correction.k_products
    m_products += correction.m_products
    push!(
      inner_stats,
      (
        outer_iteration=outer,
        iterations=correction.iterations,
        converged=correction.converged,
        relative_residual=correction.relative_residual,
      ),
    )

    if size(V, 2) >= restart_dimension
      V = reshape(copy(u), :, 1)
      KV = reshape(copy(Ku), :, 1)
      MV = reshape(copy(Mu), :, 1)
    end
    s = copy(correction.correction)
    Ks = copy(correction.Kcorrection)
    Ms = copy(correction.Mcorrection)
    for _ = 1:2
      projection = V' * Ms
      s .-= V * projection
      Ks .-= KV * projection
      Ms .-= MV * projection
    end
    s_norm = sqrt(max(dot(s, Ms), 0.0))
    s_norm <= 100 * eps(Float64) && break
    s ./= s_norm
    Ks ./= s_norm
    Ms ./= s_norm
    V = hcat(V, s)
    KV = hcat(KV, Ks)
    MV = hcat(MV, Ms)

    reduced = eigen(Symmetric(V' * KV), Symmetric(V' * MV))
    coefficient = reduced.vectors[:, 1]
    u_next = V * coefficient
    Ku_next = KV * coefficient
    Mu_next = MV * coefficient
    scale = sqrt(dot(u_next, Mu_next))
    u_next ./= scale
    Ku_next ./= scale
    Mu_next ./= scale
    lambda_next = reduced.values[1]
    residual_next = norm(Ku_next - lambda_next .* Mu_next)
    u = u_next
    Ku = Ku_next
    Mu = Mu_next
    lambda = lambda_next
    push!(
      history,
      evp_entry(
        outer,
        lambda_next,
        residual_next,
        initial;
        linear_as_batches=linear_batches,
        global_k_products=k_products,
        global_m_products=m_products,
        inner_iterations=inner_iterations,
      ),
    )
    residual_next <= relative_tolerance * initial && break
  end
  return empty_evp_result(u, history; inner_stats)
end

"varDD result with local-pencil and reduced-space diagnostics."
function evp_vardd(
  K,
  M,
  dofspar;
  maxiter,
  relative_tolerance,
  history_depth=0,
)
  u0 = normalize_M!(ones(size(K, 1)), M)
  lambda0 = rayleigh(K, M, u0)
  initial = norm(evp_residual(K, M, u0, lambda0))
  local_stats = NamedTuple[]
  combination_stats = NamedTuple[]
  local_callback = (iteration, subdomain, info) -> push!(
    local_stats,
    merge((outer_iteration=iteration, subdomain=subdomain), info),
  )
  combination_callback = (iteration, candidates) -> begin
    basis = Solvers.orthonormal_basis(candidates)
    reduced_mass = Symmetric(basis' * (M * basis))
    reduced_stiffness = Symmetric(basis' * (K * basis))
    values = eigvals(reduced_stiffness, reduced_mass)
    push!(
      combination_stats,
      (
        outer_iteration=iteration,
        basis_columns=size(candidates, 2),
        effective_rank=size(basis, 2),
        mass_condition=cond(Matrix(reduced_mass)),
        relative_gap=length(values) > 1 ?
                     (values[2] - values[1]) / max(abs(values[1]), eps()) : NaN,
      ),
    )
  end
  u, _, energies, solutions, _ = Solvers.var_dd(
    Energies.GeneralizedRayleighQuotient(K, M),
    dofspar;
    u0,
    maxiter,
    tol=relative_tolerance * initial,
    history_depth,
    local_solve_callback=local_callback,
    subspace_callback=combination_callback,
    verbose=false,
  )
  history = NamedTuple[]
  for (index, (lambda, solution)) in enumerate(zip(energies, solutions))
    iteration = index - 1
    residual = norm(evp_residual(K, M, solution, lambda))
    completed = filter(stat -> stat.outer_iteration <= iteration, local_stats)
    critical = 0
    for sweep = 1:iteration
      sweep_iterations = [
        stat.iterations for stat in local_stats if stat.outer_iteration == sweep
      ]
      !isempty(sweep_iterations) && (critical += maximum(sweep_iterations))
    end
    push!(
      history,
      evp_entry(
        iteration,
        lambda,
        residual,
        initial;
        local_eigen_batches=iteration,
        local_iterations_critical=critical,
        local_iterations_total=sum(
          (stat.iterations for stat in completed);
          init=0,
        ),
        global_k_products=-1,
        global_m_products=-1,
      ),
    )
  end
  return empty_evp_result(u, history; local_stats, combination_stats)
end

function evp_method_result(
  method,
  K,
  M,
  dofspar,
  schwarz;
  maxiter,
  relative_tolerance,
  history_depth=1,
)
  if method == :var_dd
    return evp_vardd(
      K, M, dofspar; maxiter, relative_tolerance, history_depth=0
    )
  elseif method == :var_dd_history
    return evp_vardd(
      K, M, dofspar; maxiter, relative_tolerance, history_depth
    )
  elseif method == :lopsd_as
    return evp_lopsd_as(K, M, schwarz; maxiter, relative_tolerance)
  elseif method == :lobpcg_as
    return evp_lobpcg_as(K, M, schwarz; maxiter, relative_tolerance)
  elseif method == :jd_gmres_as
    return evp_jd_gmres_as(
      K,
      M,
      schwarz;
      maxiter,
      relative_tolerance,
      inner_relative_tolerance=1e-2,
      inner_maxiter=SMALL ? 8 : 20,
      restart_dimension=20,
    )
  elseif haskey(EVP_JD_GMRES_ITERATIONS, method)
    return evp_jd_gmres_as(
      K,
      M,
      schwarz;
      maxiter,
      relative_tolerance,
      # Fixed-work FGMRES(AS) correction solves, directly comparable across
      # the prescribed iteration budgets.
      inner_relative_tolerance=0.0,
      inner_maxiter=EVP_JD_GMRES_ITERATIONS[method],
      restart_dimension=20,
    )
  elseif haskey(EVP_LANCZOS_PCG_ITERATIONS, method)
    return evp_si_lanczos_pcg_as(
      K,
      M,
      schwarz;
      maxiter,
      relative_tolerance,
      # These are deliberately fixed-work applications of PCG(AS), not
      # accurate inner solves. Otherwise AS is hidden inside an effectively
      # exact shift-and-invert operation and is not a meaningful baseline.
      inner_relative_tolerance=0.0,
      inner_maxiter=EVP_LANCZOS_PCG_ITERATIONS[method],
      restart_dimension=20,
    )
  end
  throw(ArgumentError("unknown EVP method $method"))
end
