# Study 8: variable-coefficient elliptic problem — comparison against
# one-level Schwarz baselines on the SAME overlapping partition:
#   - var_dd_additive (m independent local solves)
#   - REMDD with q=1 and q=2 (local corrections weighted by a multiplicity PoU)
#   - two-level EMDD/REMDD with the Nicolaides coarse space
#   - var_dd_additive_history (additive plus the preceding global iterate)
#   - var_dd_additive_mix_* (post-combination damping with several weights)
#   - var_dd_multiplicative (m sequential local solves)
#   - damped additive Schwarz (theta = 1/max_multiplicity; undamped AS
#     diverges as a stationary iteration)
#   - restricted additive Schwarz (RAS, Cai-Sarkis), stationary
#   - CG preconditioned with additive Schwarz
#   - right-preconditioned GMRES with restricted additive Schwarz
# Cost unit: subdomain solves (one sweep / preconditioner application = m).

isdefined(Main, :PAPER_COMMON) || include("common.jl")

"Diffusion coefficient for the Study 8 manufactured-solution problem."
study8_diffusion(x) = 2 + sinpi(2 * x.data[1] + 3 * x.data[2])

function study8_exact_solution(x, y_frequency)
  x_coordinate, y_coordinate = x.data
  return exp(3 * x_coordinate * y_coordinate) *
         sinpi(x_coordinate) *
         sinpi(y_frequency * y_coordinate)
end

"Positive exact solution for the Study 8 manufactured-solution problem."
study8_exact_solution(x) = study8_exact_solution(x, 1)

"Sign-changing exact solution used by the third row of Figure 12b."
study8_sign_changing_exact_solution(x) = study8_exact_solution(x, 2)

function study8_forcing(x, y_frequency)
  x_coordinate, y_coordinate = x.data
  exponential = exp(3 * x_coordinate * y_coordinate)
  sine_x, sine_y = sinpi(x_coordinate), sinpi(y_frequency * y_coordinate)
  cosine_x, cosine_y = cospi(x_coordinate), cospi(y_frequency * y_coordinate)

  derivative_x = exponential * sine_y * (
    3 * y_coordinate * sine_x + pi * cosine_x
  )
  derivative_y = exponential * sine_x * (
    3 * x_coordinate * sine_y + y_frequency * pi * cosine_y
  )
  laplacian = exponential * (
    (9 * (x_coordinate^2 + y_coordinate^2) -
     (1 + y_frequency^2) * pi^2) * sine_x * sine_y +
    6pi * y_coordinate * cosine_x * sine_y +
    6 * y_frequency * pi * x_coordinate * sine_x * cosine_y
  )
  phase_cosine = cospi(2 * x_coordinate + 3 * y_coordinate)
  diffusion_gradient_dot_solution_gradient = phase_cosine * (
    2pi * derivative_x + 3pi * derivative_y
  )
  return -study8_diffusion(x) * laplacian -
         diffusion_gradient_dot_solution_gradient
end

"Forcing `-div(a grad(u))` for the positive Study 8 exact solution."
study8_forcing(x) = study8_forcing(x, 1)

"Forcing `-div(a grad(u))` for the sign-changing Figure 12b solution."
study8_sign_changing_forcing(x) = study8_forcing(x, 2)

function study8_problem_setup(N, m, overlap; kwargs...)
  return FEMDiscretizations.FEM_Schroedinger(
    N,
    m;
    P = x -> 0.0,
    f = study8_forcing,
    diffusion = study8_diffusion,
    overlap = overlap,
    kwargs...,
  )
end

function study8_sign_changing_problem_setup(N, m, overlap; kwargs...)
  return FEMDiscretizations.FEM_Schroedinger(
    N,
    m;
    P = x -> 0.0,
    f = study8_sign_changing_forcing,
    diffusion = study8_diffusion,
    overlap = overlap,
    kwargs...,
  )
end

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

"""
    gmres_ras(K, b, S; maxiter, tol)

Unrestarted right-preconditioned GMRES for `K*M_RAS^{-1} y = b`, starting
from the same all-ones physical iterate as the other Study 8 methods. The
reported residual is the true physical residual represented by the Arnoldi
least-squares problem. One iteration uses one global `K` application and one
parallel batch of `m` RAS local solves.
"""
function gmres_ras(K, b, S; maxiter, tol)
  n = length(b)
  m = nsub(S)
  x0 = ones(n)
  r0 = b - K * x0
  beta = norm(r0)
  hist = [(0, beta)]
  beta < tol && return hist

  V = zeros(n, maxiter + 1)
  Z = zeros(n, maxiter)
  H = zeros(maxiter + 1, maxiter)
  V[:, 1] .= r0 ./ beta
  rhs = zeros(maxiter + 1)
  rhs[1] = beta

  for k = 1:maxiter
    Z[:, k] .= apply_RAS(S, view(V, :, k))
    w = K * view(Z, :, k)
    # Two-pass modified Gram--Schmidt keeps the unrestarted Arnoldi basis
    # reliable enough for the stringent residual tolerances in this study.
    for pass = 1:2
      for j = 1:k
        coefficient = dot(view(V, :, j), w)
        H[j, k] += coefficient
        w .-= coefficient .* view(V, :, j)
      end
    end
    H[k+1, k] = norm(w)
    H[k+1, k] > eps(beta) && (V[:, k+1] .= w ./ H[k+1, k])

    coefficients = view(H, 1:k+1, 1:k) \ view(rhs, 1:k+1)
    residual = norm(
      view(rhs, 1:k+1) - view(H, 1:k+1, 1:k) * coefficients
    )
    push!(hist, (k * m, residual))
    residual < tol && return hist
    H[k+1, k] <= eps(beta) && return hist
  end
  return hist
end

function var_dd_linear_history(K, b, dofspar; maxiter, tol, kwargs...)
  _, _, _, _, resnorm_hist = Solvers.var_dd(
    Energies.QuadraticEnergy(K, b),
    dofspar;
    maxiter = maxiter,
    tol = tol,
    verbose = false,
    kwargs...,
  )
  m = length(dofspar)
  initial_residual = norm(b - K * ones(length(b)))
  return vcat(
    [(0, initial_residual)],
    [(k * m, rn) for (k, rn) in enumerate(resnorm_hist)],
  )
end

"Generate one five-method reference row used in Figure 12b."
function run_study8_paper_problem(file, problem_label, problem_setup)
  if !needs_run(file)
    println("study8 $problem_label: cached, skipping")
    return
  end
  println("study8 $problem_label: paper comparison")
  Random.seed!(1)

  N = SMALL ? 16 : 64
  ms = SMALL ? [4] : [4, 16, 64]
  overlap = 2
  relative_tolerance = SMALL ? 1e-7 : 1e-10
  maxsweeps = SMALL ? 50 : 400
  reference_N = SMALL ? 16 : 64
  reference_owners = Dict(m => metis_cell_owners(reference_N, m) for m in ms)
  rows = (
    method = String[],
    m = Int[],
    solves = Int[],
    resnorm = Float64[],
  )
  record(method, m, hist) = for (solves, residual) in hist
    push!(rows.method, method)
    push!(rows.m, m)
    push!(rows.solves, solves)
    push!(rows.resnorm, residual)
  end

  for m in ms
    owners = prolong_cell_owners(reference_owners[m], N)
    K, _, b, dofspar, _, core_dofs = problem_setup(
      N,
      m,
      overlap;
      cell_owners = owners,
      return_core_partition = true,
    )
    S = schwarz_setup(K, dofspar; core_dofs)
    tol = relative_tolerance * norm(b - K * ones(length(b)))
    record(
      "var_dd_additive",
      m,
      var_dd_linear_history(K, b, dofspar; maxiter = maxsweeps, tol = tol),
    )
    record(
      "var_dd_additive_history",
      m,
      var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter = maxsweeps,
        tol = tol,
        history_depth = 1,
      ),
    )
    record("ras", m, ras_stationary(K, b, S; maxsweeps, tol))
    record("pcg_as", m, pcg_as(K, b, S; maxiter = maxsweeps, tol))
    record("gmres_ras", m, gmres_ras(K, b, S; maxiter = maxsweeps, tol))
    println("  $problem_label m = $m done")
  end
  savetable(file, rows)
end

run_study8_poisson_paper() = run_study8_paper_problem(
  "study8_linear_cmp_poisson.csv",
  "standard Poisson",
  laplace_setup,
)

run_study8_sign_changing_paper() = run_study8_paper_problem(
  "study8_linear_cmp_sign_changing.csv",
  "sign-changing variable coefficient",
  study8_sign_changing_problem_setup,
)

function run_study8()
  run_study8_poisson_paper()
  run_study8_sign_changing_paper()
  files = (
    "study8_linear_cmp.csv",
    "study8_partitions.csv",
    "study8_linear_work.csv",
  )
  if !needs_run(files...)
    println("study8: cached, skipping")
    return
  end
  println("study8: variable-coefficient elliptic problem vs Schwarz baselines")
  Random.seed!(1)

  N = SMALL ? 16 : 64
  ms = SMALL ? [4] : [4, 16, 64]
  overlap = 2
  relative_tolerance = SMALL ? 1e-7 : 1e-10
  maxsweeps = SMALL ? 50 : 400
  # METIS partitions this reference grid and the owners are prolonged to the
  # N x N mesh, so partition_reference_N must divide N. Setting it equal to N
  # makes the prolongation the identity, i.e. METIS runs on the actual mesh.
  partition_reference_N = SMALL ? 16 : 64
  reference_owners = Dict(
    m => metis_cell_owners(partition_reference_N, m) for m in ms
  )

  rows = (
    method = String[],
    m = Int[],
    solves = Int[],
    resnorm = Float64[],
  )
  part_rows = (
    m = Int[],
    N = Int[],
    idx = Int[],
    owner = Int[],
    mult = Int[],
  )
  work_rows = (
    method = String[],
    m = Int[],
    local_batches = Int[],
    local_solves = Int[],
    global_matvecs = Int[],
    resnorm = Float64[],
  )
  record(method, m, hist) = for (s, rn) in hist
    push!(rows.method, method)
    push!(rows.m, m)
    push!(rows.solves, s)
    push!(rows.resnorm, rn)
    push!(work_rows.method, method)
    push!(work_rows.m, m)
    push!(work_rows.local_batches, s ÷ m)
    push!(work_rows.local_solves, s)
    # Krylov methods have a precise ideal count: the initial residual plus
    # one operator application per preconditioned iteration. For the
    # variational methods a generic count is implementation-dependent, so -1
    # deliberately denotes "not comparable" rather than inventing a count.
    push!(
      work_rows.global_matvecs,
      method in ("pcg_as", "gmres_ras") ? 1 + s ÷ m : -1,
    )
    push!(work_rows.resnorm, rn)
  end

  for m in ms
    owners = prolong_cell_owners(reference_owners[m], N)
    K, M, b, dofspar, U, core_dofs = study8_problem_setup(
      N,
      m,
      overlap;
      cell_owners = owners,
      return_core_partition = true,
    )
    S = schwarz_setup(K, dofspar; core_dofs)
    tol = relative_tolerance * norm(b - K * ones(length(b)))
    nicolaides_basis = Solvers.nicolaides_coarse_basis(K, core_dofs, dofspar)

    owner, mult = cell_partition_overlap(N, m, overlap, owners)
    for idx in eachindex(owner)
      push!(part_rows.m, m)
      push!(part_rows.N, N)
      push!(part_rows.idx, idx)
      push!(part_rows.owner, owner[idx])
      push!(part_rows.mult, mult[idx])
    end

    # Every sweep performs m local solves. The additive solves can run in
    # parallel, while the multiplicative sweep has a serial critical path of m.
    record(
      "var_dd_additive",
      m,
      var_dd_linear_history(K, b, dofspar; maxiter = maxsweeps, tol = tol),
    )
    record(
      "remdd_q1",
      m,
      var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter = maxsweeps,
        tol = tol,
        restriction = :partition_of_unity,
      ),
    )
    record(
      "var_dd_additive_history",
      m,
      var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter = maxsweeps,
        tol = tol,
        history_depth = 1,
      ),
    )
    record(
      "remdd_q2",
      m,
      var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter = maxsweeps,
        tol = tol,
        restriction = :partition_of_unity,
        history_depth = 1,
      ),
    )
    for (method, restriction, history_depth) in (
      ("emdd_q1_nicolaides", :none, 0),
      ("emdd_q2_nicolaides", :none, 1),
      ("remdd_q1_nicolaides", :partition_of_unity, 0),
      ("remdd_q2_nicolaides", :partition_of_unity, 1),
    )
      record(
        method,
        m,
        var_dd_linear_history(
          K,
          b,
          dofspar;
          maxiter = maxsweeps,
          tol = tol,
          restriction = restriction,
          history_depth = history_depth,
          coarse_basis = nicolaides_basis,
        ),
      )
    end
    for (method, omega) in (
      ("var_dd_additive_mix_025", 0.25),
      ("var_dd_additive_mix_05", 0.5),
      ("var_dd_additive_mix_075", 0.75),
    )
      record(
        method,
        m,
        var_dd_linear_history(
          K,
          b,
          dofspar;
          maxiter = maxsweeps,
          tol = tol,
          mixing_omega = omega,
        ),
      )
    end
    record(
      "var_dd_multiplicative",
      m,
      var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter = maxsweeps,
        tol = tol,
        sweep = :multiplicative,
      ),
    )

    theta = 1 / S.max_mult
    record("as", m, as_stationary(K, b, S; theta, maxsweeps, tol))
    record("ras", m, ras_stationary(K, b, S; maxsweeps, tol))
    record("pcg_as", m, pcg_as(K, b, S; maxiter = maxsweeps, tol))
    record("gmres_ras", m, gmres_ras(K, b, S; maxiter = maxsweeps, tol))
    println("  m = $m done (AS damping theta = $(round(theta; digits = 3)))")
  end
  savetable("study8_linear_cmp.csv", rows)
  savetable("study8_partitions.csv", part_rows)
  savetable("study8_linear_work.csv", work_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study8()
end
