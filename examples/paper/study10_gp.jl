# Study 10: Henning--Jarlebring section 2.3 Gross--Pitaevskii example on
# [-8,8]^2 with kappa=500, harmonic-plus-optical trapping potential, and their
# prescribed polynomial initial state.
#   - gp_additive: independent local nonlinear Rayleigh-quotient minimizations
#   - gp_additive_history: additionally retain the preceding global iterate
#   - gp_quadratic: freeze the GP density for the local linear EVP solves
#   - gp_quadratic_history: frozen local EVPs with the preceding global iterate
#   - gp_quadratic_history_{2,3}: frozen local EVPs with q=3,4 history
#   - gp_tangent_quadratic[_history]: tangent physical-energy models with
#     q=1,2 and a full nonlinear combination step
#   - gp_projected_qemdd[_history]: projected Riemannian-Newton models with
#     q=1,2, rank-two local solves, and increment-anchored coarse bases
#   - gp_density_mix_{025,050,075}[_history]: charge-mixed frozen local EVPs
#     with alpha=0.25,0.5,0.75 and q=1,2
#   - gfdn_pcg_as_{1,2,4}: GFDN with a fixed-step PCG(AS) metric solve
#   - cg_gfdn_pcg_as_{1,2,4}: Fletcher--Reeves acceleration of those directions
# Cost unit: local subdomain minimizations; the additive local work can run in
# parallel, so one sweep has m units of work but a one-solve critical path. For
# the GFDN baselines, one AS application likewise consists of m local solves.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
using Optim

const HJ_GP_DOMAIN = (-8.0, 8.0, -8.0, 8.0)
const HJ_GP_KAPPA = 500.0

hj_gp_potential(x) =
  0.5 * (x[1]^2 + 4 * x[2]^2) + 10 * (sinpi(x[1])^2 + sinpi(x[2])^2)

hj_gp_initial(x) = (x[1]^2 - 8^2) * (x[2]^2 - 8^2)

hj_gp_discretization(N, m; overlap = 2) =
  FEMDiscretizations.FEM_GrossPitaevskii(
    N,
    m;
    P = hj_gp_potential,
    overlap = overlap,
    domain = HJ_GP_DOMAIN,
    quadrature_degree = 8,
  )

function hj_gp_initial_vector(U, M)
  u = collect(get_free_dof_values(interpolate_everywhere(hj_gp_initial, U)))
  Energies.normalize_M!(u, M)
  return u
end

function linear_ground_state(K, M)
  decomposition = eigen(Symmetric(Matrix(K)), Symmetric(Matrix(M)))
  u = decomposition.vectors[:, 1]
  Energies.normalize_M!(u, M)
  sum(u) < 0 && (u .*= -1)
  return u
end

function gp_reference(e, u0)
  n = length(u0)
  # The first column supplies the initial state; the identity columns span the
  # full space. Reference computation therefore uses the same second-level
  # abstraction as varDD rather than a GP-specific public solver.
  return Solvers.combine_step(
    e,
    hcat(u0, Matrix{Float64}(I, n, n));
    initial = u0,
    maxiter = 2000,
    tol = 1e-11,
  )
end

function gp_var_dd_history(e, dofspar, u0; maxiter, tol, kwargs...)
  solution, _, energy_history, solution_history, residual_history =
    Solvers.var_dd(
      e,
      dofspar;
      u0 = u0,
      maxiter = maxiter,
      tol = tol,
      verbose = false,
      kwargs...,
    )
  m = length(dofspar)
  history = Tuple{Int,Float64,Float64,Float64,Float64}[]
  for k in eachindex(energy_history)
    u = solution_history[k]
    residual = k == 1 ? Energies.residual_norm(e, u) : residual_history[k-1]
    push!(
      history,
      (
        (k - 1) * m,
        energy_history[k] / 2,
        residual,
        abs(dot(u, e.M * u) - 1),
        Energies.chemical_potential(e, u),
      ),
    )
  end
  return solution, history
end

function gp_history_entry(e, u, solves)
  return (
    solves,
    Energies.physical_energy(e, u),
    Energies.residual_norm(e, u),
    abs(dot(u, e.M * u) - 1),
    Energies.chemical_potential(e, u),
  )
end

"""
    gp_gfdn_au_exact_history(e, density_matrix, u0; maxiter, tol)

Apply Definition 5.12 of Henning--Jarlebring with an exact sparse solve of
`A(u_n) z_n = M u_n` and the adaptive step from (5.30). The scalar line search
is restricted to `0 <= tau <= 2`, the empirically recommended interval in the
paper. The returned history includes the initial state.
"""
function gp_gfdn_au_exact_history(e, density_matrix, u0; maxiter, tol = 0.0)
  u = copy(u0)
  Energies.normalize_M!(u, e.M)
  history = [gp_history_entry(e, u, 0)]
  taus = Float64[]

  for iteration = 1:maxiter
    A_u = sparse(e.K + e.beta .* density_matrix(u))
    z = cholesky(Symmetric(A_u)) \ (e.M * u)
    gamma = inv(dot(z, A_u * z))

    trial(tau) = (1 - tau) .* u .+ tau .* gamma .* z
    line_search = Optim.optimize(
      tau -> Energies.energy(e, trial(tau)),
      0.0,
      2.0,
      Optim.Brent();
      abs_tol = 1e-12,
      rel_tol = 1e-12,
    )
    tau = Optim.minimizer(line_search)
    u_new = trial(tau)
    Energies.normalize_M!(u_new, e.M)
    dot(u, e.M * u_new) < 0 && (u_new .*= -1)
    u = u_new
    push!(taus, tau)
    push!(history, gp_history_entry(e, u, iteration))
    tol > 0 && last(history)[3] < tol && break
  end
  return u, history, taus
end

"""
    run_hj_mesh_validation(; Ns=(32, 64, 128, 256), iterations=30)

Run the exact GFDN(a_u) for the section 2.3 problem on increasingly fine Q1
meshes. This is intentionally separate from the tractable common mesh used by
the DD comparison. The output can be checked against the paper values
`E_GS ≈ 10.8995` and `lambda_GS ≈ 27.7133`.
"""
function run_hj_mesh_validation(; Ns = (32, 64, 128, 256), iterations = 30)
  rows = (
    N = Int[],
    h = Float64[],
    ndofs = Int[],
    iterations = Int[],
    energy = Float64[],
    lambda = Float64[],
    energy_error_to_paper = Float64[],
    lambda_error_to_paper = Float64[],
    resnorm = Float64[],
  )
  for N in Ns
    K, M, quartic, cubic, density_matrix, _, U = hj_gp_discretization(N, 1)
    e = Energies.GrossPitaevskiiRayleighQuotient(
      K,
      M,
      HJ_GP_KAPPA,
      quartic,
      cubic,
    )
    u0 = hj_gp_initial_vector(U, M)
    u, _, _ =
      gp_gfdn_au_exact_history(e, density_matrix, u0; maxiter = iterations)
    energy = Energies.physical_energy(e, u)
    lambda = Energies.chemical_potential(e, u)
    push!(rows.N, N)
    push!(rows.h, 16 / N)
    push!(rows.ndofs, length(u))
    push!(rows.iterations, iterations)
    push!(rows.energy, energy)
    push!(rows.lambda, lambda)
    push!(rows.energy_error_to_paper, energy - 10.8995)
    push!(rows.lambda_error_to_paper, lambda - 27.7133)
    push!(rows.resnorm, Energies.residual_norm(e, u))
    @printf(
      "  N=%d, h=%.5f: E=%.10f, lambda=%.10f\n",
      N,
      16 / N,
      energy,
      lambda
    )
  end
  savetable("study10_gp_solutions.csv", solution_rows)
  savetable("study10_hj_mesh_validation.csv", rows)
  return rows
end

"Apply exactly `iterations` PCG steps with one-level additive Schwarz."
function gp_fixed_pcg_as(A, rhs, dofspar, iterations)
  iterations > 0 || throw(ArgumentError("iterations must be positive"))
  schwarz = schwarz_setup(A, dofspar)
  x = zeros(eltype(rhs), length(rhs))
  r = copy(rhs)
  z = apply_AS(schwarz, r)
  p = copy(z)
  rz = dot(r, z)
  for iteration = 1:iterations
    Ap = A * p
    denominator = dot(p, Ap)
    (!isfinite(denominator) || denominator <= 0) && break
    alpha = rz / denominator
    x .+= alpha .* p
    r .-= alpha .* Ap
    iteration == iterations && break
    z = apply_AS(schwarz, r)
    rz_new = dot(r, z)
    (!isfinite(rz_new) || rz_new <= 0) && break
    p .= z .+ (rz_new / rz) .* p
    rz = rz_new
  end
  return x
end

"GFDN/CG-GFDN with a fixed number of PCG(AS) metric iterations."
function gp_gfdn_au_pcg_as_history(
  e,
  density_matrix,
  dofspar,
  u0;
  conjugate,
  inner_iterations,
  maxiter,
  tol,
)
  inner_iterations > 0 ||
    throw(ArgumentError("inner_iterations must be positive"))
  m = length(dofspar)
  u = copy(u0)
  Energies.normalize_M!(u, e.M)
  history = [gp_history_entry(e, u, 0)]
  previous_direction = nothing
  previous_gradient_norm = 0.0

  for iteration = 1:maxiter
    density = density_matrix(u)
    A_u = sparse(e.K + e.beta .* density)
    lambda = dot(u, A_u * u)
    residual = A_u * u .- lambda .* (e.M * u)

    preconditioned = gp_fixed_pcg_as(A_u, residual, dofspar, inner_iterations)
    projected = preconditioned .- u .* dot(u, e.M * preconditioned)
    gradient_norm = dot(residual, projected)
    gradient_norm > 0 || break

    direction = -projected
    if conjugate && !isnothing(previous_direction)
      transported = previous_direction .- u .* dot(u, e.M * previous_direction)
      beta_fr = gradient_norm / previous_gradient_norm
      direction .+= beta_fr .* transported
      # Changing metrics can destroy descent; restart with the Sobolev
      # gradient whenever Fletcher--Reeves is not a descent direction.
      dot(residual, direction) < 0 || (direction .= -projected)
    end

    u_new = Solvers.combine_step(
      e,
      hcat(u, direction);
      initial = u,
      maxiter = 100,
      tol = 1e-10,
    )
    dot(u, e.M * u_new) < 0 && (u_new .*= -1)
    previous_direction = direction
    previous_gradient_norm = gradient_norm
    u = u_new
    solves = iteration * m * inner_iterations
    push!(history, gp_history_entry(e, u, solves))
    last(history)[3] < tol && break
  end
  return u, history
end

"""
    run_study10_local_work(; ms, betas, maxiter, tol)

Record the inner solver statistics of every local solve performed by nonlinear
and quadratic EMDD with `q=1` and `q=2`. One outer sweep consists of `m` local
solves. For nonlinear EMDD these are SCF solves; for quadratic EMDD they are
linear generalized EVP solves with the density frozen for the whole sweep.
"""
function run_study10_local_work(;
  N = SMALL ? 16 : 32,
  ms = SMALL ? [2] : [2, 4, 8],
  betas = SMALL ? [HJ_GP_KAPPA] : [10.0, 100.0, HJ_GP_KAPPA],
  overlap = 2,
  maxiter = SMALL ? 10 : 30,
  tol = 1e-6,
)
  rows = (
    method = String[],
    beta = Float64[],
    m = Int[],
    outer_iteration = Int[],
    subdomain = Int[],
    dimension = Int[],
    iterations = Int[],
    converged = Bool[],
  )
  K_ref, M_ref, _, _, _, _, U_ref =
    hj_gp_discretization(N, 1; overlap = overlap)
  paper_initial = hj_gp_initial_vector(U_ref, M_ref)

  for m in ms
    K, M, quartic, cubic, density_matrix, dofspar, U =
      hj_gp_discretization(N, m; overlap)
    u0 = hj_gp_initial_vector(U, M)
    for beta in betas
      e = Energies.GrossPitaevskiiRayleighQuotient(
        K,
        M,
        beta,
        quartic,
        cubic;
        density_matrix,
      )
      for (method, history_depth, frozen_gp_model) in (
        ("gp_additive", 0, false),
        ("gp_additive_history", 1, false),
        ("gp_quadratic", 0, true),
        ("gp_quadratic_history", 1, true),
      )
        stats = NamedTuple[]
        callback =
          (iteration, subdomain, info) -> push!(
            stats,
            merge((outer = iteration, subdomain = subdomain), info),
          )
        Solvers.var_dd(
          e,
          dofspar;
          u0 = u0,
          maxiter = maxiter,
          tol = tol,
          history_depth = history_depth,
          frozen_gp_model = frozen_gp_model,
          local_solve_callback = callback,
          verbose = false,
        )
        for stat in stats
          push!(rows.method, method)
          push!(rows.beta, beta)
          push!(rows.m, m)
          push!(rows.outer_iteration, stat.outer)
          push!(rows.subdomain, stat.subdomain)
          push!(rows.dimension, stat.dimension)
          push!(rows.iterations, stat.iterations)
          push!(rows.converged, stat.converged)
        end
        counts = [stat.iterations for stat in stats]
        @printf(
          "  kappa=%5.1f m=%d %-20s sweeps=%2d local solves=%3d inner its/solve: mean=%5.1f max=%3d\n",
          beta,
          m,
          method,
          length(counts) ÷ m,
          length(counts),
          sum(counts) / length(counts),
          maximum(counts),
        )
      end
    end
  end
  savetable("study10_gp_local_work.csv", rows)
  return rows
end

function run_study10()
  N = SMALL ? 16 : 32
  ms = SMALL ? [2] : [2, 4, 8]
  betas = SMALL ? [HJ_GP_KAPPA] : [10.0, 100.0, HJ_GP_KAPPA]
  overlap = 2

  # Keep the inexpensive visualization data available independently of the
  # cached nonlinear solves.
  partition_file = "study10_partitions.csv"
  partition_cache_matches = if isfile(datafile(partition_file))
    cached = loadtable(partition_file)
    all(cached.N .== N) && sort(unique(cached.m)) == ms
  else
    false
  end
  if FORCE || !partition_cache_matches
    partition_rows =
      (m = Int[], N = Int[], idx = Int[], owner = Int[], mult = Int[])
    for m in ms
      owner, mult = metis_cell_partition(N, m, overlap)
      for idx in eachindex(owner)
        push!(partition_rows.m, m)
        push!(partition_rows.N, N)
        push!(partition_rows.idx, idx)
        push!(partition_rows.owner, owner[idx])
        push!(partition_rows.mult, mult[idx])
      end
    end
    savetable(partition_file, partition_rows)
  end

  files = ("study10_gp_conv.csv", "study10_gp_solutions.csv")
  result_cache_matches = if all(isfile(datafile(file)) for file in files)
    cached_conv = loadtable(files[1])
    cached_solutions = loadtable(files[2])
    expected_methods = [
      "gp_additive",
      "gp_additive_history",
      "gp_quadratic",
      "gp_quadratic_history",
      "gp_quadratic_history_2",
      "gp_quadratic_history_3",
      "gp_tangent_quadratic",
      "gp_tangent_quadratic_history",
      "gp_projected_qemdd",
      "gp_projected_qemdd_history",
      "gp_density_mix_025",
      "gp_density_mix_025_history",
      "gp_density_mix_050",
      "gp_density_mix_050_history",
      "gp_density_mix_075",
      "gp_density_mix_075_history",
      "gfdn_pcg_as_1",
      "gfdn_pcg_as_2",
      "gfdn_pcg_as_4",
      "cg_gfdn_pcg_as_1",
      "cg_gfdn_pcg_as_2",
      "cg_gfdn_pcg_as_4",
    ]
    sort(unique(cached_conv.m)) == ms &&
      sort(unique(cached_conv.beta)) == sort(betas) &&
      sort(unique(cached_conv.method)) == sort(expected_methods) &&
      all(cached_solutions.N .== N)
  else
    false
  end
  if !FORCE && result_cache_matches
    println("study10: cached, skipping")
    return nothing
  end
  println("study10: Gross--Pitaevskii nonlinear eigenproblem")

  # The first two rows show increasing interaction strength; the final row is
  # the Henning--Jarlebring section 2.3 benchmark.
  tol = 1e-6
  maxiter = SMALL ? 10 : 30

  rows = (
    method = String[],
    beta = Float64[],
    m = Int[],
    iteration = Int[],
    solves = Int[],
    energy = Float64[],
    energy_gap = Float64[],
    resnorm = Float64[],
    mass_error = Float64[],
    lambda = Float64[],
  )
  solution_rows = (
    beta = Float64[],
    N = Int[],
    idx = Int[],
    value = Float64[],
    density = Float64[],
  )

  K_ref, M_ref, q_ref, g_ref, density_ref, _, U_ref =
    hj_gp_discretization(N, 1; overlap = overlap)
  paper_initial = hj_gp_initial_vector(U_ref, M_ref)
  references = Dict{Float64,Tuple{Float64,Vector{Float64}}}()
  for beta in betas
    e_ref = Energies.GrossPitaevskiiRayleighQuotient(
      K_ref,
      M_ref,
      beta,
      q_ref,
      g_ref;
      density_matrix = density_ref,
    )
    reference_solution, _, _ = gp_gfdn_au_exact_history(
      e_ref,
      density_ref,
      paper_initial;
      maxiter = 80,
      tol = 0.0,
    )
    reference_energy = Energies.physical_energy(e_ref, reference_solution)
    references[beta] = (reference_energy, copy(reference_solution))
    for idx in eachindex(reference_solution)
      push!(solution_rows.beta, beta)
      push!(solution_rows.N, N)
      push!(solution_rows.idx, idx)
      push!(solution_rows.value, reference_solution[idx])
      push!(solution_rows.density, reference_solution[idx]^2)
    end
    @printf(
      "  kappa = %.1f reference: E = %.10e, lambda = %.10e, residual = %.2e\n",
      beta,
      reference_energy,
      Energies.chemical_potential(e_ref, reference_solution),
      Energies.residual_norm(e_ref, reference_solution),
    )
  end

  function record(method, beta, m, reference_energy, history)
    for (iteration, entry) in enumerate(history)
      solves, energy, residual, mass_error, lambda = entry
      push!(rows.method, method)
      push!(rows.beta, beta)
      push!(rows.m, m)
      push!(rows.iteration, iteration - 1)
      push!(rows.solves, solves)
      push!(rows.energy, energy)
      push!(rows.energy_gap, abs(energy - reference_energy))
      push!(rows.resnorm, residual)
      push!(rows.mass_error, mass_error)
      push!(rows.lambda, lambda)
    end
  end

  for m in ms
    K, M, quartic, cubic, density_matrix, dofspar, U =
      hj_gp_discretization(N, m; overlap = overlap)
    u0 = hj_gp_initial_vector(U, M)
    for beta in betas
      e = Energies.GrossPitaevskiiRayleighQuotient(
        K,
        M,
        beta,
        quartic,
        cubic;
        density_matrix,
      )
      reference_energy, _ = references[beta]
      _, additive =
        gp_var_dd_history(e, dofspar, u0; maxiter = maxiter, tol = tol)
      _, with_history = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        history_depth = 1,
      )
      _, quadratic = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        frozen_gp_model = true,
      )
      _, quadratic_history = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        history_depth = 1,
        frozen_gp_model = true,
      )
      _, quadratic_history_2 = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        history_depth = 2,
        frozen_gp_model = true,
      )
      _, quadratic_history_3 = gp_var_dd_history(
        e,
        dofspar,
        u0;
        maxiter = maxiter,
        tol = tol,
        history_depth = 3,
        frozen_gp_model = true,
      )
      record("gp_additive", beta, m, reference_energy, additive)
      record("gp_additive_history", beta, m, reference_energy, with_history)
      record("gp_quadratic", beta, m, reference_energy, quadratic)
      record(
        "gp_quadratic_history",
        beta,
        m,
        reference_energy,
        quadratic_history,
      )
      record(
        "gp_quadratic_history_2",
        beta,
        m,
        reference_energy,
        quadratic_history_2,
      )
      record(
        "gp_quadratic_history_3",
        beta,
        m,
        reference_energy,
        quadratic_history_3,
      )
      for history_depth = 0:1
        _, tangent_history = gp_var_dd_history(
          e,
          dofspar,
          u0;
          maxiter,
          tol,
          history_depth,
          tangent_gp_model = true,
        )
        suffix = history_depth == 0 ? "" : "_history"
        record(
          "gp_tangent_quadratic$(suffix)",
          beta,
          m,
          reference_energy,
          tangent_history,
        )
      end
      for history_depth = 0:1
        _, projected_history = gp_var_dd_history(
          e,
          dofspar,
          u0;
          maxiter,
          tol,
          history_depth,
          projected_gp_model = true,
        )
        suffix = history_depth == 0 ? "" : "_history"
        record(
          "gp_projected_qemdd$(suffix)",
          beta,
          m,
          reference_energy,
          projected_history,
        )
      end
      for (alpha_tag, alpha) in (("025", 0.25), ("050", 0.5), ("075", 0.75))
        for history_depth = 0:1
          _, mixed_history = gp_var_dd_history(
            e,
            dofspar,
            u0;
            maxiter,
            tol,
            history_depth,
            frozen_gp_model = true,
            density_mixing_alpha = alpha,
          )
          suffix = history_depth == 0 ? "" : "_history"
          record(
            "gp_density_mix_$(alpha_tag)$(suffix)",
            beta,
            m,
            reference_energy,
            mixed_history,
          )
        end
      end
      for conjugate in (false, true), inner_iterations in (1, 2, 4)
        _, history = gp_gfdn_au_pcg_as_history(
          e,
          density_matrix,
          dofspar,
          u0;
          conjugate,
          inner_iterations,
          maxiter,
          tol,
        )
        prefix = conjugate ? "cg_gfdn" : "gfdn"
        record(
          "$(prefix)_pcg_as_$(inner_iterations)",
          beta,
          m,
          reference_energy,
          history,
        )
      end
      println("  kappa = $beta, m = $m done")
      # Checkpoint complete parameter blocks. Long full-study runs can then be
      # inspected after an unrelated baseline failure without exposing a
      # partially recorded block as valid data.
      savetable("study10_gp_conv.csv", rows)
    end
  end

  savetable("study10_gp_conv.csv", rows)
  savetable("study10_gp_solutions.csv", solution_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study10()
end
