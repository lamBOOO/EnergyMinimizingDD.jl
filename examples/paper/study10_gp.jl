# Study 10: Henning--Jarlebring section 2.3 Gross--Pitaevskii example on
# [-8,8]^2 with kappa=500, harmonic-plus-optical trapping potential, and their
# prescribed polynomial initial state.
#   - gp_additive: independent local nonlinear Rayleigh-quotient minimizations
#   - gp_additive_history: additionally retain the preceding global iterate
#   - gfdn_au_exact: exact GFDN(a_u), Definition 5.12 with optimal step (5.30)
#   - cg_gfdn_au_exact: exact metric inversion with Fletcher--Reeves history
#   - gfdn_au_as: energy-adaptive Sobolev gradient with one-level AS
#   - cg_gfdn_au_as: Fletcher--Reeves acceleration of the same direction
# Cost unit: local subdomain minimizations; the additive local work can run in
# parallel, so one sweep has m units of work but a one-solve critical path. For
# the GFDN baselines, one AS application likewise consists of m local solves.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
using Optim

const HJ_GP_DOMAIN = (-8.0, 8.0, -8.0, 8.0)
const HJ_GP_KAPPA = 500.0

hj_gp_potential(x) =
  0.5 * (x[1]^2 + 4 * x[2]^2) +
  10 * (sinpi(x[1])^2 + sinpi(x[2])^2)

hj_gp_initial(x) = (x[1]^2 - 8^2) * (x[2]^2 - 8^2)

hj_gp_discretization(N, m; overlap=2) =
  FEMDiscretizations.FEM_GrossPitaevskii(
    N,
    m;
    P=hj_gp_potential,
    overlap=overlap,
    domain=HJ_GP_DOMAIN,
    quadrature_degree=8,
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
    initial=u0,
    maxiter=2000,
    tol=1e-11,
  )
end

function gp_var_dd_history(e, dofspar, u0; maxiter, tol, kwargs...)
  solution, _, energy_history, solution_history, residual_history = Solvers.var_dd(
    e, dofspar; u0=u0, maxiter=maxiter, tol=tol, verbose=false, kwargs...
  )
  m = length(dofspar)
  history = Tuple{Int,Float64,Float64,Float64,Float64}[]
  for k in eachindex(energy_history)
    u = solution_history[k]
    residual = k == 1 ? Energies.residual_norm(e, u) : residual_history[k - 1]
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
function gp_gfdn_au_exact_history(
  e,
  density_matrix,
  u0;
  maxiter,
  tol=0.0,
)
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
      abs_tol=1e-12,
      rel_tol=1e-12,
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
function run_hj_mesh_validation(; Ns=(32, 64, 128, 256), iterations=30)
  rows = (
    N=Int[],
    h=Float64[],
    ndofs=Int[],
    iterations=Int[],
    energy=Float64[],
    lambda=Float64[],
    energy_error_to_paper=Float64[],
    lambda_error_to_paper=Float64[],
    resnorm=Float64[],
  )
  for N in Ns
    K, M, quartic, cubic, density_matrix, _, U = hj_gp_discretization(N, 1)
    e = Energies.GrossPitaevskiiRayleighQuotient(
      K, M, HJ_GP_KAPPA, quartic, cubic
    )
    u0 = hj_gp_initial_vector(U, M)
    u, _, _ = gp_gfdn_au_exact_history(
      e, density_matrix, u0; maxiter=iterations
    )
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
      "  N=%d, h=%.5f: E=%.10f, lambda=%.10f\n", N, 16 / N, energy, lambda
    )
  end
  savetable("study10_hj_mesh_validation.csv", rows)
  return rows
end

"Apply GFDN/CG-GFDN with either an exact or one-AS metric inversion."
function gp_gfdn_au_metric_history(
  e,
  density_matrix,
  u0;
  inverse,
  conjugate,
  maxiter,
  tol,
  dofspar=nothing,
)
  inverse in (:exact, :as) ||
    throw(ArgumentError("inverse must be :exact or :as"))
  inverse == :as && isnothing(dofspar) &&
    throw(ArgumentError("dofspar is required for the AS-inexact method"))
  m = isnothing(dofspar) ? 1 : length(dofspar)
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

    preconditioned = if inverse == :exact
      cholesky(Symmetric(A_u)) \ residual
    else
      # One inexact metric solve with the overlapping one-level AS operator.
      apply_AS(schwarz_setup(A_u, dofspar), residual)
    end
    projected = preconditioned .- u .* dot(u, e.M * preconditioned)
    gradient_norm = dot(residual, projected)
    gradient_norm > 0 || break

    direction = -projected
    if conjugate && !isnothing(previous_direction)
      transported = previous_direction .-
                    u .* dot(u, e.M * previous_direction)
      beta_fr = gradient_norm / previous_gradient_norm
      direction .+= beta_fr .* transported
      # Changing metrics can destroy descent; restart with the Sobolev
      # gradient whenever Fletcher--Reeves is not a descent direction.
      dot(residual, direction) < 0 || (direction .= -projected)
    end

    u_new = Solvers.combine_step(
      e,
      hcat(u, direction);
      initial=u,
      maxiter=100,
      tol=1e-10,
    )
    dot(u, e.M * u_new) < 0 && (u_new .*= -1)
    previous_direction = direction
    previous_gradient_norm = gradient_norm
    u = u_new
    solves = inverse == :exact ? iteration : iteration * m
    push!(history, gp_history_entry(e, u, solves))
    last(history)[3] < tol && break
  end
  return u, history
end

function gp_gfdn_au_as_history(
  e,
  density_matrix,
  dofspar,
  u0;
  conjugate,
  maxiter,
  tol,
)
  return gp_gfdn_au_metric_history(
    e,
    density_matrix,
    u0;
    inverse=:as,
    conjugate=conjugate,
    maxiter=maxiter,
    tol=tol,
    dofspar=dofspar,
  )
end

function gp_cg_gfdn_au_exact_history(
  e,
  density_matrix,
  u0;
  maxiter,
  tol,
)
  return gp_gfdn_au_metric_history(
    e,
    density_matrix,
    u0;
    inverse=:exact,
    conjugate=true,
    maxiter=maxiter,
    tol=tol,
  )
end

function run_study10()
  N = SMALL ? 16 : 32
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2

  # Keep the inexpensive visualization data available independently of the
  # cached nonlinear solves.
  partition_file = "study10_partitions.csv"
  if FORCE || !isfile(datafile(partition_file))
    partition_rows = (
      m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[]
    )
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
  if !needs_run(files...)
    println("study10: cached, skipping")
    return nothing
  end
  println("study10: Gross--Pitaevskii nonlinear eigenproblem")

  # The first two rows show increasing interaction strength; the final row is
  # the Henning--Jarlebring section 2.3 benchmark.
  betas = SMALL ? [HJ_GP_KAPPA] : [10.0, 100.0, HJ_GP_KAPPA]
  tol = 1e-6
  maxiter = SMALL ? 10 : 30

  rows = (
    method=String[],
    beta=Float64[],
    m=Int[],
    iteration=Int[],
    solves=Int[],
    energy=Float64[],
    energy_gap=Float64[],
    resnorm=Float64[],
    mass_error=Float64[],
    lambda=Float64[],
  )
  solution_rows = (
    beta=Float64[], N=Int[], idx=Int[], value=Float64[], density=Float64[]
  )

  K_ref, M_ref, q_ref, g_ref, density_ref, _, U_ref =
    hj_gp_discretization(N, 1; overlap=overlap)
  paper_initial = hj_gp_initial_vector(U_ref, M_ref)
  references = Dict{Float64,Tuple{Float64,Vector{Float64}}}()
  exact_histories = Dict()
  exact_cg_histories = Dict()
  for beta in betas
    e_ref = Energies.GrossPitaevskiiRayleighQuotient(
      K_ref, M_ref, beta, q_ref, g_ref
    )
    reference_solution, _, _ = gp_gfdn_au_exact_history(
      e_ref, density_ref, paper_initial; maxiter=80, tol=0.0
    )
    reference_energy = Energies.physical_energy(e_ref, reference_solution)
    references[beta] = (reference_energy, copy(reference_solution))
    _, exact_history, _ = gp_gfdn_au_exact_history(
      e_ref, density_ref, paper_initial; maxiter=maxiter, tol=0.0
    )
    exact_histories[beta] = exact_history
    _, exact_cg_history = gp_cg_gfdn_au_exact_history(
      e_ref, density_ref, paper_initial; maxiter=maxiter, tol=0.0
    )
    exact_cg_histories[beta] = exact_cg_history
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
    if beta == HJ_GP_KAPPA
      @printf(
        "    paper: E_GS ≈ 10.8995, lambda_GS ≈ 27.7133; exact GFDN energy gap after %d iterations = %.2e\n",
        maxiter,
        abs(last(exact_history)[2] - reference_energy),
      )
    end
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
      hj_gp_discretization(N, m; overlap=overlap)
    u0 = hj_gp_initial_vector(U, M)
    for beta in betas
      e = Energies.GrossPitaevskiiRayleighQuotient(K, M, beta, quartic, cubic)
      reference_energy, _ = references[beta]
      _, additive = gp_var_dd_history(e, dofspar, u0; maxiter=maxiter, tol=tol)
      _, with_history = gp_var_dd_history(
        e, dofspar, u0; maxiter=maxiter, tol=tol, history_depth=1
      )
      record("gp_additive", beta, m, reference_energy, additive)
      record("gp_additive_history", beta, m, reference_energy, with_history)
      record("gfdn_au_exact", beta, m, reference_energy, exact_histories[beta])
      record(
        "cg_gfdn_au_exact",
        beta,
        m,
        reference_energy,
        exact_cg_histories[beta],
      )
      _, gfdn = gp_gfdn_au_as_history(
        e,
        density_matrix,
        dofspar,
        u0;
        conjugate=false,
        maxiter=maxiter,
        tol=tol,
      )
      _, cg_gfdn = gp_gfdn_au_as_history(
        e,
        density_matrix,
        dofspar,
        u0;
        conjugate=true,
        maxiter=maxiter,
        tol=tol,
      )
      record("gfdn_au_as", beta, m, reference_energy, gfdn)
      record("cg_gfdn_au_as", beta, m, reference_energy, cg_gfdn)
      println("  kappa = $beta, m = $m done")
    end
  end

  savetable("study10_gp_conv.csv", rows)
  savetable("study10_gp_solutions.csv", solution_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study10()
end
