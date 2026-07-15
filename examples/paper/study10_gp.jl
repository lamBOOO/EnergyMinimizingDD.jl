# Study 10: Gross--Pitaevskii ground state with the same trapping potential and
# overlapping partitions as the linear EVP studies.
#   - gp_additive: independent local nonlinear Rayleigh-quotient minimizations
#   - gp_additive_history: additionally retain the preceding global iterate
#   - gfdn_au_as: energy-adaptive Sobolev gradient with one-level AS
#   - cg_gfdn_au_as: Fletcher--Reeves acceleration of the same direction
# Cost unit: local subdomain minimizations; the additive local work can run in
# parallel, so one sweep has m units of work but a one-solve critical path. For
# the GFDN baselines, one AS application likewise consists of m local solves.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

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
    gp_gfdn_au_as_history(e, density_matrix, dofspar, u0; conjugate, maxiter, tol)

Apply an inexact energy-adaptive `a_u`-Sobolev gradient method. At every outer
iteration, one-level additive Schwarz for

    A_u = K + beta * C(u)

approximates the inverse metric action. `conjugate=true` adds a transported
Fletcher--Reeves direction, giving the CG-GFDN(a_u)+AS baseline. The existing
GP `combine_step` performs the optimal normalized line search in the span of
the current iterate and search direction.
"""
function gp_gfdn_au_as_history(
  e,
  density_matrix,
  dofspar,
  u0;
  conjugate,
  maxiter,
  tol,
)
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

    # One inexact a_u-metric solve, realized by the same overlapping
    # one-level AS construction as Studies 8 and 9.
    schwarz = schwarz_setup(A_u, dofspar)
    preconditioned = apply_AS(schwarz, residual)
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
    push!(history, gp_history_entry(e, u, iteration * m))
    last(history)[3] < tol && break
  end
  return u, history
end

function run_study10()
  N = SMALL ? 8 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2

  # Keep the inexpensive visualization data available independently of the
  # cached nonlinear solves.
  partition_file = "study10_partitions.csv"
  if !isfile(datafile(partition_file))
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

  betas = SMALL ? [10.0] : [1.0, 10.0, 100.0]
  tol = 1e-6
  maxiter = SMALL ? 15 : 80

  rows = (
    method=String[],
    beta=Float64[],
    m=Int[],
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

  K_ref, M_ref, q_ref, g_ref, _, _, _ =
    FEMDiscretizations.FEM_GrossPitaevskii(N, 1; overlap=overlap)
  linear_initial = linear_ground_state(K_ref, M_ref)
  references = Dict{Float64,Tuple{Float64,Vector{Float64}}}()
  for beta in betas
    e_ref = Energies.GrossPitaevskiiRayleighQuotient(
      K_ref, M_ref, beta, q_ref, g_ref
    )
    # Solve every reference problem independently from the same linear ground
    # state; no continuation in beta is used.
    reference_solution = gp_reference(e_ref, linear_initial)
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
      "  beta = %.1f reference: E = %.10e, residual = %.2e\n",
      beta,
      reference_energy,
      Energies.residual_norm(e_ref, reference_solution),
    )
  end

  function record(method, beta, m, reference_energy, history)
    for (solves, energy, residual, mass_error, lambda) in history
      push!(rows.method, method)
      push!(rows.beta, beta)
      push!(rows.m, m)
      push!(rows.solves, solves)
      push!(rows.energy, energy)
      push!(rows.energy_gap, abs(energy - reference_energy))
      push!(rows.resnorm, residual)
      push!(rows.mass_error, mass_error)
      push!(rows.lambda, lambda)
    end
  end

  for m in ms
    K, M, quartic, cubic, density_matrix, dofspar, _ =
      FEMDiscretizations.FEM_GrossPitaevskii(N, m; overlap=overlap)
    u0 = linear_ground_state(K, M)
    for beta in betas
      e = Energies.GrossPitaevskiiRayleighQuotient(K, M, beta, quartic, cubic)
      reference_energy, _ = references[beta]
      _, additive = gp_var_dd_history(e, dofspar, u0; maxiter=maxiter, tol=tol)
      _, with_history = gp_var_dd_history(
        e, dofspar, u0; maxiter=maxiter, tol=tol, history_depth=1
      )
      record("gp_additive", beta, m, reference_energy, additive)
      record("gp_additive_history", beta, m, reference_energy, with_history)
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
      println("  beta = $beta, m = $m done")
    end
  end

  savetable("study10_gp_conv.csv", rows)
  savetable("study10_gp_solutions.csv", solution_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study10()
end
