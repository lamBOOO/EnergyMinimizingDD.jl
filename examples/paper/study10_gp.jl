# Study 10: Gross--Pitaevskii ground state with the same trapping potential and
# overlapping partitions as the linear EVP studies.
#   - gp_additive: independent local nonlinear Rayleigh-quotient minimizations
#   - gp_additive_history: additionally retain the preceding global iterate
# Cost unit: local subdomain minimizations; the additive local work can run in
# parallel, so one sweep has m units of work but a one-solve critical path.

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

function run_study10()
  files = ("study10_gp_conv.csv", "study10_gp_solutions.csv")
  if !needs_run(files...)
    println("study10: cached, skipping")
    return nothing
  end
  println("study10: Gross--Pitaevskii nonlinear eigenproblem")

  N = SMALL ? 8 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  betas = SMALL ? [10.0] : [1.0, 10.0, 100.0]
  overlap = 2
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

  K_ref, M_ref, q_ref, g_ref, _, _ = FEMDiscretizations.FEM_GrossPitaevskii(
    N, 1; overlap=overlap
  )
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
    K, M, quartic, cubic, dofspar, _ = FEMDiscretizations.FEM_GrossPitaevskii(
      N, m; overlap=overlap
    )
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
      println("  beta = $beta, m = $m done")
    end
  end

  savetable("study10_gp_conv.csv", rows)
  savetable("study10_gp_solutions.csv", solution_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study10()
end
