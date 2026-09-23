# Gross--Pitaevskii comparison used by Figure 14b of the paper.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
using Optim

const HJ_GP_DOMAIN = (-8.0, 8.0, -8.0, 8.0)
const HJ_GP_KAPPA = 500.0

hj_gp_potential(x) =
  0.5 * (x[1]^2 + 4 * x[2]^2) + 10 * (sinpi(x[1])^2 + sinpi(x[2])^2)
hj_gp_initial(x) = (x[1]^2 - 8^2) * (x[2]^2 - 8^2)

hj_gp_discretization(N, m; overlap=2) =
  FEMDiscretizations.FEM_GrossPitaevskii(
    N, m;
    P=hj_gp_potential,
    overlap,
    domain=HJ_GP_DOMAIN,
    quadrature_degree=8,
  )

function hj_gp_initial_vector(U, M)
  u = collect(get_free_dof_values(interpolate_everywhere(hj_gp_initial, U)))
  Energies.normalize_M!(u, M)
  return u
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

function gp_var_dd_history(e, dofs, u0; maxiter, tol, kwargs...)
  result = Solvers.var_dd(
    e, dofs; u0, maxiter, tol, verbose=false, kwargs...
  )
  solution, energies = result.u, result.energy_history
  iterates, residuals = result.iterate_history, result.residual_history
  m = length(dofs)
  history = Tuple{Int,Float64,Float64,Float64,Float64}[]
  for index in eachindex(energies)
    u = iterates[index]
    residual = index == 1 ? Energies.residual_norm(e, u) : residuals[index-1]
    push!(history, (
      (index - 1) * m,
      energies[index] / 2,
      residual,
      abs(dot(u, e.M * u) - 1),
      Energies.chemical_potential(e, u),
    ))
  end
  return solution, history
end

"GFDN with an exact metric solve, used to compute the reference state."
function gp_gfdn_exact(e, density_matrix, u0; maxiter)
  u = copy(u0)
  Energies.normalize_M!(u, e.M)
  for _ = 1:maxiter
    A = sparse(e.K + e.beta .* density_matrix(u))
    z = cholesky(Symmetric(A)) \ (e.M * u)
    gamma = inv(dot(z, A * z))
    trial(tau) = (1 - tau) .* u .+ tau .* gamma .* z
    search = Optim.optimize(
      tau -> Energies.energy(e, trial(tau)), 0.0, 2.0, Optim.Brent();
      abs_tol=1e-12, rel_tol=1e-12,
    )
    next = trial(Optim.minimizer(search))
    Energies.normalize_M!(next, e.M)
    dot(u, e.M * next) < 0 && (next .*= -1)
    u = next
  end
  return u
end

function gp_fixed_pcg_as(
  A, rhs, dofs, iterations;
  initial=zeros(eltype(rhs), length(rhs)),
)
  schwarz = schwarz_setup(A, dofs)
  x = copy(initial)
  cg!(
    x, A, rhs;
    Pl=ASPreconditioner(schwarz),
    maxiter=iterations,
    abstol=0.0,
    reltol=0.0,
  )
  return x
end

"Minimize the GP energy on the retracted ray R_M(u + tau*d), tau >= 0."
function gp_retracted_line_search(e, u, direction)
  # theta in [0, pi/2] parametrizes the complete ray tau = tan(theta),
  # including its limit as tau tends to infinity.
  trial(theta) = cos(theta) .* u .+ sin(theta) .* direction
  objective(theta) = Energies.energy(e, trial(theta))
  search = Optim.optimize(
    objective, 0.0, pi / 2, Optim.Brent(); abs_tol=1e-12, rel_tol=1e-12,
  )
  next = trial(Optim.minimizer(search))
  Energies.normalize_M!(next, e.M)
  return next
end

"""
Paper GFDN(a_u) and Fletcher--Reeves CG-GFDN(a_u), with PCG(AS) metric
solves. In matrix notation, `A` represents the current inner product `a_u`,
`M` represents the L2 inner product, and `metric_inverse_u` approximates the
paper's `A_u^(-1)u`, i.e. the solution of `A*w = M*u`.

Implemented with assistance from ChatGPT 5.6 Sol, following P. Henning and
E. Jarlebring, "The Gross--Pitaevskii Equation and Eigenvector Nonlinearities:
Numerical Methods and Algorithms," SIAM Review 67(2), 256--317 (2025),
Definition 5.12, Remark 5.14, and equations (5.40)--(5.42).
"""
function gp_gfdn_pcg_as_history(
  e, density_matrix, dofs, u0;
  conjugate, inner_iterations, maxiter, tol,
)
  m = length(dofs)
  u = copy(u0)
  Energies.normalize_M!(u, e.M)
  history = [gp_history_entry(e, u, 0)]
  previous_direction = nothing
  previous_gradient_norm_squared = 0.0
  for iteration = 1:maxiter
    A = sparse(e.K + e.beta .* density_matrix(u))
    lambda = dot(u, A * u)

    # The paper uses the exact solution of A*w=M*u and Remark 5.14 permits a
    # few linear-solver steps. We use fixed PCG(AS) work and warm-start with
    # u/lambda; its initial residual is the GP residual scaled by -1/lambda.
    metric_inverse_u = gp_fixed_pcg_as(
      A, e.M * u, dofs, inner_iterations; initial=u ./ lambda,
    )
    normalization = dot(u, e.M * metric_inverse_u)
    isfinite(normalization) && normalization > 0 || break

    # For an exact metric solve, u'M*w = w'A*w, so this is both Definition
    # 5.12 and the projection in (5.40). With truncated warm-started PCG these
    # scalars can differ; we retain u'M*w, as required by the tangent projector.
    gradient = u .- metric_inverse_u ./ normalization

    # Equations (5.40)--(5.41): a_u-orthogonal transport and the
    # Fletcher--Reeves quotient in the changing a_u metric.
    gradient_norm_squared = dot(gradient, A * gradient)
    isfinite(gradient_norm_squared) && gradient_norm_squared > 0 || break
    direction = -gradient
    if conjugate && !isnothing(previous_direction)
      beta = gradient_norm_squared / previous_gradient_norm_squared
      candidate = beta .* previous_direction .- u
      direction = candidate .- metric_inverse_u .* (
        dot(u, e.M * candidate) / normalization
      )
    end

    # Equation (5.42): optimal energy step along the normalized search ray.
    next = gp_retracted_line_search(e, u, direction)
    previous_direction = direction
    previous_gradient_norm_squared = gradient_norm_squared
    u = next
    push!(history, gp_history_entry(e, u, iteration * m * inner_iterations))
    last(history)[3] < tol && break
  end
  return u, history
end

function run_study10()
  N = SMALL ? 16 : 32
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  partition_file = "study10_partitions.csv"
  if needs_run(partition_file)
    partition_rows = (m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[])
    for m in ms
      owner, multiplicity = metis_cell_partition(N, m, overlap)
      for index in eachindex(owner)
        push!(partition_rows.m, m)
        push!(partition_rows.N, N)
        push!(partition_rows.idx, index)
        push!(partition_rows.owner, owner[index])
        push!(partition_rows.mult, multiplicity[index])
      end
    end
    savetable(partition_file, partition_rows)
  end

  files = ("study10_gp_conv.csv", "study10_gp_solutions.csv")
  if !needs_run(files...)
    println("study10: cached, skipping")
    return
  end
  tolerance = 1e-6
  maxiterations = SMALL ? 10 : 30
  rows = (
    method=String[], beta=Float64[], m=Int[], iteration=Int[], solves=Int[],
    energy=Float64[], energy_gap=Float64[], resnorm=Float64[],
    mass_error=Float64[], lambda=Float64[],
  )
  solutions = (
    beta=Float64[], N=Int[], idx=Int[], value=Float64[], density=Float64[],
  )

  K, M, quartic, cubic, density_matrix, _, U =
    hj_gp_discretization(N, 1; overlap)
  initial = hj_gp_initial_vector(U, M)
  reference_energy_model = Energies.GrossPitaevskiiRayleighQuotient(
    K, M, HJ_GP_KAPPA, quartic, cubic; density_matrix
  )
  reference = gp_gfdn_exact(
    reference_energy_model, density_matrix, initial; maxiter=80
  )
  reference_energy = Energies.physical_energy(reference_energy_model, reference)
  for index in eachindex(reference)
    push!(solutions.beta, HJ_GP_KAPPA)
    push!(solutions.N, N)
    push!(solutions.idx, index)
    push!(solutions.value, reference[index])
    push!(solutions.density, reference[index]^2)
  end

  function record(method, m, history)
    for (iteration, entry) in enumerate(history)
      solves, energy, residual, mass_error, lambda = entry
      push!(rows.method, method)
      push!(rows.beta, HJ_GP_KAPPA)
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
    K, M, quartic, cubic, density_matrix, dofs, U =
      hj_gp_discretization(N, m; overlap)
    initial = hj_gp_initial_vector(U, M)
    energy = Energies.GrossPitaevskiiRayleighQuotient(
      K, M, HJ_GP_KAPPA, quartic, cubic; density_matrix
    )
    for history_depth = 0:1
      _, history = gp_var_dd_history(
        energy, dofs, initial;
        maxiter=maxiterations, tol=tolerance, history_depth,
      )
      record(history_depth == 0 ? "gp_additive" : "gp_additive_history", m, history)
      _, projected = gp_var_dd_history(
        energy, dofs, initial;
        maxiter=maxiterations, tol=tolerance, history_depth,
        projected_gp_model=true,
      )
      record(
        history_depth == 0 ? "gp_projected_qemdd" :
                             "gp_projected_qemdd_history",
        m,
        projected,
      )
    end
    for conjugate in (false, true), inner_iterations in (1, 2, 4)
      _, history = gp_gfdn_pcg_as_history(
        energy, density_matrix, dofs, initial;
        conjugate, inner_iterations, maxiter=maxiterations, tol=tolerance,
      )
      prefix = conjugate ? "cg_gfdn" : "gfdn"
      record("$(prefix)_pcg_as_$(inner_iterations)", m, history)
    end
    println("  kappa = $(HJ_GP_KAPPA), m = $m done")
    savetable(files[1], rows)
  end
  savetable(files[2], solutions)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_study10()
