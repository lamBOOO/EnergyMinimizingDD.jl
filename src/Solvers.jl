module Solvers

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf
using Optim

using EnergyMinimizingDD.Energies

"""
  inf_step(e::Energies.AbstractEnergy{Float64}, u_current::Vector{Float64}, idx_sub::AbstractVector)

Perform a local optimization step in the subdomain for variational domain decomposition.

This is an abstract method that must be implemented by concrete subtypes of `AbstractEnergy`.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: The energy functional object
- `u_current::Vector{Float64}`: Current solution vector
- `idx_sub::AbstractVector`: Indices defining the subdomain

# Throws
- `ErrorException`: Always throws since this is not implemented for the abstract type

# Notes
Concrete implementations should override this method to provide specific behavior
for different energy types in the variational domain decomposition framework.
"""
function inf_step(
  e::Energies.AbstractEnergy{Float64},
  u_current::Vector{Float64},
  idx_sub::AbstractVector,
)
  # TODO: Use the more efficient DD operations with views
  throw(ErrorException("inf_step not implemented for $(typeof(e))"))
end

function inf_step(
  e::Energies.QuadraticEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  if iszero(norm(u_cur))
    @warn "the zero current iterate contributes no enriched direction; minimizing over the local coordinate space only" maxlog=1
    active = collect(Int, idx_sub)
    isempty(active) && return copy(u_cur)
    u_new = zeros(Float64, length(u_cur))
    u_new[active] .= e.A[active, active] \ e.b[active]
    return u_new
  end

  α_new = quadratic_local_coefficients(e, u_cur, idx_sub)

  # Return the actual minimizer in span{u_cur, e_j : j ∈ idx_sub}.
  # Rescaling by α_new[1] would generate the same line when that coefficient is
  # nonzero, but fails for valid local minimizers with α_new[1] == 0.
  return reconstruct_implicit!(α_new, u_cur, idx_sub)
end

function quadratic_local_coefficients(
  e::Energies.QuadraticEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  localdim = 1 + length(idx_sub)

  A, b = e.A, e.b

  # u_cur = copy(u_cur)
  # zero_out_local!(u_cur, idx_sub)

  # Extract relevant rows/columns from A and b
  if length(idx_sub) > 0
    # Build the local matrix more efficiently
    A_local = zeros(localdim, localdim)
    b_local = zeros(localdim)

    # First row/column: u_cur' * A * [u_cur, e_j1, e_j2, ...]
    Au = A * u_cur
    A_local[1, 1] = dot(u_cur, Au)  # u' * A * u
    b_local[1] = dot(b, u_cur)      # b' * u

    # First row/column: u_cur' * A * e_j
    for (k, j) in pairs(idx_sub)
      A_local[1, k+1] = Au[j]  # u' * A * e_j = (A * u)[j]
      A_local[k+1, 1] = Au[j]  # e_j' * A * u = (A * u)[j] (symmetric)
      b_local[k+1] = b[j]      # b' * e_j = b[j]
    end

    # Remaining entries: e_i' * A * e_j = A[i,j]
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        A_local[k1+1, k2+1] = A[j1, j2]
      end
    end

    # A_local = K' * A * K
    # where K = [u_cur, e_j1, e_j2, ...] and j1, j2 are subdomain indices
  else
    # Degenerate case: only current solution
    A_local = reshape([dot(u_cur, A * u_cur)], 1, 1)
    b_local = reshape([dot(b, u_cur)], 1)
  end

  @debug "Size of A_local: $(size(A_local))"
  # Solve the local quadratic minimization problem
  # min_{α} ½ α' A_local α - b_local' α
  # where α is the coefficient vector in the basis [u_cur, e_j1, e_j2, ...]
  return A_local \ b_local
end

"""
  inf_step(u_cur, K, M, idx_sub)

Calcualted x_new = argmin_{x ∈ span{u_cur, e_j, j ∈ idx_sub} \\ {0}} (x' K x)/(x' M x)
where {e_j} are standard basis vectors. The output is re-normalized in the M-norm.
"""
function generalized_rayleigh_inf_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  localdim = 1 + length(idx_sub)

  K, M = e.A, e.B

  # remove local constribution from u_curr: u_curr - Ri^T Ri u_cur
  # => set entries in idx_sub to zero
  u_cur = copy(u_cur)
  zero_out_local!(u_cur, idx_sub)

  # Extract relevant rows/columns from K and M
  if length(idx_sub) > 0
    # The augmented pencil has arrowhead form. Keeping it sparse avoids the
    # artificial dense O(n_i^3) bottleneck that otherwise dominates mesh
    # refinement studies, while representing exactly the same local space.
    K_u = K * u_cur
    M_u = M * u_cur
    K_block = sparse(K[idx_sub, idx_sub])
    M_block = sparse(M[idx_sub, idx_sub])
    K_cross = sparse(reshape(K_u[idx_sub], :, 1))
    M_cross = sparse(reshape(M_u[idx_sub], :, 1))
    K_local =
      [sparse(reshape([dot(u_cur, K_u)], 1, 1)) K_cross'; K_cross K_block]
    M_local =
      [sparse(reshape([dot(u_cur, M_u)], 1, 1)) M_cross'; M_cross M_block]
  else
    # Degenerate case: only current solution
    K_local = reshape([dot(u_cur, K * u_cur)], 1, 1)
    M_local = reshape([dot(u_cur, M * u_cur)], 1, 1)
  end

  # K_local = B' * K * B with B = [u_cur e(isd_1) e(isd_2) ...] where e(isd_1) are standard basis vectors in isd_1

  @debug "Size of K_local: $(size(K_local))"
  @debug "Size of M_local: $(size(M_local))"
  K_local_sym = Symmetric(K_local)
  M_local_sym = Symmetric(M_local)
  F = cholesky(K_local_sym)  # ≈ A^{-1} preconditioner
  coefficients = nothing
  iterations = 0
  converged = false
  local_residual = Inf
  try
    res = lobpcg(
      K_local_sym,
      M_local_sym,
      false,
      1;
      P = F,
      tol = 1e-8,
      maxiter = 500,
      log = collect_info,
    )
    coefficients = res.X[:, 1]
    iterations = Int(res.iterations)
    converged = Bool(res.converged)
    local_residual = Float64(res.residual_norms[1])
  catch error
    error isa PosDefException || rethrow()
    # Near convergence, the exterior column can become numerically dependent
    # on the active coordinates. LOBPCG then sees a singular reduced mass
    # matrix. Whiten its numerical range and solve the rank-revealed pencil.
    mass_decomposition = eigen(Symmetric(Matrix(M_local)))
    mass_scale = maximum(abs, mass_decomposition.values)
    keep = findall(>(1e-12 * mass_scale), mass_decomposition.values)
    isempty(keep) && rethrow()
    mass_basis = mass_decomposition.vectors[:, keep] *
      Diagonal(inv.(sqrt.(mass_decomposition.values[keep])))
    reduced_K = Symmetric(mass_basis' * Matrix(K_local) * mass_basis)
    reduced_decomposition = eigen(reduced_K, 1:1)
    coefficients = mass_basis * reduced_decomposition.vectors[:, 1]
    theta = reduced_decomposition.values[1]
    local_residual = norm(K_local * coefficients - theta .* (M_local * coefficients))
    iterations = 1
    converged = true
  end

  # Return the Ritz vector itself. Its normalization is immaterial to the
  # subsequent Rayleigh--Ritz combination and is performed there.
  x_new = reconstruct_implicit!(coefficients, u_cur, idx_sub)
  factor_nnz = if !collect_info
    -1
  elseif issparse(K_local)
    nnz(sparse(F.L))
  else
    localdim * (localdim + 1) ÷ 2
  end
  info = (
    dimension = localdim,
    iterations = iterations,
    converged = converged,
    residual = local_residual,
    k_nnz = collect_info ? (issparse(K_local) ? nnz(K_local) : localdim^2) : -1,
    factor_nnz = factor_nnz,
  )
  return (u = x_new, info = info)
end

function inf_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  return generalized_rayleigh_inf_step(e, u_cur, idx_sub).u
end

function inf_step_with_info(
  e::Energies.AbstractEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  u = inf_step(e, u_cur, idx_sub)
  return (
    u = u,
    info = (
      dimension = 1 + length(idx_sub),
      iterations = -1,
      converged = true,
      residual = NaN,
      k_nnz = -1,
      factor_nnz = -1,
    ),
  )
end


function inf_step_with_info(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  return generalized_rayleigh_inf_step(e, u_cur, idx_sub; collect_info)
end

function inf_step_with_info(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  result = nonlinear_enriched_local_minimize(e, u_cur, idx_sub)
  info = (
    dimension = 1 + length(idx_sub),
    iterations = result.iterations,
    converged = result.converged,
    residual = result.residual,
    energy_evaluations = result.energy_evaluations,
    k_nnz = -1,
    factor_nnz = -1,
  )
  return (u = result.u, info = info)
end

"""
    _gp_local_space(u_cur, idx_sub)

Basis of the local trial space `V_i + span{u_cur}` used by the Gross--Pitaevskii
local minimizations. The first column is the exterior part of `u_cur`, the
remaining columns are the active coordinate vectors.
"""
function _gp_local_space(u_cur::Vector{Float64}, idx_sub::AbstractVector)
  local_space = zeros(Float64, length(u_cur), 1 + length(idx_sub))
  local_space[:, 1] .= u_cur
  zero_out_local!(view(local_space, :, 1), idx_sub)
  for (k, j) in pairs(idx_sub)
    local_space[j, k+1] = 1.0
  end
  return local_space
end

function inf_step(
  e::Energies.GrossPitaevskiiRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  if iszero(e.beta)
    return inf_step(
      Energies.GeneralizedRayleighQuotient(e.K, e.M),
      u_cur,
      idx_sub,
    )
  end

  return _scf_subspace(e, _gp_local_space(u_cur, idx_sub); initial = u_cur)
end

function inf_step_with_info(
  e::Energies.GrossPitaevskiiRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  if iszero(e.beta)
    return inf_step_with_info(
      Energies.GeneralizedRayleighQuotient(e.K, e.M),
      u_cur,
      idx_sub;
      collect_info,
    )
  end

  local_space = _gp_local_space(u_cur, idx_sub)
  stats = Ref{Any}(nothing)
  u = _scf_subspace(e, local_space; initial = u_cur, info_ref = stats)
  return (
    u = u,
    info = (
      dimension = size(local_space, 2),
      iterations = stats[].iterations,
      converged = stats[].converged,
      residual = stats[].residual_norm,
      k_nnz = -1,
      factor_nnz = -1,
    ),
  )
end

function inf_step(
  model::Energies.GrossPitaevskiiProjectedNewtonModel{Float64},
  ::Vector{Float64},
  idx_sub::AbstractVector,
)
  return projected_newton_gp_step(model, idx_sub).u
end

function inf_step_with_info(
  model::Energies.GrossPitaevskiiProjectedNewtonModel{Float64},
  ::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool = false,
)
  return projected_newton_gp_step(model, idx_sub)
end

"""
Solve the local Riemannian Newton model in `P*V_i`, where
`P = I - u*(M*u)'`. The projected matrix is a rank-two update of the local
principal block of `H_lagrangian`. A Levenberg--Marquardt shift is increased
until the complete projected matrix is positive definite; its inertia is
checked through the two-by-two Woodbury Schur complement.

The returned vector is `u + w_i`, with `w_i` supported on the subdomain. Since
the combination space also contains `u`, this spans the same space as the
actual tangent direction `P*w_i` without introducing a dense global vector.
"""
function projected_newton_gp_step(
  model::Energies.GrossPitaevskiiProjectedNewtonModel{Float64},
  idx_sub::AbstractVector,
)
  active = collect(Int, idx_sub)
  isempty(active) && return (
    u = copy(model.u),
    info = (
      dimension = 0,
      iterations = 0,
      converged = true,
      residual = 0.0,
      k_nnz = 0,
      factor_nnz = 0,
      shift = 0.0,
      fallback = false,
    ),
  )

  u = model.u
  M = model.M
  H_lagrangian = model.H_lagrangian
  Mu = M * u
  Hu = H_lagrangian * u
  residual_local = model.residual[active]
  if norm(residual_local) <= eps(Float64)
    return (
      u = copy(u),
      info = (
        dimension = length(active),
        iterations = 0,
        converged = true,
        residual = norm(residual_local),
        k_nnz = 0,
        factor_nnz = 0,
        shift = 0.0,
        fallback = false,
      ),
    )
  end

  g = Mu[active]
  h = Hu[active]
  gamma = dot(u, Hu)
  H_diagonal = abs.(diag(H_lagrangian)[active])
  M_diagonal = abs.(diag(M)[active])
  scale =
    max(maximum(H_diagonal), eps(Float64)) /
    max(maximum(M_diagonal), eps(Float64))
  sigma = 0.0
  accepted = false
  coefficients = zeros(Float64, length(active))
  factor_nnz = 0
  projected_nnz = 0
  attempts = 0

  for attempt = 1:50
    attempts = attempt
    T = sparse(H_lagrangian[active, active] .+ sigma .* M[active, active])
    h_sigma = h .+ sigma .* g
    gamma_sigma = gamma + sigma
    local factor
    try
      factor = cholesky(Symmetric(T); check = true)
    catch error
      if error isa PosDefException ||
         error isa SingularException ||
         error isa ZeroPivotException
        sigma = iszero(sigma) ? 1e-8 * scale : 2sigma
        continue
      end
      rethrow()
    end

    U = hcat(g, h_sigma)
    solves = factor \ hcat(-residual_local, U)
    base_step = view(solves, :, 1)
    Tinv_U = view(solves, :, 2:3)
    C_inverse = [0.0 -1.0; -1.0 -gamma_sigma]
    S = Symmetric(C_inverse + U' * Tinv_U)
    determinant = det(S)
    determinant_scale = max(1.0, opnorm(S)^2)

    # With T positive definite, the rank-two update is positive definite iff
    # S has the same (one-positive, one-negative) inertia as C_inverse.
    if determinant < -100eps(Float64) * determinant_scale
      coefficients .= base_step .- Tinv_U * (S \ (U' * base_step))
      if all(isfinite, coefficients) && dot(residual_local, coefficients) < 0
        accepted = true
        factor_nnz = nnz(factor)
        projected_nnz = nnz(T) + 4length(active)
        break
      end
    end
    sigma = iszero(sigma) ? 1e-8 * scale : 2sigma
  end

  fallback = !accepted
  if fallback
    # The residual is a dual vector. Convert it to a primal direction with the
    # positive GP linear-energy metric instead of treating its coefficients as
    # a Euclidean vector.
    metric_local = sparse(model.metric[active, active])
    metric_factor = cholesky(Symmetric(metric_local); check = true)
    coefficients .= metric_factor \ (-residual_local)
    factor_nnz = nnz(metric_factor)
    projected_nnz = nnz(metric_local)
    sigma = Inf
  end

  candidate = copy(u)
  candidate[active] .+= coefficients
  linear_residual =
    norm(residual_local) == 0 ? 0.0 :
    abs(dot(residual_local, coefficients)) / norm(residual_local)
  return (
    u = candidate,
    info = (
      dimension = length(active),
      iterations = attempts,
      converged = dot(residual_local, coefficients) < 0,
      residual = linear_residual,
      k_nnz = projected_nnz,
      factor_nnz = factor_nnz,
      shift = sigma,
      fallback = fallback,
    ),
  )
end

function inf_step(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  return nonlinear_enriched_local_minimize(e, u_cur, idx_sub).u
end

"""
    nonlinear_enriched_local_minimize(e, u_cur, idx_sub; kwargs...)

Minimize a nonlinear energy over `V_i + span{u_cur}` with L-BFGS. The
coordinate basis consists of the exterior part of `u_cur` and the active
coordinate vectors, which spans the requested space without constructing a
dense global-by-local matrix. Only energy and gradient evaluations are used;
the Hessian stored by `e` is deliberately ignored.
"""
function nonlinear_enriched_local_minimize(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  relative_tolerance::Float64 = 1e-10,
  absolute_tolerance::Float64 = 1e-12,
  maxiter::Int = 200,
)
  active = collect(Int, idx_sub)
  isempty(active) && return (
    u = copy(u_cur),
    iterations = 0,
    converged = true,
    residual = 0.0,
    energy_evaluations = 0,
  )

  exterior = copy(u_cur)
  exterior[active] .= 0.0
  exterior_norm = norm(exterior)
  has_global_direction = !iszero(exterior_norm)
  global_direction = has_global_direction ? exterior ./ exterior_norm : exterior
  first_local_coordinate = has_global_direction ? 2 : 1
  initial_coordinates =
    has_global_direction ? vcat(exterior_norm, u_cur[active]) :
    copy(u_cur[active])

  function reconstruct(coordinates)
    u =
      has_global_direction ? coordinates[1] .* global_direction :
      zeros(Float64, length(u_cur))
    u[active] .= view(coordinates, first_local_coordinate:length(coordinates))
    return u
  end

  function project_gradient(gradient)
    local_gradient = gradient[active]
    return has_global_direction ?
           vcat(dot(global_direction, gradient), local_gradient) :
           local_gradient
  end

  initial_gradient = project_gradient(Energies.gradient(e, u_cur))
  target = max(absolute_tolerance, relative_tolerance * norm(initial_gradient))
  energy_evaluations = 0
  function objective(coordinates)
    energy_evaluations += 1
    return Energies.energy(e, reconstruct(coordinates))
  end
  function reduced_gradient!(storage, coordinates)
    storage .= project_gradient(Energies.gradient(e, reconstruct(coordinates)))
    return storage
  end

  result = Optim.optimize(
    objective,
    reduced_gradient!,
    initial_coordinates,
    Optim.LBFGS(),
    Optim.Options(
      iterations = maxiter,
      g_abstol = target,
      # A global FE energy can be unchanged to roundoff even though the
      # projected local gradient is not small. Let the line search globalize
      # the step, but do not mistake objective stagnation or a rounded increase
      # for convergence before the requested gradient tolerance is met.
      allow_f_increases = true,
      successive_f_tol = maxiter,
      show_warnings = false,
    ),
  )
  u = reconstruct(Optim.minimizer(result))
  residual = norm(project_gradient(Energies.gradient(e, u)))
  return (
    u = u,
    iterations = Optim.iterations(result),
    converged = residual <= target,
    residual = residual,
    energy_evaluations = energy_evaluations,
  )
end

"""Minimize a nonlinear energy with all exterior degrees of freedom fixed."""
function nonlinear_local_minimize(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  relative_tolerance::Float64 = 1e-10,
  absolute_tolerance::Float64 = 1e-12,
  maxiter::Int = 50,
)
  active = collect(Int, idx_sub)
  isempty(active) &&
    return (u = copy(u_cur), iterations = 0, energy_evaluations = 0)
  u = copy(u_cur)
  initial_gradient = Energies.gradient(e, u)[active]
  target = max(absolute_tolerance, relative_tolerance * norm(initial_gradient))
  energy_evaluations = 0

  # Analytic sparse Newton is used when available. The legacy generic energy
  # constructor remains supported through a reduced L-BFGS fallback.
  if isnothing(e.hess_assembler)
    z0 = copy(u[active])
    function objective(z)
      trial = copy(u_cur)
      trial[active] .= z
      energy_evaluations += 1
      return Energies.energy(e, trial)
    end
    function reduced_gradient!(storage, z)
      trial = copy(u_cur)
      trial[active] .= z
      storage .= Energies.gradient(e, trial)[active]
      return storage
    end
    result = Optim.optimize(
      objective,
      reduced_gradient!,
      z0,
      Optim.LBFGS(),
      Optim.Options(
        iterations = maxiter,
        g_abstol = target,
        allow_f_increases = false,
        show_warnings = false,
      ),
    )
    u[active] .= Optim.minimizer(result)
    return (
      u = u,
      iterations = Optim.iterations(result),
      energy_evaluations = energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    full_gradient = Energies.gradient(e, u)
    gradient = full_gradient[active]
    norm(gradient) <= target && return (
      u = u,
      iterations = iteration,
      energy_evaluations = energy_evaluations,
    )
    iteration == maxiter && break
    iterations_done = iteration + 1

    H = Energies.hessian(e, u)[active, active]
    step = try
      -(H \ gradient)
    catch
      -gradient
    end
    if !all(isfinite, step) || dot(gradient, step) >= 0
      step = -gradient
    end

    energy0 = Energies.energy(e, u)
    energy_evaluations += 1
    slope = dot(gradient, step)
    accepted = false
    step_length = 1.0
    trial_energy = energy0
    energy_resolution = 100 * eps(Float64) * max(1.0, abs(energy0))
    while step_length >= 2.0^-30
      trial = copy(u)
      trial[active] .+= step_length .* step
      trial_energy = Energies.energy(e, trial)
      energy_evaluations += 1
      if trial_energy <= energy0 + 1e-4 * step_length * slope
        u = trial
        accepted = true
        break
      end
      # Close to the minimizer, the predicted energy reduction can be below
      # the resolution of a Float64 energy evaluation even though the
      # projected derivative can still be reduced. In that regime, permit a
      # numerically non-increasing step only when it improves stationarity.
      if trial_energy <= energy0 + energy_resolution &&
         norm(Energies.gradient(e, trial)[active]) < norm(gradient)
        u = trial
        accepted = true
        break
      end
      step_length *= 0.5
    end
    accepted || break
    if step_length * norm(step) <= 1e-12 * (1 + norm(u[active]))
      break
    end
  end
  return (
    u = u,
    iterations = iterations_done,
    energy_evaluations = energy_evaluations,
  )
end

"""
  combine_step(e::Energies.AbstractEnergy{Float64}, sspace::Matrix{Float64})

Combine step function for variational domain decomposition methods.

This is an abstract interface that must be implemented by concrete energy types.
The function is intended to perform a combination step in the variational domain
decomposition algorithm, typically involving operations on the given subspace.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: Energy functional or operator
- `sspace::Matrix{Float64}`: Subspace matrix, typically containing basis vectors
  or coefficients for the current iteration

# Throws
- `ErrorException`: Always throws an error indicating that the method must be
  implemented for the specific energy type

# Notes
This is a fallback method that serves as a template. Concrete implementations
should override this method for specific energy types to provide the actual
combination step logic.
"""
function combine_step(
  e::Energies.AbstractEnergy{Float64},
  sspace::Matrix{Float64},
)
  throw(ErrorException("combine_step not implemented for $(typeof(e))"))
end

"""
  orthonormal_basis(X; rtol=max(size(X)...)*eps(eltype(X)))

Compute an orthonormal basis for the column space of `X` using a
column-pivoted, rank-revealing QR factorization. Dependent candidate vectors
must be removed before a combination step; otherwise the unused QR columns
can introduce arbitrary directions that are not in the intended trial space.
"""
function orthonormal_basis(
  X::AbstractMatrix{T};
  rtol::Real = max(size(X)...) * eps(T),
) where {T<:AbstractFloat}
  F = qr(X, ColumnNorm())
  diagonal = abs.(diag(F.R))
  isempty(diagonal) &&
    throw(ArgumentError("cannot form a basis from an empty matrix"))

  scale = maximum(diagonal)
  scale == zero(T) &&
    throw(ArgumentError("candidate matrix has zero column space"))
  rank = count(>(rtol * scale), diagonal)
  rank == 0 && throw(ArgumentError("candidate matrix has zero numerical rank"))

  return Matrix(F.Q)[:, 1:rank]
end

"""
    m_orthonormal_basis(X, M)

Return a basis `Q` for `range(X)` satisfying `Q' * M * Q = I`. The initial
Euclidean rank-revealing QR removes dependent candidates before the reduced
mass matrix is factored.
"""
function m_orthonormal_basis(X::AbstractMatrix, M::AbstractMatrix)
  B = orthonormal_basis(X)
  reduced_mass = Symmetric(B' * M * B)
  F = cholesky(reduced_mass)
  return B / F.U
end

"""
    anchored_m_orthonormal_basis(anchor, increments, M; rtol=1e-12)

Construct an M-orthonormal basis from a separately retained anchor and a set
of increments. Rank truncation is relative to the increments' own scale, so
small but independent corrections are not discarded merely because the anchor
has unit norm.
"""
function anchored_m_orthonormal_basis(
  anchor::AbstractVector,
  increments::AbstractMatrix,
  M::AbstractMatrix;
  rtol::Real = 1e-12,
)
  rtol > 0 || throw(ArgumentError("rtol must be positive"))
  q0 = copy(anchor)
  Energies.normalize_M!(q0, M)
  size(increments, 1) == length(q0) ||
    throw(DimensionMismatch("increment rows must match the anchor length"))
  isempty(increments) && return reshape(q0, :, 1)

  W = Matrix(increments)
  W .-= q0 * (q0' * (M * W))
  gram = Symmetric(W' * (M * W))
  decomposition = eigen(gram)
  largest = maximum(decomposition.values)
  largest <= 0 && return reshape(q0, :, 1)
  keep = findall(>(rtol * largest), decomposition.values)
  isempty(keep) && return reshape(q0, :, 1)
  Q1 =
    W * (
      decomposition.vectors[:, keep] *
      Diagonal(inv.(sqrt.(decomposition.values[keep])))
    )
  return hcat(q0, Q1)
end

"""
    _scf_subspace(e, X; initial=X[:, 1], maxiter=200, tol=1e-9)

Solve the Gross--Pitaevskii nonlinear eigenproblem in `range(X)` by a damped
self-consistent-field iteration. The basis is M-orthonormal, so every SCF
iterate is normalized explicitly. Each update uses the ground state of the
Hamiltonian frozen at the current density and backtracks along that SCF
direction only when needed to retain energy monotonicity.
"""
function _scf_subspace(
  e::Energies.GrossPitaevskiiRayleighQuotient{Float64},
  X::AbstractMatrix{Float64};
  initial::AbstractVector{Float64} = X[:, 1],
  maxiter::Int = 200,
  tol::Float64 = 1e-9,
  info_ref::Union{Nothing,Ref} = nothing,
)
  maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
  tol > 0 || throw(ArgumentError("tol must be positive"))

  Q = m_orthonormal_basis(X, e.M)
  alpha = Q' * (e.M * initial)
  if norm(alpha) <= 100 * eps(Float64)
    alpha = zeros(size(Q, 2))
    alpha[1] = 1.0
  else
    alpha ./= norm(alpha)
  end

  K_reduced = Symmetric(Q' * e.K * Q)

  function reduced_density(alpha)
    u = Q * alpha
    if !isnothing(e.density_matrix)
      return Symmetric(Q' * e.density_matrix(u) * Q)
    end

    # If C(x) = T(x,x,x), polarization gives
    # T(u,u,v) = (C(u+v) - C(u-v) - 2C(v)) / 6. Thus the
    # frozen-density action can be recovered from the legacy cubic callback.
    C_u_plus = Vector{Float64}(undef, size(Q, 1))
    C_u_minus = similar(C_u_plus)
    density_Q = similar(Q)
    for j in axes(Q, 2)
      q = view(Q, :, j)
      C_u_plus .= e.cubic_gradient(u .+ q)
      C_u_minus .= e.cubic_gradient(u .- q)
      view(density_Q, :, j) .=
        (C_u_plus .- C_u_minus .- 2 .* e.cubic_gradient(q)) ./ 6
    end
    return Symmetric(Q' * density_Q)
  end

  function state_energy(alpha)
    return Energies.energy(e, Q * alpha)
  end

  converged = false
  residual_norm = Inf
  iterations = 0
  for iteration = 0:maxiter
    H = Symmetric(K_reduced + e.beta .* reduced_density(alpha))
    Halpha = H * alpha
    residual = Halpha .- dot(alpha, Halpha) .* alpha
    residual_norm = norm(residual)
    if residual_norm <= tol
      converged = true
      iterations = iteration
      break
    end
    iteration == maxiter && (iterations = maxiter; break)

    next_alpha = eigen(H, 1:1).vectors[:, 1]
    dot(alpha, next_alpha) < 0 && (next_alpha .*= -1)
    direction = next_alpha .- dot(alpha, next_alpha) .* alpha
    norm(direction) > eps(Float64) || (iterations = iteration; break)

    energy0 = state_energy(alpha)
    slope = 2 * dot(residual, direction)
    step = 1.0
    accepted = false
    while step >= 2.0^-40
      trial = alpha .+ step .* direction
      trial ./= norm(trial)
      trial_energy = state_energy(trial)
      if trial_energy <=
         energy0 +
         1e-4 * step * min(slope, 0.0) +
         10eps(Float64) * max(1.0, abs(energy0))
        alpha = trial
        accepted = true
        break
      end
      step /= 2
    end
    iterations = iteration + 1
    accepted || break
  end

  u = Q * alpha
  isnothing(info_ref) || (
    info_ref[] = (
      iterations = iterations,
      converged = converged,
      residual_norm = residual_norm,
    )
  )

  # Fix the arbitrary sign for stable histories and post-processing.
  if dot(initial, e.M * u) < 0
    u .*= -1
  end
  Energies.normalize_M!(u, e.M)
  return u
end

function combine_step(
  e::Energies.QuadraticEnergy{Float64},
  sspace::Matrix{Float64},
)
  # TODO: Is combine step the same as inf step in general?
  # => Just 1st order optimality in subspace?
  A, b = e.A, e.b

  B = orthonormal_basis(sspace)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_A = Matrix{Float64}(undef, N, localdim)
  A_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_A, A, B)
  mul!(A_local, B', temp_A)  # => A_local = B' A B

  b_local = B' * b  # => b_local = B' b

  # Solve the linear system A_local α = b_local to minimize the quadratic energy
  α_new = A_local \ b_local
  x_new = B * α_new
  return x_new
end

"""
  combine_step(e::Energies.GeneralizedRayleighQuotient, u_collection::Matrix)

Perform a Rayleigh-Ritz procedure in the span of the columns of sspace.
The output is re-normalized in the M-norm.
"""
function combine_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  sspace::Matrix{Float64},
)
  K, M = e.A, e.B

  B = orthonormal_basis(sspace)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_K = Matrix{Float64}(undef, N, localdim)
  temp_M = Matrix{Float64}(undef, N, localdim)
  K_local = Matrix{Float64}(undef, localdim, localdim)
  M_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_K, K, B)
  mul!(temp_M, M, B)
  mul!(K_local, B', temp_K)  # => K_local = B' K B
  mul!(M_local, B', temp_M)  # => M_local = B' M B

  _, eigvecs = eigen(K_local, M_local)
  x_new = B * eigvecs[:, 1]  # a[1] * u + a[2] * v1
  Energies.normalize_M!(x_new, M)
  return x_new
end

function combine_step(
  e::Energies.GrossPitaevskiiRayleighQuotient{Float64},
  sspace::Matrix{Float64};
  initial::AbstractVector{Float64} = view(sspace, :, 1),
  maxiter::Int = 200,
  tol::Float64 = 1e-9,
)
  if iszero(e.beta)
    return combine_step(Energies.GeneralizedRayleighQuotient(e.K, e.M), sspace)
  end
  return _scf_subspace(
    e,
    sspace;
    initial = initial,
    maxiter = maxiter,
    tol = tol,
  )
end

"Minimize a nonlinear energy over the linear span of solution candidates."
function combine_step(
  e::Energies.NonlinearEnergy{Float64},
  sspace::Matrix{Float64},
)
  return minimize_linear_subspace(e, sspace; initial = sspace[:, 1]).u
end

"""
    minimize_linear_subspace(e, candidates; initial=candidates[:, 1])

Minimize a nonlinear energy in `span(candidates)`. This is the nonlinear
source-problem implementation of the same candidate-space `combine_step`
interface used by quadratic energies and Rayleigh quotients.
"""
function minimize_linear_subspace(
  e::Energies.NonlinearEnergy{Float64},
  candidates::Matrix{Float64};
  initial::AbstractVector{Float64} = view(candidates, :, 1),
  relative_tolerance::Float64 = 1e-10,
  absolute_tolerance::Float64 = 1e-12,
  maxiter::Int = 30,
)
  B = orthonormal_basis(candidates)
  alpha = B' * initial
  initial_gradient = B' * Energies.gradient(e, B * alpha)
  target = max(absolute_tolerance, relative_tolerance * norm(initial_gradient))
  energy_evaluations = 0

  if isnothing(e.hess_assembler)
    function objective(coefficients)
      energy_evaluations += 1
      return Energies.energy(e, B * coefficients)
    end
    function reduced_gradient!(storage, coefficients)
      storage .= B' * Energies.gradient(e, B * coefficients)
      return storage
    end
    result = Optim.optimize(
      objective,
      reduced_gradient!,
      alpha,
      Optim.LBFGS(),
      Optim.Options(
        iterations = maxiter,
        g_abstol = target,
        allow_f_increases = false,
        show_warnings = false,
      ),
    )
    return (
      u = B * Optim.minimizer(result),
      iterations = Optim.iterations(result),
      energy_evaluations = energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    u = B * alpha
    gradient = B' * Energies.gradient(e, u)
    norm(gradient) <= target && return (
      u = u,
      iterations = iteration,
      energy_evaluations = energy_evaluations,
    )
    iteration == maxiter && break
    iterations_done = iteration + 1
    reduced_hessian = Symmetric(B' * Energies.hessian(e, u) * B)
    step = try
      -(reduced_hessian \ gradient)
    catch
      -gradient
    end
    if !all(isfinite, step) || dot(gradient, step) >= 0
      step = -gradient
    end

    energy0 = Energies.energy(e, u)
    energy_evaluations += 1
    slope = dot(gradient, step)
    accepted = false
    step_length = 1.0
    energy_resolution = 100 * eps(Float64) * max(1.0, abs(energy0))
    while step_length >= 2.0^-30
      trial_alpha = alpha .+ step_length .* step
      trial_u = B * trial_alpha
      trial_energy = Energies.energy(e, trial_u)
      energy_evaluations += 1
      if trial_energy <= energy0 + 1e-4 * step_length * slope || (
        trial_energy <= energy0 + energy_resolution &&
        norm(B' * Energies.gradient(e, trial_u)) < norm(gradient)
      )
        alpha = trial_alpha
        accepted = true
        break
      end
      step_length *= 0.5
    end
    accepted || break
    step_length * norm(step) <= 1e-12 * (1 + norm(alpha)) && break
  end
  return (
    u = B * alpha,
    iterations = iterations_done,
    energy_evaluations = energy_evaluations,
  )
end

"""
    minimize_affine_corrections(e, anchor, directions)

Minimize `e(anchor + directions*alpha)` in the affine correction space. This
general reduced optimizer is used by study-local direction methods such as
optimally damped AS/RAS; varDD itself uses `combine_step` on solution
candidates.
"""
function minimize_affine_corrections(
  e::Energies.NonlinearEnergy{Float64},
  anchor::Vector{Float64},
  directions::Matrix{Float64};
  relative_tolerance::Float64 = 1e-10,
  absolute_tolerance::Float64 = 1e-12,
  maxiter::Int = 30,
)
  size(directions, 2) == 0 &&
    return (u = copy(anchor), iterations = 0, energy_evaluations = 0)
  B = try
    orthonormal_basis(directions)
  catch error
    error isa ArgumentError || rethrow()
    return (u = copy(anchor), iterations = 0, energy_evaluations = 0)
  end
  alpha = zeros(size(B, 2))
  initial_gradient = B' * Energies.gradient(e, anchor)
  target = max(absolute_tolerance, relative_tolerance * norm(initial_gradient))
  energy_evaluations = 0

  if isnothing(e.hess_assembler)
    function objective(coefficients)
      energy_evaluations += 1
      return Energies.energy(e, anchor + B * coefficients)
    end
    function reduced_gradient!(storage, coefficients)
      storage .= B' * Energies.gradient(e, anchor + B * coefficients)
      return storage
    end
    result = Optim.optimize(
      objective,
      reduced_gradient!,
      alpha,
      Optim.LBFGS(),
      Optim.Options(
        iterations = maxiter,
        g_abstol = target,
        allow_f_increases = false,
        show_warnings = false,
      ),
    )
    return (
      u = anchor + B * Optim.minimizer(result),
      iterations = Optim.iterations(result),
      energy_evaluations = energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    u = anchor + B * alpha
    gradient = B' * Energies.gradient(e, u)
    norm(gradient) <= target && return (
      u = u,
      iterations = iteration,
      energy_evaluations = energy_evaluations,
    )
    iteration == maxiter && break
    iterations_done = iteration + 1
    reduced_hessian = Symmetric(B' * Energies.hessian(e, u) * B)
    step = try
      -(reduced_hessian \ gradient)
    catch
      -gradient
    end
    if !all(isfinite, step) || dot(gradient, step) >= 0
      step = -gradient
    end

    energy0 = Energies.energy(e, u)
    energy_evaluations += 1
    slope = dot(gradient, step)
    accepted = false
    step_length = 1.0
    trial_energy = energy0
    energy_resolution = 100 * eps(Float64) * max(1.0, abs(energy0))
    while step_length >= 2.0^-30
      trial_alpha = alpha .+ step_length .* step
      trial_u = anchor + B * trial_alpha
      trial_energy = Energies.energy(e, trial_u)
      energy_evaluations += 1
      if trial_energy <= energy0 + 1e-4 * step_length * slope
        alpha = trial_alpha
        accepted = true
        break
      end
      if trial_energy <= energy0 + energy_resolution &&
         norm(B' * Energies.gradient(e, trial_u)) < norm(gradient)
        alpha = trial_alpha
        accepted = true
        break
      end
      step_length *= 0.5
    end
    accepted || break
    if step_length * norm(step) <= 1e-12 * (1 + norm(alpha))
      break
    end
  end
  return (
    u = anchor + B * alpha,
    iterations = iterations_done,
    energy_evaluations = energy_evaluations,
  )
end

function zero_out_local!(u_global::AbstractVector, idx_sub::AbstractVector)
  for j in idx_sub
    u_global[j] = 0.0
  end
  return u_global
end

"""
  reconstruct_implicit!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)

Reconstruct x_new from implicit basis [u_cur, e_j1, e_j2, ...] where e_j are standard
basis vectors, without normalization. This avoids explicitly constructing the basis matrix.
"""
function reconstruct_implicit!(
  α::Vector{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  # The following is equivalent to x_new = [u_cur e_j1 e_j2 ...] * α
  # but avoids constructing e_j explicitly
  x_new = α[1] * u_cur  # Coefficient for current solution
  # Add contributions from standard basis vectors
  for (k, j) in pairs(idx_sub)
    x_new[j] += α[k+1]  # Add coefficient for e_j
  end
  return x_new
end

"""
    VarDDResult

Result of [`var_dd`](@ref).

# Fields
- `u::Vector{Float64}`: final iterate
- `energy::Float64`: energy of `u`
- `energy_history::Vector{Float64}`: energies of `u_0, ..., u_k`, so its length
  is `iterations + 1`
- `iterate_history::Vector{Vector{Float64}}`: the iterates `u_0, ..., u_k`
- `residual_history::Vector{Float64}`: residual norms of `u_1, ..., u_k`, so
  its length is `iterations`
- `local_update_history::Union{Nothing,Vector{Vector{Vector{Float64}}}}`: for
  each sweep, the local increments `u_i - u_cur` of every subdomain. This is
  `nothing` unless `var_dd` was called with `save_local_updates=true`
- `converged::Bool`: whether the residual tolerance was reached
- `iterations::Int`: number of outer sweeps performed
"""
struct VarDDResult
  u::Vector{Float64}
  energy::Float64
  energy_history::Vector{Float64}
  iterate_history::Vector{Vector{Float64}}
  residual_history::Vector{Float64}
  local_update_history::Union{Nothing,Vector{Vector{Vector{Float64}}}}
  converged::Bool
  iterations::Int
end

function Base.show(io::IO, result::VarDDResult)
  status = result.converged ? "converged" : "not converged"
  @printf(
    io,
    "VarDDResult(%s after %d iterations, energy = %.6e, residual = %.6e)",
    status,
    result.iterations,
    result.energy,
    isempty(result.residual_history) ? NaN : last(result.residual_history),
  )
  return nothing
end

"""
    var_dd(e, subdomain_dofs; maxiter=50, tol=1e-8, ...) -> VarDDResult

Variational domain decomposition algorithm for solving various energy minimization problems.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: Energy functional to minimize
- `subdomain_dofs::Vector{Vector{Int32}}`: DOF indices for each subdomain

# Keyword Arguments
- `maxiter::Int=50`: Maximum number of iterations
- `tol::Float64=1e-8`: Convergence tolerance based on residual norm
- `save_local_updates::Bool=false`: Whether to save and output local updates from each subdomain
- `u0::Union{Nothing,Vector{Float64}}=nothing`: Initial guess (defaults to all-ones);
  useful for warm starts, e.g. across time steps of a gradient flow
- `history_depth::Int=0`: Number of previous global iterates added to the
  second-level trial space. `history_depth=1` adds `u_{k-1}` alongside `u_k`.
- `quadratic_model::Bool=false`: Build one quadratic Taylor model of `e` at
  the start of each outer sweep and use it for the local subdomain solves. The
  second-level combination, convergence test, and histories use the original
  energy. This requires an energy Hessian.
- `projected_gp_model::Bool=false`: Use the projected Riemannian quadratic
  local model for the Gross--Pitaevskii experiment.
- `subspace_callback=nothing`: Optional study/diagnostic hook called as
  `subspace_callback(iteration, combined_matrix)` immediately before the
  second-level minimization. It does not alter the algorithm.
- `local_solve_callback=nothing`: Optional diagnostic hook called as
  `local_solve_callback(iteration, subdomain, info)` after each local solve.
  For generalized Rayleigh quotients, `info` records the augmented-pencil
  dimension, LOBPCG iterations, convergence, and sparse factor sizes. For
  Gross--Pitaevskii energies it records the reduced dimension, SCF iterations,
  convergence, and projected nonlinear-eigenproblem residual. For other
  nonlinear energies it records the reduced dimension, L-BFGS iterations,
  convergence, projected residual, and energy evaluations.
- `verbose::Bool=true`: Print per-iteration convergence information

# Returns
A [`VarDDResult`](@ref) holding the final iterate, its energy, the energy,
iterate and residual histories, and the convergence status.
"""
function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{<:AbstractVector{<:Integer}};
  maxiter::Int = 50,
  tol::Float64 = 1e-8,
  save_local_updates::Bool = false,
  u0::Union{Nothing,Vector{Float64}} = nothing,
  history_depth::Int = 0,
  quadratic_model::Bool = false,
  projected_gp_model::Bool = false,
  subspace_callback = nothing,
  local_solve_callback = nothing,
  verbose::Bool = true,
)
  history_depth >= 0 ||
    throw(ArgumentError("history_depth must be nonnegative"))
  quadratic_model + projected_gp_model <= 1 || throw(
    ArgumentError(
      "quadratic_model and projected_gp_model are mutually exclusive",
    ),
  )
  projected_gp_model &&
    !(e isa Energies.GrossPitaevskiiRayleighQuotient) &&
    throw(
      ArgumentError(
        "projected_gp_model is only available for GrossPitaevskiiRayleighQuotient",
      ),
    )

  # Initial guess, no need to normalize apparently
  u_cur = isnothing(u0) ? ones(Energies.dimension(e)) : copy(u0)
  projected_gp_model && Energies.normalize_M!(u_cur, e.M)

  e_hist = Float64[]
  sol_hist = Vector{Vector{Float64}}()
  resnorm_hist = Float64[]
  local_update_hist = Vector{Vector{Vector{Float64}}}()
  previous_iterates = Vector{Vector{Float64}}()

  e_cur = e(u_cur)
  push!(e_hist, e_cur)
  push!(sol_hist, copy(u_cur))

  m = length(subdomain_dofs)
  local_updates = zeros(size(u_cur, 1), m)  # preallocate for efficiency
  for n = 1:maxiter
    sweep_energy = if quadratic_model
      Energies.quadratic_model(e, u_cur)
    elseif projected_gp_model
      Energies.projected_newton_model(e, u_cur)
    else
      e
    end
    current_local_updates = Vector{Vector{Float64}}()
    for i = 1:m
      local_result = inf_step_with_info(
        sweep_energy,
        u_cur,
        subdomain_dofs[i];
        collect_info = !isnothing(local_solve_callback),
      )
      u_next_i = local_result.u
      !isnothing(local_solve_callback) &&
        local_solve_callback(n, i, local_result.info)
      local_updates[:, i] = u_next_i

      if save_local_updates
        push!(current_local_updates, copy(u_next_i .- u_cur))
      end
    end

    if save_local_updates
      push!(local_update_hist, current_local_updates)
    end

    combination_anchor = u_cur
    combined_matrix = if projected_gp_model
      history_increments =
        [iterate .- combination_anchor for iterate in previous_iterates]
      local_increments = local_updates .- combination_anchor
      increments = hcat(history_increments..., local_increments)
      anchored_m_orthonormal_basis(combination_anchor, increments, e.M)
    else
      hcat(combination_anchor, previous_iterates..., local_updates)
    end
    !isnothing(subspace_callback) && subspace_callback(n, combined_matrix)
    u_trial = combine_step(e, combined_matrix)
    if projected_gp_model
      energy_tolerance = 10eps(Float64) * max(1.0, abs(e_cur))
      if e(u_trial) > e_cur + energy_tolerance
        u_trial = copy(u_cur)
      end
    end
    u_new = u_trial

    resnorm = Energies.residual_norm(e, u_new)
    push!(resnorm_hist, resnorm)
    verbose && @printf(
      "Iteration %3d: Residual norm ≈ %12.6e energy = %12.6e\n",
      n,
      resnorm,
      e(u_new)
    )

    e_new = e(u_new)
    push!(e_hist, e_new)
    push!(sol_hist, copy(u_new))
    if resnorm < tol
      verbose && println("Converged at iteration $n with energy e = $e_new")
      return VarDDResult(
        u_new,
        e_new,
        e_hist,
        sol_hist,
        resnorm_hist,
        save_local_updates ? local_update_hist : nothing,
        true,
        n,
      )
    end

    if history_depth > 0
      push!(previous_iterates, copy(u_cur))
      length(previous_iterates) > history_depth && popfirst!(previous_iterates)
    end
    u_cur = u_new
    e_cur = e_new
  end

  @warn "Reached maxiter=$maxiter with energy ≈ $e_cur"
  return VarDDResult(
    u_cur,
    e_cur,
    e_hist,
    sol_hist,
    resnorm_hist,
    save_local_updates ? local_update_hist : nothing,
    false,
    maxiter,
  )
end

end # module
