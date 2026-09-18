module Solvers

using LinearAlgebra
using SparseArrays
using FiniteDiff
using Gridap
using GridapDistributed
using Metis
using IterativeSolvers
using Arpack
using Printf
using Random
using LineSearches
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
Apply one local step of the original multiplicative quadratic-energy sweep.

The local Ritz vector is rescaled to retain coefficient one in front of the
incoming iterate, then used immediately as the input to the next subdomain.
This is deliberately separate from `inf_step`: it reproduces the historical
serial algorithm without reintroducing mutation into the additive path.
"""
function multiplicative_inf_step(
  e::Energies.QuadraticEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  α = quadratic_local_coefficients(e, u_cur, idx_sub)
  iszero(α[1]) && throw(ArgumentError(
    "multiplicative projective update is undefined because its current-iterate coefficient is zero",
  ))
  u_new = copy(u_cur)
  for (k, j) in pairs(idx_sub)
    u_new[j] += α[k+1] / α[1]
  end
  return u_new
end

function multiplicative_inf_step(
  e::Energies.AbstractEnergy{Float64},
  ::Vector{Float64},
  ::AbstractVector,
)
  throw(ArgumentError(
    "sweep=:multiplicative is currently implemented only for QuadraticEnergy, not $(typeof(e))",
  ))
end

function inf_step(
  e::Energies.LinearRegressionEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  localdim = 1 + length(idx_sub)

  A, b = e.A, e.b

  # For linear regression energy ||Ax - b||², the optimal solution in any subspace
  # is found by solving the normal equations for that subspace
  u_cur = copy(u_cur)
  zero_out_local!(u_cur, idx_sub)

  if length(idx_sub) > 0
    # Build local system: minimize ||A*[u_cur, e_j1, e_j2, ...]*α - b||²
    # This gives normal equations: (A'A)_local α = (A'b)_local

    A_local = zeros(localdim, localdim)
    b_local = zeros(localdim)

    # Compute A*u_cur once for efficiency
    A_u_cur = A * u_cur
    AtA_u_cur = A' * A_u_cur  # A'A * u_cur
    Atb = A' * b              # A' * b

    # First entry: (A*u_cur)' * (A*u_cur) = u_cur' * A' * A * u_cur
    A_local[1, 1] = dot(A_u_cur, A_u_cur)
    b_local[1] = dot(A_u_cur, b)  # (A*u_cur)' * b

    # Cross terms: u_cur' * A' * A * e_j = (A'A * u_cur)[j]
    for (k, j) in pairs(idx_sub)
      A_local[1, k+1] = AtA_u_cur[j]  # u_cur' * A' * A * e_j
      A_local[k+1, 1] = AtA_u_cur[j]  # e_j' * A' * A * u_cur (symmetric)
      b_local[k+1] = Atb[j]           # e_j' * A' * b = (A' * b)[j]
    end

    # Diagonal entries: e_i' * A' * A * e_j = (A' * A)[i,j]
    AtA = A' * A
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        A_local[k1+1, k2+1] = AtA[j1, j2]
      end
    end
  else
    # Degenerate case: only current solution
    A_u_cur = A * u_cur
    A_local = reshape([dot(A_u_cur, A_u_cur)], 1, 1)
    b_local = reshape([dot(A_u_cur, b)], 1)
  end

  # Solve normal equations: A_local * α = b_local
  α_new = A_local \ b_local

  # Reconstruct the actual minimizer in the local trial subspace.
  x_new = reconstruct_implicit!(α_new, u_cur, idx_sub)
  return x_new
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
  collect_info::Bool=false,
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
    K_local = [sparse(reshape([dot(u_cur, K_u)], 1, 1)) K_cross'; K_cross K_block]
    M_local = [sparse(reshape([dot(u_cur, M_u)], 1, 1)) M_cross'; M_cross M_block]
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
  res = lobpcg(
    K_local_sym,
    M_local_sym,
    false,
    1;
    P=F,
    tol=1e-8,
    maxiter=500,
    log=collect_info,
  )

  # Return the Ritz vector itself. Its normalization is immaterial to the
  # subsequent Rayleigh--Ritz combination and is performed there.
  x_new = reconstruct_implicit!(res.X[:, 1], u_cur, idx_sub)
  factor_nnz = if !collect_info
    -1
  elseif issparse(K_local)
    nnz(sparse(F.L))
  else
    localdim * (localdim + 1) ÷ 2
  end
  info = (
    dimension=localdim,
    iterations=Int(res.iterations),
    converged=Bool(res.converged),
    residual=Float64(res.residual_norms[1]),
    k_nnz=collect_info ? (issparse(K_local) ? nnz(K_local) : localdim^2) : -1,
    factor_nnz=factor_nnz,
  )
  return (u=x_new, info=info)
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
  collect_info::Bool=false,
)
  u = inf_step(e, u_cur, idx_sub)
  return (
    u=u,
    info=(
      dimension=1 + length(idx_sub),
      iterations=-1,
      converged=true,
      residual=NaN,
      k_nnz=-1,
      factor_nnz=-1,
    ),
  )
end


function inf_step_with_info(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool=false,
)
  return generalized_rayleigh_inf_step(
    e,
    u_cur,
    idx_sub;
    collect_info,
  )
end

function inf_step_with_info(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool=false,
)
  result = nonlinear_enriched_local_minimize(e, u_cur, idx_sub)
  info = (
    dimension=1 + length(idx_sub),
    iterations=result.iterations,
    converged=result.converged,
    residual=result.residual,
    energy_evaluations=result.energy_evaluations,
    k_nnz=-1,
    factor_nnz=-1,
  )
  return (u=result.u, info=info)
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
  collect_info::Bool=false,
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
  u = _scf_subspace(e, local_space; initial=u_cur, info_ref=stats)
  return (
    u=u,
    info=(
      dimension=size(local_space, 2),
      iterations=stats[].iterations,
      converged=stats[].converged,
      residual=stats[].residual_norm,
      k_nnz=-1,
      factor_nnz=-1,
    ),
  )
end

function inf_step(
  model::Energies.GrossPitaevskiiTangentQuadraticModel{Float64},
  ::Vector{Float64},
  idx_sub::AbstractVector,
)
  return tangent_quadratic_gp_step(model, idx_sub).u
end

function inf_step_with_info(
  model::Energies.GrossPitaevskiiTangentQuadraticModel{Float64},
  ::Vector{Float64},
  idx_sub::AbstractVector;
  collect_info::Bool=false,
)
  return tangent_quadratic_gp_step(model, idx_sub)
end

"""
Solve one local tangent quadratic GP model by a sparse KKT system in
`span{u, e_j : j in idx_sub}`. The last KKT equation enforces the linearized
mass constraint, and the resulting candidate is retracted to the mass sphere.
"""
function tangent_quadratic_gp_step(
  model::Energies.GrossPitaevskiiTangentQuadraticModel{Float64},
  idx_sub::AbstractVector,
)
  active = collect(Int, idx_sub)
  isempty(active) && return (
    u=copy(model.u),
    info=(
      dimension=0,
      iterations=0,
      converged=true,
      residual=0.0,
      k_nnz=0,
      factor_nnz=0,
    ),
  )

  u = model.u
  H = model.H
  Mu = model.M * u
  Hu = H * u
  local_dimension = 1 + length(active)
  reduced_hessian = spzeros(Float64, local_dimension, local_dimension)
  reduced_hessian[1, 1] = dot(u, Hu)
  reduced_hessian[1, 2:end] .= Hu[active]
  reduced_hessian[2:end, 1] .= Hu[active]
  reduced_hessian[2:end, 2:end] = sparse(H[active, active])
  reduced_gradient = vcat(dot(u, model.residual), model.residual[active])
  tangent_constraint = vcat(dot(u, Mu), Mu[active])

  kkt = [
    reduced_hessian sparse(reshape(tangent_constraint, :, 1));
    sparse(reshape(tangent_constraint, 1, :)) spzeros(1, 1)
  ]
  rhs = vcat(-reduced_gradient, 0.0)
  factor = lu(kkt)
  solution = factor \ rhs
  coefficients = view(solution, 1:local_dimension)
  all(isfinite, coefficients) || throw(ErrorException(
    "local tangent quadratic GP solve produced non-finite coefficients",
  ))

  candidate = (1 + coefficients[1]) .* u
  candidate[active] .+= view(coefficients, 2:local_dimension)
  Energies.normalize_M!(candidate, model.M)
  dot(candidate, model.M * u) < 0 && (candidate .*= -1)
  kkt_residual = norm(kkt * solution - rhs)
  return (
    u=candidate,
    info=(
      dimension=local_dimension - 1,
      iterations=1,
      converged=kkt_residual <= 1e-9 * max(1.0, norm(rhs)),
      residual=kkt_residual,
      k_nnz=nnz(kkt),
      factor_nnz=nnz(factor.L) + nnz(factor.U),
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
  relative_tolerance::Float64=1e-10,
  absolute_tolerance::Float64=1e-12,
  maxiter::Int=200,
)
  active = collect(Int, idx_sub)
  isempty(active) && return (
    u=copy(u_cur),
    iterations=0,
    converged=true,
    residual=0.0,
    energy_evaluations=0,
  )

  exterior = copy(u_cur)
  exterior[active] .= 0.0
  exterior_norm = norm(exterior)
  has_global_direction = !iszero(exterior_norm)
  global_direction = has_global_direction ? exterior ./ exterior_norm : exterior
  first_local_coordinate = has_global_direction ? 2 : 1
  initial_coordinates = has_global_direction ?
                        vcat(exterior_norm, u_cur[active]) :
                        copy(u_cur[active])

  function reconstruct(coordinates)
    u = has_global_direction ?
        coordinates[1] .* global_direction :
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
      iterations=maxiter,
      g_abstol=target,
      # A global FE energy can be unchanged to roundoff even though the
      # projected local gradient is not small. Let the line search globalize
      # the step, but do not mistake objective stagnation or a rounded increase
      # for convergence before the requested gradient tolerance is met.
      allow_f_increases=true,
      successive_f_tol=maxiter,
      show_warnings=false,
    ),
  )
  u = reconstruct(Optim.minimizer(result))
  residual = norm(project_gradient(Energies.gradient(e, u)))
  return (
    u=u,
    iterations=Optim.iterations(result),
    converged=residual <= target,
    residual=residual,
    energy_evaluations=energy_evaluations,
  )
end

"""Minimize a nonlinear energy with all exterior degrees of freedom fixed."""
function nonlinear_local_minimize(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector;
  relative_tolerance::Float64=1e-10,
  absolute_tolerance::Float64=1e-12,
  maxiter::Int=50,
)
  active = collect(Int, idx_sub)
  isempty(active) && return (u=copy(u_cur), iterations=0, energy_evaluations=0)
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
        iterations=maxiter,
        g_abstol=target,
        allow_f_increases=false,
        show_warnings=false,
      ),
    )
    u[active] .= Optim.minimizer(result)
    return (
      u=u,
      iterations=Optim.iterations(result),
      energy_evaluations=energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    full_gradient = Energies.gradient(e, u)
    gradient = full_gradient[active]
    norm(gradient) <= target && return (
      u=u, iterations=iteration, energy_evaluations=energy_evaluations
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
    u=u,
    iterations=iterations_done,
    energy_evaluations=energy_evaluations,
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
  for iteration in 0:maxiter
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
      if trial_energy <= energy0 + 1e-4 * step * min(slope, 0.0) +
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
  isnothing(info_ref) || (info_ref[] = (
    iterations=iterations,
    converged=converged,
    residual_norm=residual_norm,
  ))

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
    return combine_step(
      Energies.GeneralizedRayleighQuotient(e.K, e.M),
      sspace,
    )
  end
  return _scf_subspace(
    e,
    sspace;
    initial = initial,
    maxiter = maxiter,
    tol = tol,
  )
end

function combine_step(
  e::Energies.LinearRegressionEnergy{Float64},
  sspace::Matrix{Float64},
)
  # For linear regression, the combine step solves the least squares problem
  # in the subspace spanned by the columns of sspace: min ||A*B*α - b||²
  # where B = QR factorization of sspace

  A, b = e.A, e.b

  B = orthonormal_basis(sspace)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_A = Matrix{Float64}(undef, size(A, 1), localdim)
  A_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_A, A, B)        # temp_A = A * B
  mul!(A_local, temp_A', temp_A)  # A_local = B' * A' * A * B = (A*B)' * (A*B)

  b_local = temp_A' * b     # b_local = B' * A' * b = (A*B)' * b

  # Solve the normal equations: A_local α = b_local
  α_new = A_local \ b_local
  x_new = B * α_new
  return x_new
end

"Minimize a nonlinear energy over the linear span of solution candidates."
function combine_step(
  e::Energies.NonlinearEnergy{Float64},
  sspace::Matrix{Float64},
)
  return minimize_linear_subspace(e, sspace; initial=sspace[:, 1]).u
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
  initial::AbstractVector{Float64}=view(candidates, :, 1),
  relative_tolerance::Float64=1e-10,
  absolute_tolerance::Float64=1e-12,
  maxiter::Int=30,
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
        iterations=maxiter,
        g_abstol=target,
        allow_f_increases=false,
        show_warnings=false,
      ),
    )
    return (
      u=B * Optim.minimizer(result),
      iterations=Optim.iterations(result),
      energy_evaluations=energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    u = B * alpha
    gradient = B' * Energies.gradient(e, u)
    norm(gradient) <= target && return (
      u=u, iterations=iteration, energy_evaluations=energy_evaluations
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
      if trial_energy <= energy0 + 1e-4 * step_length * slope ||
         (trial_energy <= energy0 + energy_resolution &&
          norm(B' * Energies.gradient(e, trial_u)) < norm(gradient))
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
    u=B * alpha,
    iterations=iterations_done,
    energy_evaluations=energy_evaluations,
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
  relative_tolerance::Float64=1e-10,
  absolute_tolerance::Float64=1e-12,
  maxiter::Int=30,
)
  size(directions, 2) == 0 && return (
    u=copy(anchor), iterations=0, energy_evaluations=0
  )
  B = try
    orthonormal_basis(directions)
  catch error
    error isa ArgumentError || rethrow()
    return (u=copy(anchor), iterations=0, energy_evaluations=0)
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
        iterations=maxiter,
        g_abstol=target,
        allow_f_increases=false,
        show_warnings=false,
      ),
    )
    return (
      u=anchor + B * Optim.minimizer(result),
      iterations=Optim.iterations(result),
      energy_evaluations=energy_evaluations,
    )
  end

  iterations_done = 0
  for iteration = 0:maxiter
    u = anchor + B * alpha
    gradient = B' * Energies.gradient(e, u)
    norm(gradient) <= target && return (
      u=u, iterations=iteration, energy_evaluations=energy_evaluations
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
    u=anchor + B * alpha,
    iterations=iterations_done,
    energy_evaluations=energy_evaluations,
  )
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(
  u::Vector{Float64},
  v::Vector{Float64},
  M::AbstractMatrix,
)
  w = u .- v
  return sqrt(dot(w, M * w))
end

"""
  restrict_to_subdomain!(u_local::AbstractVector, u_global::AbstractVector, idx_sub::AbstractVector)

Efficient restriction operator: extract subdomain values from global vector.
Writes u_local[k] = u_global[idx_sub[k]] for k = 1:length(idx_sub).
"""
function restrict_to_subdomain!(
  u_local::AbstractVector,
  u_global::AbstractVector,
  idx_sub::AbstractVector,
)
  for (k, j) in pairs(idx_sub)
    u_local[k] = u_global[j]
  end
  return u_local
end

"""
  restrict_to_subdomain(u_global::AbstractVector, idx_sub::AbstractVector)

Allocating version of restriction operator.
"""
function restrict_to_subdomain(
  u_global::AbstractVector,
  idx_sub::AbstractVector,
)
  u_local = similar(u_global, length(idx_sub))
  return restrict_to_subdomain!(u_local, u_global, idx_sub)
end

"""
  extend_from_subdomain!(u_global::AbstractVector, u_local::AbstractVector, idx_sub::AbstractVector)

Efficient extension operator: scatter subdomain values into global vector.
Writes u_global[idx_sub[k]] = u_local[k] for k = 1:length(idx_sub).
Does not zero out other entries - use zero_out_complement! if needed.
"""
function extend_from_subdomain!(
  u_global::AbstractVector,
  u_local::AbstractVector,
  idx_sub::AbstractVector,
)
  for (k, j) in pairs(idx_sub)
    u_global[j] = u_local[k]
  end
  return u_global
end

"""
  zero_out_complement!(u_global::AbstractVector, idx_sub::AbstractVector)

Zero out all entries of u_global except those indexed by idx_sub.
Useful for creating functions supported only on subdomain.
"""
function zero_out_complement!(u_global::AbstractVector, idx_sub::AbstractVector)
  # Create a set for fast lookup
  idx_set = Set(idx_sub)
  for i in eachindex(u_global)
    if i ∉ idx_set
      u_global[i] = 0.0
    end
  end
  return u_global
end

function zero_out_local!(u_global::AbstractVector, idx_sub::AbstractVector)
  for j in idx_sub
    u_global[j] = 0.0
  end
  return u_global
end

"""
  SubdomainView{T,V<:AbstractVector{T}} <: AbstractVector{T}

A view into a global vector that presents only the subdomain DOFs.
Avoids allocation when working with subdomain data.
"""
struct SubdomainView{T,V<:AbstractVector{T}} <: AbstractVector{T}
  parent::V
  indices::Vector{Int}
end

# AbstractArray interface
Base.size(v::SubdomainView) = (length(v.indices),)
Base.getindex(v::SubdomainView, i::Int) = v.parent[v.indices[i]]
Base.setindex!(v::SubdomainView, val, i::Int) = (v.parent[v.indices[i]] = val)
Base.IndexStyle(::Type{<:SubdomainView}) = IndexLinear()

"""
  subdomain_view(u_global::AbstractVector, idx_sub::AbstractVector)

Create a view into the global vector that presents only the subdomain DOFs.
Changes to the view are reflected in the original vector.
"""
function subdomain_view(
  u_global::AbstractVector{T},
  idx_sub::AbstractVector,
) where {T}
  return SubdomainView{T,typeof(u_global)}(u_global, collect(Int, idx_sub))
end

"""
  restrict_matrix_block!(A_local::AbstractMatrix, A_global::AbstractMatrix,
                        row_indices::AbstractVector, col_indices::AbstractVector)

Efficient matrix restriction: extract block from global matrix.
A_local[i,j] = A_global[row_indices[i], col_indices[j]]
"""
function restrict_matrix_block!(
  A_local::AbstractMatrix,
  A_global::AbstractMatrix,
  row_indices::AbstractVector,
  col_indices::AbstractVector,
)
  for (j, col_idx) in pairs(col_indices)
    for (i, row_idx) in pairs(row_indices)
      A_local[i, j] = A_global[row_idx, col_idx]
    end
  end
  return A_local
end

"""
  restrict_matrix_block(A_global::AbstractMatrix, indices::AbstractVector)

Allocating version for symmetric case: extract A_global[indices, indices].
"""
function restrict_matrix_block(
  A_global::AbstractMatrix,
  indices::AbstractVector,
)
  n = length(indices)
  A_local = Matrix{eltype(A_global)}(undef, n, n)
  return restrict_matrix_block!(A_local, A_global, indices, indices)
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
    partition_of_unity_weights(subdomain_dofs, ndofs)

Build diagonal algebraic partition-of-unity weights for an overlapping DOF
partition. If degree of freedom `j` belongs to `q` subdomains, its weight is
`1/q` in each of them. Thus every row of the returned `ndofs × nsubdomains`
matrix sums to one.

The weights can restrict overlapping local corrections before the global
combination step without requiring a separate, nonoverlapping owner partition.
"""
function partition_of_unity_weights(subdomain_dofs, ndofs::Integer)
  ndofs > 0 || throw(ArgumentError("ndofs must be positive"))
  nsubdomains = length(subdomain_dofs)
  nsubdomains > 0 || throw(ArgumentError("at least one subdomain is required"))

  weights = zeros(Float64, ndofs, nsubdomains)
  for (subdomain, indices) in enumerate(subdomain_dofs)
    all(index -> 1 <= index <= ndofs, indices) || throw(
      ArgumentError("subdomain $subdomain contains a DOF outside 1:$ndofs")
    )
    length(unique(indices)) == length(indices) ||
      throw(ArgumentError("subdomain $subdomain contains duplicate DOFs"))
    weights[indices, subdomain] .= 1.0
  end

  multiplicity = vec(sum(weights; dims=2))
  uncovered = findall(iszero, multiplicity)
  isempty(uncovered) || throw(
    ArgumentError(
      "partition-of-unity restriction requires every DOF to be covered; " *
      "uncovered DOFs: $(join(uncovered, ", "))",
    ),
  )
  weights ./= multiplicity
  return weights
end

"Assign every degree of freedom to one of its nonoverlapping core candidates."
function _balanced_disjoint_cores(core_dofs, ndofs::Integer)
  memberships = [Int[] for _ in 1:ndofs]
  for (subdomain, indices) in enumerate(core_dofs)
    length(unique(indices)) == length(indices) ||
      throw(ArgumentError("core $subdomain contains duplicate DOFs"))
    for index in indices
      1 <= index <= ndofs ||
        throw(ArgumentError("core $subdomain contains a DOF outside 1:$ndofs"))
      push!(memberships[index], subdomain)
    end
  end
  uncovered = findall(isempty, memberships)
  isempty(uncovered) || throw(
    ArgumentError(
      "the nonoverlapping cores must cover every DOF; uncovered DOFs: " *
      join(uncovered, ", "),
    ),
  )

  owned = [Int[] for _ in core_dofs]
  loads = zeros(Int, length(core_dofs))
  for degree in sortperm(length.(memberships))
    candidates = memberships[degree]
    owner = candidates[argmin(view(loads, candidates))]
    push!(owned[owner], degree)
    loads[owner] += 1
  end
  return owned
end

"""
    nicolaides_coarse_basis(A, core_dofs, subdomain_dofs; normalize=true)

Construct a low-energy Nicolaides-type partition-of-unity coarse space for a
scalar elliptic operator `A`. There must be one nonoverlapping core and one
overlapping DOF set per subdomain. Interface DOFs appearing in multiple cores
are assigned to one core with balanced ownership.

For subdomain `i`, the raw basis function is one on its owned core, zero
outside its overlapping DOFs, and discrete harmonic in the transition region
`T_i`:

```text
A[T_i,T_i] * theta_hat_i = -A[T_i,C_i] * 1.
```

With `normalize=true`, the default, the raw functions are divided pointwise by
their sum. The returned columns therefore form a partition of unity. Use this
basis as `coarse_basis` in [`var_dd`](@ref). Multiplicity weights from
[`partition_of_unity_weights`](@ref) serve a different purpose: they restrict
REMDD local corrections and are not a low-energy coarse basis.
"""
function nicolaides_coarse_basis(
  A::AbstractMatrix, core_dofs, subdomain_dofs; normalize::Bool=true
)
  ndofs = size(A, 1)
  size(A, 2) == ndofs || throw(DimensionMismatch("A must be square"))
  nsubdomains = length(subdomain_dofs)
  nsubdomains > 0 || throw(ArgumentError("at least one subdomain is required"))
  length(core_dofs) == nsubdomains || throw(
    DimensionMismatch(
      "there must be one core and one overlapping DOF set per subdomain"
    ),
  )
  all(isfinite, A) || throw(ArgumentError("A must contain only finite values"))
  issymmetric(A) || throw(ArgumentError("A must be symmetric"))

  overlapping = Vector{Vector{Int}}(undef, nsubdomains)
  for (subdomain, indices) in enumerate(subdomain_dofs)
    length(unique(indices)) == length(indices) ||
      throw(ArgumentError("subdomain $subdomain contains duplicate DOFs"))
    all(index -> 1 <= index <= ndofs, indices) || throw(
      ArgumentError("subdomain $subdomain contains a DOF outside 1:$ndofs")
    )
    overlapping[subdomain] = collect(Int, indices)
  end

  owned_cores = _balanced_disjoint_cores(core_dofs, ndofs)
  basis = zeros(Float64, ndofs, nsubdomains)
  for subdomain in 1:nsubdomains
    core = owned_cores[subdomain]
    isempty(core) && throw(ArgumentError("core $subdomain owns no DOFs"))
    overlap = overlapping[subdomain]
    overlap_membership = Set(overlap)
    all(index -> index in overlap_membership, core) || throw(
      ArgumentError(
        "owned core $subdomain must be contained in its overlapping DOF set"
      ),
    )
    transition = setdiff(overlap, core)
    basis[core, subdomain] .= 1.0
    if !isempty(transition)
      rhs = -(A[transition, core] * ones(length(core)))
      basis[transition, subdomain] .= Symmetric(A[transition, transition]) \ rhs
    end
  end

  all(isfinite, basis) || throw(
    ArgumentError("the local harmonic extensions produced non-finite values")
  )
  if normalize
    row_sums = vec(sum(basis; dims=2))
    threshold = sqrt(eps(Float64)) * max(1.0, maximum(abs, row_sums))
    all(sum -> abs(sum) > threshold, row_sums) || throw(
      ArgumentError(
        "the harmonic basis cannot be normalized because its pointwise sum vanishes",
      ),
    )
    basis ./= row_sums
  end
  return basis
end

"""
Restrict the genuinely local part of each candidate in-place with diagonal
PoU weights. A local candidate has the form `alpha_i * anchor + z_i`, with
`z_i` supported on subdomain `i`; removing `alpha_i` makes the operation
invariant under the arbitrary scaling of solution candidates.
"""
function restrict_local_candidates!(
  candidates::AbstractMatrix,
  anchor::AbstractVector,
  weights::AbstractMatrix,
  subdomain_dofs,
  exterior_dofs=nothing,
)
  size(candidates) == size(weights) || throw(
    DimensionMismatch(
      "candidate and partition-of-unity matrices must have the same size"
    ),
  )
  size(candidates, 1) == length(anchor) ||
    throw(DimensionMismatch("candidate rows must match the anchor length"))
  size(candidates, 2) == length(subdomain_dofs) ||
    throw(DimensionMismatch("there must be one DOF set per candidate"))
  for subdomain in axes(candidates, 2)
    candidate = view(candidates, :, subdomain)
    indices = subdomain_dofs[subdomain]
    exterior = if isnothing(exterior_dofs)
      setdiff(eachindex(anchor), indices)
    else
      exterior_dofs[subdomain]
    end
    exterior_anchor = view(anchor, exterior)
    denominator = dot(exterior_anchor, exterior_anchor)
    alpha = if iszero(denominator)
      0.0
    else
      dot(exterior_anchor, view(candidate, exterior)) / denominator
    end
    local_direction = candidate[indices] .- alpha .* anchor[indices]
    candidate .= anchor
    candidate[indices] .+= weights[indices, subdomain] .* local_direction
  end
  return candidates
end

"""
    var_dd(e, subdomain_dofs; maxiter=50, tol=1e-8, quadratic_model=false,
           frozen_gp_model=false, tangent_gp_model=false,
           density_mixing_alpha=1.0, ...)

Variational domain decomposition algorithm for solving various energy minimization problems.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: Energy functional to minimize
- `subdomain_dofs::Vector{Vector{Int32}}`: DOF indices for each subdomain

# Keyword Arguments
- `maxiter::Int=50`: Maximum number of iterations
- `tol::Float64=1e-8`: Convergence tolerance based on residual norm
- `save_local_updates::Bool=false`: Whether to save and output local updates from each subdomain
- `fe_space=nothing`: FE space for VTK output (required if save_local_updates=true)
- `output_prefix::String="dd_local_update"`: Prefix for VTK files of local updates
- `u0::Union{Nothing,Vector{Float64}}=nothing`: Initial guess (defaults to all-ones);
  useful for warm starts, e.g. across time steps of a gradient flow
- `history_depth::Int=0`: Number of previous global iterates added to the
  second-level trial space. `history_depth=1` adds `u_{k-1}` alongside `u_k`.
- `mixing_omega::Float64=0.0`: Post-combination damping parameter in
  `u_{k+1} = ω u_k + (1-ω) ũ_{k+1}`. The default zero keeps the exact
  second-level minimizer.
- `quadratic_model::Bool=false`: Build one quadratic Taylor model of `e` at
  the start of each outer sweep and use it for the local subdomain solves. The
  second-level combination, convergence test, and histories use the original
  energy. This requires an energy Hessian.
- `frozen_gp_model::Bool=false`: For a Gross--Pitaevskii energy, freeze the
  nonlinear density at the current normalized global iterate and use the
  resulting generalized linear eigenproblem for every local solve in that
  sweep. The second-level combination remains a full nonlinear GP solve.
- `density_mixing_alpha::Float64=1.0`: For the frozen GP model, use the mixed
  density `alpha * rho_k + (1-alpha) * rho_{k-1}` after the first sweep.
  `alpha=1` recovers the unmixed frozen-density method.
- `tangent_gp_model::Bool=false`: For a Gross--Pitaevskii energy, solve one
  quadratic Taylor model of the physical energy in each enriched local tangent
  space. Candidates are retracted to unit mass; the second-level combination
  still minimizes the full nonlinear GP quotient.
- `sweep::Symbol=:additive`: Local-update mode. `:additive` forms all local
  candidates from `u_k`, so those `m` solves can run in parallel.
  `:multiplicative` feeds each local update into the next subdomain and thus
  has a serial critical path of `m` local solves. The latter currently supports
  `QuadraticEnergy` and reproduces the original projectively rescaled sweep.
- `restriction::Symbol=:none`: Treatment of overlapping local corrections in
  the second-level trial space. Writing each local candidate as
  `u_i=alpha_i*u_k+z_i`, with `z_i` supported on subdomain `i`,
  `:partition_of_unity` replaces it by `u_k+D_i*z_i`, where the diagonal
  multiplicity weights obey `sum(D_i)=I`. This restricted mode is available
  for additive sweeps and is referred to as restricted EMDD (REMDD).
- `coarse_basis=nothing`: Optional global coarse-space basis appended to the
  second-level trial space. For scalar Poisson problems,
  `nicolaides_coarse_basis(e.A, core_dofs, subdomain_dofs)` supplies one
  discrete-harmonic partition-of-unity mode per subdomain.
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
"""
function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{Vector{Int32}};
  maxiter::Int = 50,
  tol::Float64 = 1e-8,
  save_local_updates::Bool = false,
  u0::Union{Nothing,Vector{Float64}} = nothing,
  history_depth::Int = 0,
  mixing_omega::Float64 = 0.0,
  quadratic_model::Bool = false,
  frozen_gp_model::Bool = false,
  tangent_gp_model::Bool = false,
  density_mixing_alpha::Float64 = 1.0,
  sweep::Symbol = :additive,
  restriction::Symbol = :none,
  coarse_basis = nothing,
  subspace_callback = nothing,
  local_solve_callback = nothing,
  verbose::Bool = true,
)
  # TODO: Make local updates and other returns more elgant with info struct?

  history_depth >= 0 || throw(ArgumentError("history_depth must be nonnegative"))
  0.0 <= mixing_omega < 1.0 ||
    throw(ArgumentError("mixing_omega must satisfy 0 <= mixing_omega < 1"))
  sweep in (:additive, :multiplicative) || throw(ArgumentError(
    "sweep must be :additive or :multiplicative",
  ))
  restriction in (:none, :partition_of_unity) || throw(ArgumentError(
    "restriction must be :none or :partition_of_unity",
  ))
  restriction == :partition_of_unity && sweep != :additive && throw(
    ArgumentError("partition-of-unity restriction requires sweep=:additive"),
  )
  quadratic_model + frozen_gp_model + tangent_gp_model <= 1 || throw(ArgumentError(
    "quadratic_model, frozen_gp_model, and tangent_gp_model are mutually exclusive",
  ))
  0.0 < density_mixing_alpha <= 1.0 || throw(ArgumentError(
    "density_mixing_alpha must satisfy 0 < density_mixing_alpha <= 1",
  ))
  density_mixing_alpha != 1.0 && !frozen_gp_model && throw(ArgumentError(
    "density_mixing_alpha requires frozen_gp_model=true",
  ))
  frozen_gp_model && !(e isa Energies.GrossPitaevskiiRayleighQuotient) &&
    throw(ArgumentError(
      "frozen_gp_model is only available for GrossPitaevskiiRayleighQuotient",
    ))
  tangent_gp_model && !(e isa Energies.GrossPitaevskiiRayleighQuotient) &&
    throw(ArgumentError(
      "tangent_gp_model is only available for GrossPitaevskiiRayleighQuotient",
    ))
  tangent_gp_model && sweep != :additive && throw(ArgumentError(
    "tangent_gp_model currently requires sweep=:additive",
  ))

  # Initial guess, no need to normalize apparently
  u_cur = isnothing(u0) ? ones(Energies.dimension(e)) : copy(u0)
  if !isnothing(coarse_basis)
    ndims(coarse_basis) == 2 || throw(ArgumentError(
      "coarse_basis must be a matrix",
    ))
    size(coarse_basis, 1) == length(u_cur) || throw(DimensionMismatch(
      "coarse_basis rows must match the energy dimension",
    ))
    size(coarse_basis, 2) > 0 || throw(ArgumentError(
      "coarse_basis must contain at least one vector",
    ))
    all(isfinite, coarse_basis) || throw(ArgumentError(
      "coarse_basis must contain only finite values",
    ))
  end

  e_hist = Float64[]
  sol_hist = Vector{Vector{Float64}}()
  resnorm_hist = Float64[]
  local_update_hist = Vector{Vector{Vector{Float64}}}()
  previous_iterates = Vector{Vector{Float64}}()
  previous_density_iterate = nothing

  e_cur = e(u_cur)
  push!(e_hist, e_cur)
  push!(sol_hist, copy(u_cur))

  m = length(subdomain_dofs)
  restriction_weights = restriction == :partition_of_unity ?
    partition_of_unity_weights(subdomain_dofs, length(u_cur)) : nothing
  restriction_exteriors = restriction == :partition_of_unity ? [
    setdiff(eachindex(u_cur), indices) for indices in subdomain_dofs
  ] : nothing

  local_updates = zeros(size(u_cur, 1), m)  # preallocate for efficiency
  for n = 1:maxiter
    sweep_energy = if quadratic_model
      Energies.quadratic_model(e, u_cur)
    elseif frozen_gp_model
      Energies.frozen_density_model(
        e,
        u_cur;
        previous=previous_density_iterate,
        alpha=density_mixing_alpha,
      )
    elseif tangent_gp_model
      Energies.tangent_quadratic_model(e, u_cur)
    else
      e
    end
    current_local_updates = Vector{Vector{Float64}}()
    multiplicative_iterate = copy(u_cur)

    for i = 1:m
      local_base = sweep == :additive ? u_cur : multiplicative_iterate
      local_result = if sweep == :additive
        inf_step_with_info(
          sweep_energy,
          u_cur,
          subdomain_dofs[i];
          collect_info=!isnothing(local_solve_callback),
        )
      else
        multiplicative_iterate = multiplicative_inf_step(
          sweep_energy, multiplicative_iterate, subdomain_dofs[i]
        )
        (
          u=multiplicative_iterate,
          info=(
            dimension=1 + length(subdomain_dofs[i]),
            iterations=-1,
            converged=true,
            residual=NaN,
            k_nnz=-1,
            factor_nnz=-1,
          ),
        )
      end
      u_next_i = local_result.u
      !isnothing(local_solve_callback) &&
        local_solve_callback(n, i, local_result.info)
      local_updates[:, i] = u_next_i

      if save_local_updates
        push!(current_local_updates, copy(u_next_i .- local_base))
      end
    end

    if save_local_updates
      push!(local_update_hist, current_local_updates)
    end

    if restriction == :partition_of_unity
      restrict_local_candidates!(
        local_updates,
        u_cur,
        restriction_weights,
        subdomain_dofs,
        restriction_exteriors,
      )
    end

    combination_anchor = sweep == :additive ? u_cur : multiplicative_iterate
    combined_matrix = isnothing(coarse_basis) ?
      hcat(combination_anchor, previous_iterates..., local_updates) :
      hcat(
        combination_anchor,
        previous_iterates...,
        local_updates,
        coarse_basis,
      )
    !isnothing(subspace_callback) && subspace_callback(n, combined_matrix)
    u_trial = combine_step(e, combined_matrix)
    u_new = mix_iterates(e, u_cur, u_trial, mixing_omega)

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
      if save_local_updates
        return u_new, e_new, e_hist, sol_hist, resnorm_hist, local_update_hist
      else
        return u_new, e_new, e_hist, sol_hist, resnorm_hist
      end
    end

    if history_depth > 0
      push!(previous_iterates, copy(u_cur))
      length(previous_iterates) > history_depth && popfirst!(previous_iterates)
    end
    frozen_gp_model && (previous_density_iterate = copy(u_cur))
    u_cur = u_new
    e_cur = e_new
  end

  @warn "Reached maxiter=$maxiter with energy ≈ $e_cur"
  if save_local_updates
    return u_cur, e_cur, e_hist, sol_hist, resnorm_hist, local_update_hist
  else
    return u_cur, e_cur, e_hist, sol_hist, resnorm_hist
  end
end

function mix_iterates(
  e::Energies.AbstractEnergy,
  u_cur::AbstractVector,
  u_trial::AbstractVector,
  omega::Real,
)
  omega == 0 && return u_trial
  return omega .* u_cur .+ (1 - omega) .* u_trial
end

function mix_iterates(
  e::Energies.GeneralizedRayleighQuotient,
  u_cur::AbstractVector,
  u_trial::AbstractVector,
  omega::Real,
)
  omega == 0 && return u_trial
  # Ritz vectors are defined only up to sign. Align the trial vector with the
  # current iterate before interpolation to avoid artificial cancellation.
  aligned_trial = dot(u_cur, e.B * u_trial) < 0 ? -u_trial : u_trial
  u_new = omega .* u_cur .+ (1 - omega) .* aligned_trial
  Energies.normalize_M!(u_new, e.B)
  return u_new
end

function mix_iterates(
  e::Energies.GrossPitaevskiiRayleighQuotient,
  u_cur::AbstractVector,
  u_trial::AbstractVector,
  omega::Real,
)
  omega == 0 && return u_trial
  aligned_trial = dot(u_cur, e.M * u_trial) < 0 ? -u_trial : u_trial
  u_new = omega .* u_cur .+ (1 - omega) .* aligned_trial
  Energies.normalize_M!(u_new, e.M)
  return u_new
end

end # module
