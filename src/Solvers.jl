module Solvers

using LinearAlgebra
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

using VariationalDD.Energies

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
function inf_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  localdim = 1 + length(idx_sub)

  K, M = e.A, e.B

  # remove local constribution from u_curr: u_curr - Ri^T Ri u_cur
  # => set entries in idx_sub to zero
  u_cur = copy(u_cur)
  zero_out_local!(u_cur, idx_sub)

  # Extract relevant rows/columns from K and M
  if length(idx_sub) > 0
    # Build the local matrices more efficiently
    K_local = zeros(localdim, localdim)
    M_local = zeros(localdim, localdim)

    # First row/column: u_cur' * K/M * [u_cur, e_j1, e_j2, ...]
    K_u = K * u_cur
    M_u = M * u_cur

    K_local[1, 1] = dot(u_cur, K_u)  # u' * K * u
    M_local[1, 1] = dot(u_cur, M_u)  # u' * M * u

    # First row/column: u_cur' * K/M * e_j
    for (k, j) in pairs(idx_sub)
      K_local[1, k+1] = K_u[j]  # u' * K * e_j = (K * u)[j]
      K_local[k+1, 1] = K_u[j]  # e_j' * K * u = (K * u)[j] (symmetric)
      M_local[1, k+1] = M_u[j]  # u' * M * e_j = (M * u)[j]
      M_local[k+1, 1] = M_u[j]  # e_j' * M * u = (M * u)[j] (symmetric)
    end

    # Remaining entries: e_i' * K/M * e_j = K[i,j] and M[i,j]
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        K_local[k1+1, k2+1] = K[j1, j2]
        M_local[k1+1, k2+1] = M[j1, j2]
      end
    end
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
  res =
    lobpcg(K_local_sym, M_local_sym, false, 1; P = F, tol = 1e-8, maxiter = 500)

  # Return the Ritz vector itself. Its normalization is immaterial to the
  # subsequent Rayleigh--Ritz combination and is performed there.
  x_new = reconstruct_implicit!(res.X[:, 1], u_cur, idx_sub)

  return x_new
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

  n = length(u_cur)
  local_space = zeros(Float64, n, 1 + length(idx_sub))
  local_space[:, 1] .= u_cur
  zero_out_local!(view(local_space, :, 1), idx_sub)
  for (k, j) in pairs(idx_sub)
    local_space[j, k+1] = 1.0
  end
  return _minimize_subspace(e, local_space; initial = u_cur)
end

function inf_step(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  return nonlinear_local_minimize(e, u_cur, idx_sub).u
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
    _minimize_subspace(e, X; initial=X[:, 1], maxiter=200, tol=1e-9)

Minimize the scale-invariant Gross--Pitaevskii quotient in `range(X)` using
unconstrained L-BFGS in reduced coordinates. Since the quotient itself is
scale invariant, no normalization constraint is imposed during optimization;
only the returned representative is M-normalized.
"""
function _minimize_subspace(
  e::Energies.GrossPitaevskiiRayleighQuotient{Float64},
  X::AbstractMatrix{Float64};
  initial::AbstractVector{Float64} = X[:, 1],
  maxiter::Int = 200,
  tol::Float64 = 1e-9,
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

  function objective(alpha)
    dot(alpha, alpha) > eps(Float64) || return Inf
    return Energies.energy(e, Q * alpha)
  end

  function reduced_gradient!(storage, alpha)
    dot(alpha, alpha) > eps(Float64) || throw(
      ArgumentError("reduced GP quotient is undefined at the zero vector"),
    )
    storage .= Q' * Energies.gradient(e, Q * alpha)
    return storage
  end

  result = Optim.optimize(
    objective,
    reduced_gradient!,
    alpha,
    Optim.LBFGS(),
    Optim.Options(
      iterations = maxiter,
      g_abstol = tol,
      allow_f_increases = false,
      show_warnings = false,
    ),
  )
  u = Q * Optim.minimizer(result)

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
  return _minimize_subspace(
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
    var_dd(e, subdomain_dofs; maxiter=50, tol=1e-8, save_local_updates=false, fe_space=nothing, output_prefix="dd_local_update")

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
- `sweep::Symbol=:additive`: Local-update mode. `:additive` forms all local
  candidates from `u_k`, so those `m` solves can run in parallel.
  `:multiplicative` feeds each local update into the next subdomain and thus
  has a serial critical path of `m` local solves. The latter currently supports
  `QuadraticEnergy` and reproduces the original projectively rescaled sweep.
- `subspace_callback=nothing`: Optional study/diagnostic hook called as
  `subspace_callback(iteration, combined_matrix)` immediately before the
  second-level minimization. It does not alter the algorithm.
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
  sweep::Symbol = :additive,
  subspace_callback = nothing,
  verbose::Bool = true,
)
  # TODO: Make local updates and other returns more elgant with info struct?

  history_depth >= 0 || throw(ArgumentError("history_depth must be nonnegative"))
  0.0 <= mixing_omega < 1.0 ||
    throw(ArgumentError("mixing_omega must satisfy 0 <= mixing_omega < 1"))
  sweep in (:additive, :multiplicative) || throw(ArgumentError(
    "sweep must be :additive or :multiplicative",
  ))

  # Initial guess, no need to normalize apparently
  u_cur = isnothing(u0) ? ones(Energies.dimension(e)) : copy(u0)

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
    current_local_updates = Vector{Vector{Float64}}()
    multiplicative_iterate = copy(u_cur)

    for i = 1:m
      local_base = sweep == :additive ? u_cur : multiplicative_iterate
      u_next_i = if sweep == :additive
        inf_step(e, u_cur, subdomain_dofs[i])
      else
        multiplicative_iterate = multiplicative_inf_step(
          e, multiplicative_iterate, subdomain_dofs[i]
        )
      end
      local_updates[:, i] = u_next_i

      if save_local_updates
        push!(current_local_updates, copy(u_next_i .- local_base))
      end
    end

    if save_local_updates
      push!(local_update_hist, current_local_updates)
    end

    combination_anchor = sweep == :additive ? u_cur : multiplicative_iterate
    combined_matrix = hcat(combination_anchor, previous_iterates..., local_updates)
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

function mix_iterates(
  e::Energies.RayleighQuotient,
  u_cur::AbstractVector,
  u_trial::AbstractVector,
  omega::Real,
)
  omega == 0 && return u_trial
  aligned_trial = dot(u_cur, u_trial) < 0 ? -u_trial : u_trial
  u_new = omega .* u_cur .+ (1 - omega) .* aligned_trial
  u_new ./= norm(u_new)
  return u_new
end

end # module
