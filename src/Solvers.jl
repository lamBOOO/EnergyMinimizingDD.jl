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

using VariationalDomainDecomposition.Energies

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
  α_new = A_local \ b_local

  # Use helper function to reconstruct (no normalization needed for QuadraticEnergy)
  x_new = reconstruct_implicit_affine!(α_new, u_cur, idx_sub)
  return x_new
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

  # Reconstruct solution in original space
  x_new = reconstruct_implicit_affine!(α_new, u_cur, idx_sub)
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

  # Use helper function to reconstruct, then normalize
  x_new = reconstruct_implicit_affine!(res.X[:, 1], u_cur, idx_sub)
  # Energies.normalize_M!(x_new, M)

  return x_new
end

function inf_step(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)

  m = length(idx_sub)
  if m == 0
    return copy(u_cur)
  end

  # u = u_cur - Rᵢ u_cur + Rᵢᵀz z
  function affine_extension(z, u_cur)
    u = copy(u_cur)
    @inbounds for (k, j) in pairs(idx_sub)
      u[j] = z[k]
    end
    return u
  end

  # Reduced quantities at z
  function reduced_grad(z)
    u = affine_extension(z, u_cur)
    g_full = Energies.gradient(e, u)
    g = similar(z)
    @inbounds for (k, j) in pairs(idx_sub)
      g[k] = g_full[j]                 # g = Vᵀ ∇E
    end
    return g, u, g_full
  end

  # Finite-diff reduced Hessian (or replace by Hessian–vector products)
  function reduced_hessian_fd(z, g_at_z; eps = 1e-6)
    H = zeros(m, m)
    for i = 1:m
      zpert = copy(z)
      zpert[i] += eps
      g_pert, _, _ = reduced_grad(zpert)
      @inbounds H[:, i] = (g_pert .- g_at_z) ./ eps
    end
    # Tikhonov for stability
    @inbounds for i = 1:m
      H[i, i] += 1e-10
    end
    return H
  end

  # Backtracking on the true energy along x + V(z + t*p)
  function linesearch(z, p, E0, u0; c = 1e-4, tau = 0.5, tmin = 1e-6)
    t = 1.0
    while t ≥ tmin
      u = affine_extension(z .+ t .* p, u_cur)
      if Energies.energy(e, u) ≤ E0 - c * t * dot(p, p) # simple decrease test
        return t, u
      end
      t *= tau
    end
    return 0.0, u0
  end

  z = zeros(m)  # start at the current point: u = u_cur + V*z with z=0
  maxit = 50
  tol = 1e-6

  for k = 1:maxit
    g, u, _ = reduced_grad(z)
    if norm(g) < tol
      return u
    end
    H = reduced_hessian_fd(z, g)  # better: use analytic Hessian or Hv products
    # Try Newton; fall back to gradient step if singular
    p = try
      -H \ g
    catch
      -g / (norm(g) + 1e-12)
    end
    E0 = Energies.energy(e, u)
    t, u_new = linesearch(z, p, E0, u)
    if t == 0.0
      # fall back to steepest descent
      p = -g
      t, u_new = linesearch(z, p, E0, u)
      if t == 0.0
        return u  # give up (likely very flat)
      end
    end
    z .+= t .* p
  end
  return affine_extension(z, zeros(length(u_cur)))
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

function combine_step(
  e::Energies.QuadraticEnergy{Float64},
  sspace::Matrix{Float64},
)
  # TODO: Is combine step the same as inf step in general?
  # => Just 1st order optimality in subspace?
  A, b = e.A, e.b

  B = Matrix(qr(sspace).Q)

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

  B = Matrix(qr(sspace).Q)

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
  e::Energies.LinearRegressionEnergy{Float64},
  sspace::Matrix{Float64},
)
  # For linear regression, the combine step solves the least squares problem
  # in the subspace spanned by the columns of sspace: min ||A*B*α - b||²
  # where B = QR factorization of sspace

  A, b = e.A, e.b

  B = Matrix(qr(sspace).Q)

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

"""
  combine_step(e::NonlinearEnergy, sspace)

Perform combine step for nonlinear energy by minimizing energy in the subspace
spanned by the columns of sspace using Newton's method.
Works for any nonlinear problem: PDEs, algebraic systems, etc.
"""
function combine_step(
  e::Energies.NonlinearEnergy{Float64},
  sspace::Matrix{Float64},
)
  B = Matrix(qr(sspace).Q)
  localdim = size(B, 2)

  # Newton iteration in subspace
  α = zeros(localdim)
  α[1] = 1.0  # Initial guess: mostly first component

  max_newton_iter = 10
  newton_tol = 1e-6

  for iter = 1:max_newton_iter
    u_current = B * α

    # Compute gradient and project to subspace
    grad_full = Energies.gradient(e, u_current)
    g_local = B' * grad_full

    # Check convergence
    if norm(g_local) < newton_tol
      break
    end

    # Approximate Hessian in subspace using finite differences
    H_local = zeros(localdim, localdim)
    eps_fd = 1e-6

    for i = 1:localdim
      α_plus = copy(α)
      α_plus[i] += eps_fd
      u_plus = B * α_plus
      grad_plus = Energies.gradient(e, u_plus)
      g_plus = B' * grad_plus
      H_local[:, i] = (g_plus .- g_local) / eps_fd
    end

    # Newton step with regularization
    H_reg = H_local + 1e-8 * I
    try
      Δα = H_reg \ (-g_local)

      # Simple line search
      step_size = 1.0
      while step_size > 1e-4
        α_test = α + step_size * Δα
        u_test = B * α_test
        if Energies.energy(e, u_test) <= Energies.energy(e, u_current)
          break
        end
        step_size *= 0.5
      end

      α += step_size * Δα
    catch
      # Fallback to steepest descent
      α -= 0.01 * g_local / (norm(g_local) + 1e-12)
    end
  end

  return B * α
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
  reconstruct_implicit_affine!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)

Builds x_new = 1 * u_cur + Σ α[k+1]/α[1] * e_jk (affine since u_curr has coeff 1)
where e_j are standard basis vectors, without normalization.
This avoids explicitly constructing the basis matrix.
- Work directly with u_cur to avoid allocation
- Mutates u_cur in-place
"""
function reconstruct_implicit_affine!(
  α::Vector{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector,
)
  # TODO: Avoid that function and the gloabl reconstruction
  # => Work on local coeffs directly and only to global in the end
  # Add contributions from standard basis vectors
  for (k, j) in pairs(idx_sub)
    u_cur[j] += α[k+1] / α[1] # Add coefficient for e_j
  end
  return u_cur
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
"""
function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{Vector{Int32}};
  maxiter::Int = 50,
  tol::Float64 = 1e-8,
  save_local_updates::Bool = false,
)
  # TODO: Add "sweep" option [3->1->5->7], multiplicative version [1->2->3]
  # TODO: Make local updates and other returns more elgant with info struct?

  # Initial guess, no need to normalize apparently
  u_cur = ones(Energies.dimension(e))

  e_hist = Float64[]
  sol_hist = Vector{Vector{Float64}}()
  local_update_hist = Vector{Vector{Vector{Float64}}}()

  e_cur = e(u_cur)
  push!(e_hist, e_cur)
  push!(sol_hist, copy(u_cur))

  m = length(subdomain_dofs)

  local_updates = zeros(size(u_cur, 1), m)  # preallocate for efficiency
  for n = 1:maxiter
    current_local_updates = Vector{Vector{Float64}}()

    for i = 1:m
      u_next_i = inf_step(e, u_cur, subdomain_dofs[i])
      local_updates[:, i] = u_next_i .- u_cur

      if save_local_updates
        push!(current_local_updates, copy(u_next_i .- u_cur))
      end
    end

    if save_local_updates
      push!(local_update_hist, current_local_updates)
    end

    combined_matrix = hcat(u_cur, local_updates)
    u_new = combine_step(e, combined_matrix)

    resnorm = Energies.residual_norm(e, u_new)
    @printf(
      "Iteration %3d: Residual norm ≈ %12.6e energy = %12.6e\n",
      n,
      resnorm,
      e(u_new)
    )

    e_new = e(u_new)
    push!(e_hist, e_new)
    push!(sol_hist, copy(u_new))
    if resnorm < tol
      println("Converged at iteration $n with energy e = $e_new")
      if save_local_updates
        return u_new, e_new, e_hist, sol_hist, local_update_hist
      else
        return u_new, e_new, e_hist, sol_hist
      end
    end

    u_cur = u_new
    e_cur = e_new
  end

  @warn "Reached maxiter=$maxiter with energy ≈ $e_cur"
  if save_local_updates
    return u_cur, e_cur, e_hist, sol_hist, local_update_hist
  else
    return u_cur, e_cur, e_hist, sol_hist
  end
end

end # module
