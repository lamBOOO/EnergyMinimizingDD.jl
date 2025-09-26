using LinearAlgebra
using FiniteDiff
using Gridap
using GridapDistributed
using Metis
using IterativeSolvers
using Arpack
using Printf


module Energies

using LinearAlgebra
"""
AbstractEnergy{T} represents a scalar-valued energy E(x)::T.
Implement at least `energy(e, x)`. Optionally `gradient`, `gradient!`,
`hessian`, and `hessian!`.
"""
abstract type AbstractEnergy{T} end

# Callable sugar: e(x) calls energy(e, x)
(e::AbstractEnergy)(x) = energy(e, x)

# Generic fallbacks (you can AD these later if you want)
energy(e::AbstractEnergy, x) = error("energy not implemented for $(typeof(e))")

# Underlying dimension
dimension(e::AbstractEnergy) = error("dimension not implemented for $(typeof(e))")

# Optionally provide defaults via AD; otherwise keep them abstract.
gradient(e::AbstractEnergy, x) = error("gradient not implemented for $(typeof(e))")
hessian(e::AbstractEnergy, x) = error("hessian not implemented for $(typeof(e))")

# In-place variants are optional but nice for performance
function gradient!(g, e::AbstractEnergy, x)
  g .= gradient(e, x)       # default: out-of-place -> in-place
  return g
end
function hessian!(H, e::AbstractEnergy, x)
  H .= hessian(e, x)
  return H
end

residual_norm(e::AbstractEnergy, x) = error("residual_norm not implemented for $(typeof(e))")

"""
  QuadraticEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    <: AbstractEnergy{T}

A concrete implementation of `AbstractEnergy` representing a quadratic energy
function.

This structure is parameterized by:
- `T`: The numeric type (e.g., Float64, Float32)
- `M`: The matrix type that must be a subtype of `AbstractMatrix{T}`
- `V`: The vector type that must be a subtype of `AbstractVector{T}`

Quadratic energy functions typically have the form
E(x) = ½xᵀAx + bᵀx + c, where A is a matrix,
b is a vector, and c is a scalar constant.
"""
struct QuadraticEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractEnergy{T}
  A::M           # can be Dense, Sparse, or Symmetric wrapper
  b::V
  c::T
end

# Make a convenient constructor; wrap A as Symmetric if you know it.
QuadraticEnergy(A::AbstractMatrix{T}, b::AbstractVector{T}; c::T=zero(T)) where {T} =
  QuadraticEnergy{T,typeof(A),typeof(b)}(A, b, c)

# E(x)
energy(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  T(0.5) * dot(x, e.A * x) - dot(e.b, x) + e.c

# dimension
dimension(e::QuadraticEnergy) = size(e.A, 1)

# ∇E(x) = Ax - b  (if A symmetric; if not, this is gradient of 1/2 x'(A+A')x - b'x)
gradient(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  e.A * x .- e.b

# ∇²E(x) = A (constant)
hessian(e::QuadraticEnergy{T}) where {T} = e.A
hessian(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = hessian(e)

function residual_norm(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T}
  return e.A * x .- e.b |> norm
end


# 2) Rayleigh quotient:  ρ(x) = (x'Ax) / (x'x), scale-invariant in x ≠ 0
struct RayleighQuotient{T,M<:AbstractMatrix{T}} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
end

RayleighQuotient(A::AbstractMatrix{T}) where {T} =
  RayleighQuotient{T,typeof(A)}(A)

# ρ(x)
energy(e::RayleighQuotient{T}, x::AbstractVector{T}) where {T} = begin
  num = dot(x, e.A * x)
  den = dot(x, x)
  @assert den != zero(T) "Rayleigh quotient undefined at x=0"
  num / den
end

# dimension
dimension(e::RayleighQuotient) = size(e.A, 1)

# ∇ρ(x) = 2 * ( (Ax)(x⋅x) - x(x⋅Ax) ) / (x⋅x)^2
gradient(e::RayleighQuotient{T}, x::AbstractVector{T}) where {T} = begin
  Ax = e.A * x
  xx = dot(x, x)
  xAx = dot(x, Ax)
  @assert xx != zero(T) "Rayleigh quotient gradient undefined at x=0"
  (2 / (xx * xx)) * (Ax .* xx .- x .* xAx)
end

# A helper: in-place gradient for performance
function gradient!(g::AbstractVector, e::RayleighQuotient, x::AbstractVector)
  Ax = e.A * x
  xx = dot(x, x)
  xAx = dot(x, Ax)
  @assert xx != 0 "Rayleigh quotient gradient undefined at x=0"
  # g = 2 * (Ax*xx - x*xAx) / xx^2
  @. g = 2 * (Ax * xx - x * xAx) / (xx * xx)
  return g
end

function hessian(e::RayleighQuotient, x::AbstractVector)
  xx = dot(x, x)
  @assert xx != zero(eltype(x)) "Rayleigh quotient hessian undefined at x=0"

  # Compute Ax once and reuse
  Ax = e.A * x
  xAx = dot(x, Ax)
  rho = xAx / xx  # More efficient than calling energy(e, x)

  # Precompute common factors
  xx_inv = 1 / xx
  factor1 = 2 * xx_inv
  factor2 = -4 * xx_inv * xx_inv

  grad_unnorm = Ax - rho * x
  H = factor1 * (e.A - rho * I) + factor2 * (x * grad_unnorm' + grad_unnorm * x')

  return H
end

# In-place version for better performance with large matrices
function hessian!(H::AbstractMatrix, e::RayleighQuotient, x::AbstractVector)
  xx = dot(x, x)
  @assert xx != 0 "Rayleigh quotient hessian undefined at x=0"

  # Compute Ax once and reuse
  Ax = e.A * x
  xAx = dot(x, Ax)
  rho = xAx / xx

  # Precompute common factors
  xx_inv = 1 / xx
  factor1 = 2 * xx_inv
  factor2 = -4 * xx_inv * xx_inv

  grad_unnorm = Ax - rho * x

  # Build H step by step to avoid broadcasting issues with UniformScaling
  # H = factor1 * (e.A - rho * I) + factor2 * (x * grad_unnorm' + grad_unnorm * x')

  # First: H = factor1 * e.A
  @. H = factor1 * e.A

  # Subtract factor1 * rho * I (identity matrix)
  for i in axes(H, 1)
    H[i, i] -= factor1 * rho
  end

  # Add factor2 * (x * grad_unnorm' + grad_unnorm * x')
  for i in axes(H, 1), j in axes(H, 2)
    H[i, j] += factor2 * (x[i] * grad_unnorm[j] + grad_unnorm[i] * x[j])
  end

  return H
end

# 3) Generalized Rayleigh quotient:  ρ(x) = (x'Ax) / (x'Bx), scale-invariant in x ≠ 0
#    (covers generalized eigenvalue problems Ax = λBx)
struct GeneralizedRayleighQuotient{T,M<:AbstractMatrix{T},N<:AbstractMatrix{T}} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
  B::N           # typically symmetric/hermitian positive definite
end
GeneralizedRayleighQuotient(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} =
  GeneralizedRayleighQuotient{T,typeof(A),typeof(B)}(A, B)
# ρ(x)
energy(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T
}) where {T} = begin
  num = dot(x, e.A * x)
  den = dot(x, e.B * x)
  @assert den != zero(T) "Generalized Rayleigh quotient undefined at x with x'Bx=0"
  num / den
end

# dimension
dimension(e::GeneralizedRayleighQuotient) = size(e.A, 1)

# ∇ρ(x) = 2 * ( (Ax)(x'Bx) - (Bx)(x'Ax) ) / (x'Bx)^2
gradient(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T
}) where {T} = begin
  Ax = e.A * x
  Bx = e.B * x
  xx_Bx = dot(x, Bx)
  x_Ax = dot(x, Ax)
  @assert xx_Bx != zero(T) "Generalized Rayleigh quotient gradient undefined at x with x'Bx=0"
  (2 / (xx_Bx * xx_Bx)) * (Ax .* xx_Bx .- Bx .* x_Ax)
end
# ∇²ρ(x) is more complicated; omitted for brevity
# You can implement it similarly to RayleighQuotient if needed.

# ∇²ρ(x) = 2 * ( (Ax)(x'Bx) - (Bx)(x'Ax) ) / (x'Bx)^2
#         + 2 * ( (Bx)(x'Ax) - (Ax)(x'Bx) ) / (x'Bx)^3 * 2 * Bx
#         + 2 * (A - ρ B) / (x'Bx)
#         - 4 * (Ax * (x'Bx) - Bx * (
#             x'Ax)) * (x'Bx) / (x'Bx)^4
function hessian(e::GeneralizedRayleighQuotient, x::AbstractVector)
  Ax = e.A * x
  Bx = e.B * x
  s = dot(x, Bx)
  @assert s != zero(eltype(x)) "Generalized Rayleigh quotient Hessian undefined at x with x'Bx = 0"

  rho = dot(x, Ax) / s
  M = e.A - rho * e.B

  factor1 = 2 / s
  factor2 = -4 / (s * s)

  # H = (2/s)(A - ρB) - (4/s^2)[ (A-ρB) x (Bx)' + (Bx) x' (A-ρB) ]
  H = factor1 * M + factor2 * (M * (x * Bx') + Bx * (x' * M))
  return H
end
# In-place version for better performance with large matrices
function hessian!(H::AbstractMatrix, e::GeneralizedRayleighQuotient,
  x::AbstractVector)
  throw(ErrorException(
    "In-place Hessian not implemented for GeneralizedRayleighQuotient"
  ))
end

# A helper function to compute the residual norm ||Ax - ρ(x)Bx||
function residual_norm(e::GeneralizedRayleighQuotient, x::AbstractVector)
  rho = energy(e, x)
  r = e.A * x .- rho * (e.B * x)
  return norm(r)
end


# ---------- Utilities ----------
# Promote to Symmetric if you know A is symmetric to avoid accidental double work.
as_symmetric(A) = Symmetric(A)  # no-op if already Symmetric

function normalize_M!(u::Vector{Float64}, M::AbstractMatrix)
  nu = sqrt(dot(u, M * u))
  @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
  u ./= nu
end
end # module

# Example usage:
using .Energies
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
E = Energies.QuadraticEnergy(A, b)
x = [0.5, 0.5]
E(x)                     # Evaluate energy
Energies.gradient(E, x)  # Compute gradient
Energies.hessian(E)      # Get Hessian (constant)

# Example usage of RayleighQuotient
RQ = Energies.RayleighQuotient(A)
RQ(x)                    # Evaluate Rayleigh quotient
Energies.gradient(RQ, x) # Compute gradient
Energies.hessian(RQ, x)   # Hessian implemented

# Test

# Make matrix exactly symmetric to avoid issues
A_sym = Symmetric(A)
RQ_sym = Energies.RayleighQuotient(A_sym)

# Check gradient first
@assert FiniteDiff.finite_difference_gradient(z -> Energies.energy(RQ_sym, z), x; absstep=1e-8) - Energies.gradient(RQ_sym, x) |> norm < 1E-6

# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - Energies.hessian(RQ_sym, x) |> norm < 1E-6

# Check Hessian in-place
H = zeros(length(x), length(x))
Energies.hessian!(H, RQ_sym, x)
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - H |> norm < 1E-6

# Check GeneralizedRayleighQuotient
B = [2.0 0.0; 0.0 1.0]
GRQ = Energies.GeneralizedRayleighQuotient(A_sym, Symmetric(B))
GRQ(x)                    # Evaluate generalized Rayleigh quotient
Energies.gradient(GRQ, x) # Compute gradient
Energies.hessian(GRQ, x)   # Hessian implemented
# Check gradient first
@assert FiniteDiff.finite_difference_gradient(z -> Energies.energy(GRQ, z), x; absstep=1e-8) - Energies.gradient(GRQ, x) |> norm < 1E-6
# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(GRQ, z), x; absstep=1e-8) - Energies.hessian(GRQ, x) |> norm < 1E-6






function FEM_Schroedinger(
  N::Int,
  m::Int=9,
  P::F1=(x -> exp(sqrt((x.data[1])^2 + (x.data[2])^2))),
  # also return RHS to solve source problem
  f::F2=(x -> 1.0)
) where {F1<:Function, F2<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model = CartesianDiscreteModel(domain, partition1; isperiodic=(false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags=["boundary"])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  U = TrialFESpace(VV, 0)
  a1(u, v) = ∫(∇(u) ⋅ ∇(v) + (x -> P(x)) * u * v)dΩ
  a2(u, v) = ∫(u * v)dΩ
  b(v) = ∫( (x -> f(x)) * v )dΩ
  K = assemble_matrix(a1, VV, U)
  M = assemble_matrix(a2, VV, U)
  b = assemble_vector(b, VV)
  g = GridapDistributed.compute_cell_graph(model)
  par = Metis.partition(g, m)
  elpar = create_elements_partition(par, m)
  create_overlapping_elements_partition!(elpar, g, m, 2)
  t1 = time()
  dofspar = create_dofs_partition(elpar, VV)
  elapsed = time() - t1
  println("create_dofs_partition finished in $elapsed seconds")
  return K, M, b, dofspar, U
end

function create_dofs_partition(
  elemsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace
)
  m = sp.fe_basis.trian.model
  dim = size(m.grid_topology.n_m_to_nface_to_mfaces, 2) - 1
  npars = length(elemsp)
  @debug "create nodesp"
  nodesp = [Vector{Int32}() for _ in 1:npars]
  Threads.@threads for ipar = 1:npars
    nodesp[ipar] = sort(unique(vcat([m.grid_topology.n_m_to_nface_to_mfaces[dim+1][el] for el in elemsp[ipar]]...)))
  end

  @debug "create freenodesp"
  freenodesp = copy(nodesp)
  Threads.@threads for ipar = 1:npars
    filter!(e -> e in sp.metadata.free_dof_to_node, nodesp[ipar])
  end

  @debug "create dofsp"
  reverse_map = zeros(Int32, maximum(sp.metadata.free_dof_to_node))
  for (node, freenode) in enumerate(sp.metadata.free_dof_to_node)
    reverse_map[freenode] = node
  end
  dofsp = [reverse_map[freenodesp[ipar]] for ipar = 1:npars]
  return dofsp
end

function create_elements_partition(partition::Vector{Int32}, npars::Integer) # Helper function from DDEigenlab
  nelems = length(partition)
  @debug nelems, length(partition)
  @assert nelems == length(partition)
  elemsp = [Vector{Int32}() for _ in 1:npars]
  for iel = 1:nelems
    push!(elemsp[partition[iel]], iel)
  end
  @debug nelems, sum(length.(elemsp))
  @assert nelems == sum(length.(elemsp))
  return elemsp
end

function create_overlapping_elements_partition!(elemsp, g, npars::Integer, ol) # Helper function from DDEigenlab
  for iol = 1:ol
    @debug "overlap" iol
    Threads.@threads for ipar = 1:npars
      tmp = copy(elemsp)
      elemsp[ipar] = sort(unique(vcat([g[:, i].nzind for i in tmp[ipar]]...)))
    end
  end
end


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
  idx_sub::AbstractVector
)
  throw(ErrorException("inf_step not implemented for $(typeof(e))"))
end

function inf_step(
  e::Energies.QuadraticEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  A, b = e.A, e.b

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
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  K, M = e.A, e.B

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

  @debug "Size of K_local: $(size(K_local))"
  @debug "Size of M_local: $(size(M_local))"
  K_local_sym = Symmetric(K_local)
  M_local_sym = Symmetric(M_local)
  F = cholesky(K_local_sym)  # ≈ A^{-1} preconditioner
  res = lobpcg(K_local_sym, M_local_sym, false, 1; P=F, tol=1e-8, maxiter=500)

  # Use helper function to reconstruct, then normalize
  x_new = reconstruct_implicit!(res.X[:, 1], u_cur, idx_sub)
  Energies.normalize_M!(x_new, M)

  return x_new
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
  x_new = B * eigvecs[:, 1]
  Energies.normalize_M!(x_new, M)
  return x_new
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::AbstractMatrix)
  w = u .- v
  return sqrt(dot(w, M * w))
end

"""
  reconstruct_implicit!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)

Reconstruct x_new from implicit basis [u_cur, e_j1, e_j2, ...] where e_j are standard
basis vectors, without normalization. This avoids explicitly constructing the basis matrix.
"""
function reconstruct_implicit!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)
  # The following is equivalent to x_new = [u_cur e_j1 e_j2 ...] * α
  # but avoids constructing e_j explicitly
  x_new = α[1] * u_cur  # Coefficient for current solution
  # Add contributions from standard basis vectors
  for (k, j) in pairs(idx_sub)
    x_new[j] += α[k+1]  # Add coefficient for e_j
  end
  return x_new
end# not implemented for abstract energy
function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subspaces::Vector{Vector{Int32}};
  kwargs...
)
  throw(ErrorException("var_dd not implemented for $(typeof(e))"))
end

function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{Vector{Int32}};
  maxiter::Int=50,
  tol::Float64=1e-8
)
  # TODO: Add sweep option, multiplicative version

  # Initial guess, no need to normalize apparently
  u_cur = ones(Energies.dimension(e))

  e_hist = Float64[]
  sol_hist = Vector{Vector{Float64}}()

  e_cur = e(u_cur)
  push!(e_hist, e_cur)
  push!(sol_hist, copy(u_cur))

  local_updates = zeros(size(u_cur, 1), m)  # preallocate for efficiency
  for n in 1:maxiter
    for i = 1:m
      u_next_i = inf_step(e, u_cur, subdomain_dofs[i])
      local_updates[:, i] = u_next_i
    end

    combined_matrix = hcat(u_cur, local_updates)
    u_new = combine_step(e, combined_matrix)

    resnorm = Energies.residual_norm(e, u_new)
    @printf(
      "Iteration %3d: Residual norm ≈ %12.6e energy = %12.6e\n",
      n, resnorm, e(u_new)
    )

    e_new = e(u_new)
    push!(e_hist, e_new)
    push!(sol_hist, copy(u_new))
    if abs(e_new - e_cur) < tol
      # TODO: Change to resnorm < tol or M-norm distance of u_new, u_cur < tol
      println("Converged at iteration $n with energy e = $e_new")
      return u_new, e_new, e_hist, sol_hist
    end

    u_cur = u_new
    e_cur = e_new
  end

  println("Reached maxiter=$maxiter with final Rayleigh quotient ≈ $e_cur")
  return u_cur, e_cur, e_hist, sol_hist
end



N = 20
m = 9
maxiter = 200
tol = 1e-10



# Schroedinger EVP FEM
K, M, b, part, U = FEM_Schroedinger(N, m)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
u_approx, lambda_approx, lambda_history, solutions = var_dd(
  energy_eigen_fem,
  part,
  maxiter=maxiter,
  tol=tol
)
println("Final approximate eigenvalue = $lambda_approx")
exact_sol = eigs(K, M, nev=1, which=:SM)
println("Exact eigenvalue = $(exact_sol[1][1])")
@assert abs(lambda_approx - exact_sol[1][1]) < 1e-6
# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  "eigen_solution",
  cellfields = ["u_approx" => FEFunction(U, u_approx)]
)



# Poisson problem FEM
K, M, b, part, U = FEM_Schroedinger(N, m, (x -> 0.0), (x -> 1.0))
energy_poisson_fem = Energies.QuadraticEnergy(K, b, 0.0)
# direct solve
u_poisson_direct = K \ b
# var_dd solve
u_poisson, E_poisson, E_hist, sols = var_dd(energy_poisson_fem, part, maxiter=maxiter, tol=tol)
# difference
@assert norm(u_poisson - u_poisson_direct) < 1e-4
E_poisson = Energies.energy(energy_poisson_fem, u_poisson)
println("Poisson energy = $E_poisson")
# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  "poisson_solution",
  cellfields = [
    "u_poisson" => FEFunction(U, u_poisson),
    "u_poisson_direct" => FEFunction(U, u_poisson_direct)
  ]
)
