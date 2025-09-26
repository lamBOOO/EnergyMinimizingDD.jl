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

# ---------- Concrete energies ----------

# 1) Quadratic energy:  E(x) = 1/2 x'Ax - b'x + c
#    (covers linear systems and least-squares normal equations)
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

# ∇E(x) = Ax - b  (if A symmetric; if not, this is gradient of 1/2 x'(A+A')x - b'x)
gradient(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  e.A * x .- e.b

# ∇²E(x) = A (constant)
hessian(e::QuadraticEnergy{T}) where {T} = e.A
hessian(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = hessian(e)

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



# ---------- Utilities ----------
# Promote to Symmetric if you know A is symmetric to avoid accidental double work.
as_symmetric(A) = Symmetric(A)  # no-op if already Symmetric
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





P(x) = exp(sqrt((x.data[1])^2 + (x.data[2])^2))

function Setup_FEM(N::Int, m::Int=9) # Discretizing the domain, building mass and stiffness matrix, specifying the overlapping domains
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
  K = assemble_matrix(a1, VV, U)
  M = assemble_matrix(a2, VV, U)
  g = GridapDistributed.compute_cell_graph(model)
  par = Metis.partition(g, m)
  elpar = create_elements_partition(par, m)
  create_overlapping_elements_partition!(elpar, g, m, 2)
  t1 = time()
  dofspar = create_dofs_partition(elpar, VV)
  elapsed = time() - t1
  println("dofspar needs $elapsed seconds ")
  return K, M, dofspar, VV
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

# 2) Rayleigh quotient
function R(u::Vector{Float64}, K::AbstractMatrix, M::AbstractMatrix)
  numerator = dot(u, K * u)
  denominator = dot(u, M * u)
  return numerator / denominator
end

# 3) Normalization in the M-norm
function normalize_M!(u::Vector{Float64}, M::AbstractMatrix)
  nu = sqrt(dot(u, M * u))
  @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
  u ./= nu
end

function pu_matrices(dofsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace) #pu as in Eigenlab
  npars = length(dofsp)
  Ri = Vector{SparseMatrixCSC}(undef, npars)
  Di = Vector{SparseMatrixCSC}(undef, npars)
  Threads.@threads for ipar = 1:npars
    Ri[ipar] = spzeros(length(dofsp[ipar]), sp.nfree)
    Di[ipar] = spzeros(length(dofsp[ipar]), length(dofsp[ipar]))
    for idof = 1:length(dofsp[ipar])
      Ri[ipar][idof, dofsp[ipar][idof]] = 1
      Di[ipar][idof, idof] = 1 / sum(map(p -> dofsp[ipar][idof] in p, dofsp))
    end
  end
  return Ri, Di
end

function coarse_space_corr(dofsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace)
  Ri, Di = pu_matrices(dofsp, sp)
  n = size(Di, 1) # no. subdomains
  m = size(Ri[1], 2) # no. of DOFS
  Z = zeros(m, n)
  for i = 1:n
    Z[:, i] = (Ri[i]' * Di[i] * Ri[i]) * ones(m) #Z as in Nicolaides in DD Book
  end
  return Z
end

function inf_step(
  e::Energies.AbstractEnergy{Float64},
  u_current::Vector{Float64},
  idx_sub::AbstractVector
)
  throw(ErrorException("inf_step not implemented for $(typeof(e))"))
end

# 5) Inf step on subspace D_i
"""
  inf_step(u_current, K, M, idx_sub)

Calcualted x_new = argmin_{x ∈ span{u_current, e_j, j ∈ idx_sub} \\ {0}} (x' K x)/(x' M x)
where {e_j} are standard basis vectors. The output is re-normalized in the M-norm.
"""
function inf_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_current::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  K, M = e.A, e.B

  # Extract relevant rows/columns from K and M
  if length(idx_sub) > 0
    # Build the local matrices more efficiently
    K_local = zeros(localdim, localdim)
    M_local = zeros(localdim, localdim)

    # First row/column: u_current' * K/M * [u_current, e_j1, e_j2, ...]
    K_u = K * u_current
    M_u = M * u_current

    K_local[1, 1] = dot(u_current, K_u)  # u' * K * u
    M_local[1, 1] = dot(u_current, M_u)  # u' * M * u

    # First row/column: u_current' * K/M * e_j
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
    K_local = reshape([dot(u_current, K * u_current)], 1, 1)
    M_local = reshape([dot(u_current, M * u_current)], 1, 1)
  end

  @debug "Size of K_local: $(size(K_local))"
  @debug "Size of M_local: $(size(M_local))"
  K_local_sym = Symmetric(K_local)
  M_local_sym = Symmetric(M_local)
  F = cholesky(K_local_sym)  # ≈ A^{-1} preconditioner
  res = lobpcg(K_local_sym, M_local_sym, false, 1; P=F, tol=1e-8, maxiter=500)

  # The following is equivelant to Rayleigh Ritz procedure
  # x_new = [u_current e_j1 e_j2 ...] * α_min
  # where α_min is the eigenvector associated with the smallest eigenvalue
  # of the generalized eigenvalue problem in the subspace
  # (B' K B) α = λ (B' M B) α
  # with B = [u_current e_j1 e_j2 ...]
  # but avoids constructing e_j explicitly
  x_new = res.X[:, 1][1] * u_current  # Coefficient for current solution
  # Add contributions from standard basis vectors
  for (k, j) in pairs(idx_sub)
    x_new[j] += res.X[:, 1][k+1]  # Add coefficient for e_j
  end

  normalize_M!(x_new, M)

  return x_new
end

# 6) Combine step
"""
  combine_step(u_collection, K, M)

Given a collection of M-normalized (not necessarily mutually orthogonal) vectors
`{u_i}` this forms the matrix `B = [u_1 ... u_p]`, computes an orthonormal (in
the Euclidean sense) basis `Q` of its column space via QR, and then solves the
reduced generalized eigenproblem

  (Q' K Q) α = λ (Q' M Q) α

returning the vector `x_new = Q α_min` associated with the smallest Rayleigh
quotient restricted to span(B). The output is re-normalized in the M-norm.

Mathematically this performs the exact minimization

  x_new = argmin_{x ∈ span(u_collection) \\ {0}} (x' K x)/(x' M x).

Returns the updated vector `x_new` with `x_new' * M * x_new = 1`.
"""
function combine_step(u_collection::Matrix{Float64},
  K::AbstractMatrix, M::AbstractMatrix)
  t_qr_start = time()

  B = Matrix(qr(u_collection).Q)
  t_qr = time() - t_qr_start

  t_matrices_start = time()
  # Optimized matrix multiplications using temporary arrays and mul!
  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices
  temp_K = Matrix{Float64}(undef, N, localdim)
  temp_M = Matrix{Float64}(undef, N, localdim)
  K_local = Matrix{Float64}(undef, localdim, localdim)
  M_local = Matrix{Float64}(undef, localdim, localdim)

  # Use mul! for in-place operations
  mul!(temp_K, K, B)
  mul!(temp_M, M, B)
  mul!(K_local, B', temp_K)
  mul!(M_local, B', temp_M)

  t_matrices = time() - t_matrices_start

  t_eigen_start = time()
  eigvals, eigvecs = eigen(K_local, M_local)
  i_min = argmin(eigvals)
  α_min = eigvecs[:, i_min]
  t_eigen = time() - t_eigen_start

  # println("eigen(K_local): ",eigen(K_local).values)
  # println("eigen(M_local): ",eigen(M_local).values)
  # println(α_min)

  t_finalize_start = time()
  x_new = B * α_min
  normalize_M!(x_new, M)
  t_finalize = time() - t_finalize_start

  @debug "combine_step breakdown: QR=$t_qr, matrices=$t_matrices, eigen=$t_eigen, finalize=$t_finalize"
  return x_new
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::AbstractMatrix)
  w = u .- v
  return sqrt(dot(w, M * w))
end

# not implemented for abstract energy
function ddm_eigen_solver(
  e::Energies.AbstractEnergy{Float64},
  subspaces::Vector{Vector{Int32}};
  kwargs...
)
  throw(ErrorException("ddm_eigen_solver not implemented for $(typeof(e))"))
end

function ddm_eigen_solver(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  subspaces::Vector{Vector{Int32}};
  maxiter::Int=50,
  tol::Float64=1e-8
)

  # TODO: Add sweep option
  setup_time = time()
  # K, M, subspaces, fesp = Setup_FEM(N, m)
  K = e.A
  M = e.B
  elapsed_setup = time() - setup_time
  println("$elapsed_setup seconds needed for setup")

  # Initial guess
  u_cur = ones((N - 1)^2)
  normalize_M!(u_cur, M)

  lambda_history = Float64[]
  solutions = Vector{Vector{Float64}}()

  λ_cur = R(u_cur, K, M)
  push!(lambda_history, λ_cur)
  push!(solutions, copy(u_cur))

  for n in 1:maxiter
    local_updates = zeros(size(u_cur, 1), m + 1)
    for i = 1:m
      u_next_i = inf_step(e, u_cur, subspaces[i])
      local_updates[:, i+1] = u_next_i
    end

    combined_matrix = hcat(u_cur, local_updates)
    u_new = combine_step(combined_matrix, K, M)

    @printf("Iteration %3d: Residual norm ≈ %12.6e energy = %12.6e\n", n, norm(K * u_new - λ_cur * M * u_new), e(u_new))

    # TODO: Needed?
    if dot(u_new, u_cur) < 0
      u_new .*= -1.0
    end

    λ_new = R(u_new, K, M)
    push!(lambda_history, λ_new)
    push!(solutions, copy(u_new))
    if abs(λ_new - λ_cur) < tol
      println("Converged at iteration $n with eigenvalue λ = $λ_new")
      return u_new, λ_new, lambda_history, solutions
    end

    u_cur = u_new
    λ_cur = λ_new
  end

  println("Reached maxiter=$maxiter with final Rayleigh quotient ≈ $λ_cur")
  return u_cur, λ_cur, lambda_history, solutions
end



N = 20
m = 9
maxiter = 200
tol = 1e-10
K, M, part = Setup_FEM(N, m)

energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)

u_approx, lambda_approx, lambda_history, solutions = ddm_eigen_solver(
  energy_eigen_fem,
  part,
  maxiter=maxiter,
  tol=tol
)
println("Final approximate eigenvalue = $lambda_approx")
exact_sol = eigs(K, M, nev=1, which=:SM)
println("Exact eigenvalue = $(exact_sol[1][1])")
@assert abs(lambda_approx - exact_sol[1][1]) < 1e-6
