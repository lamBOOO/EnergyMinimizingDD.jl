module Energies

export AbstractEnergy, QuadraticEnergy, RayleighQuotient, GeneralizedRayleighQuotient, GrossPitaevskiiRayleighQuotient, NonlinearEnergy, LinearRegressionEnergy
export physical_energy, chemical_potential, projected_residual

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
dimension(e::AbstractEnergy) =
  error("dimension not implemented for $(typeof(e))")

# Optionally provide defaults via AD; otherwise keep them abstract.
gradient(e::AbstractEnergy, x) =
  error("gradient not implemented for $(typeof(e))")
hessian(e::AbstractEnergy, x) =
  error("hessian not implemented for $(typeof(e))")

# In-place variants are optional but nice for performance
function gradient!(g, e::AbstractEnergy, x)
  g .= gradient(e, x)       # default: out-of-place -> in-place
  return g
end
function hessian!(H, e::AbstractEnergy, x)
  H .= hessian(e, x)
  return H
end

# Default implementation: residual norm is the gradient norm for all energy types
residual_norm(e::AbstractEnergy, x) = norm(gradient(e, x))

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
struct QuadraticEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractEnergy{T}
  A::M           # can be Dense, Sparse, or Symmetric wrapper
  b::V
  c::T
end

# Make a convenient constructor; wrap A as Symmetric if you know it.
QuadraticEnergy(
  A::AbstractMatrix{T},
  b::AbstractVector{T};
  c::T = zero(T),
) where {T} = QuadraticEnergy{T,typeof(A),typeof(b)}(A, b, c)

# E(x)
energy(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  T(0.5) * dot(x, e.A * x) - dot(e.b, x) + e.c

# dimension
dimension(e::QuadraticEnergy) = size(e.A, 1)

# ∇E(x) = Ax - b  (if A symmetric; if not, this is gradient of 1/2 x'(A+A')x - b'x)
gradient(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = e.A * x .- e.b

# ∇²E(x) = A (constant)
hessian(e::QuadraticEnergy{T}) where {T} = e.A
hessian(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = hessian(e)




# 2) Rayleigh quotient:  ρ(x) = (x'Ax) / (x'x), scale-invariant in x ≠ 0
struct RayleighQuotient{T,M<:AbstractMatrix{T}} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
end

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
  (2 / xx) * (Ax .- x .* (xAx / xx))
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
  H =
    factor1 * (e.A - rho * I) + factor2 * (x * grad_unnorm' + grad_unnorm * x')

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
struct GeneralizedRayleighQuotient{
  T,
  M<:AbstractMatrix{T},
  N<:AbstractMatrix{T},
} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
  B::N           # typically symmetric/hermitian positive definite
end
# ρ(x)
energy(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T}) where {T} =
  begin
    num = dot(x, e.A * x)
    den = dot(x, e.B * x)
    @assert den != zero(T) "Generalized Rayleigh quotient undefined at x with x'Bx=0"
    num / den
  end

# dimension
dimension(e::GeneralizedRayleighQuotient) = size(e.A, 1)

# ∇ρ(x) = 2 * ( (Ax)(x'Bx) - (Bx)(x'Ax) ) / (x'Bx)^2
gradient(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T}) where {T} =
  begin
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
function hessian!(
  H::AbstractMatrix,
  e::GeneralizedRayleighQuotient,
  x::AbstractVector,
)
  throw(
    ErrorException(
      "In-place Hessian not implemented for GeneralizedRayleighQuotient",
    ),
  )
end

"""
    GrossPitaevskiiRayleighQuotient(K, M, beta, quartic, cubic_gradient)

Scale-invariant Gross--Pitaevskii objective

    R_beta(u) = (u'Ku)/(u'Mu) + (beta/2) quartic(u)/(u'Mu)^2,

where `quartic(u) = integral(u_h^4)` and `cubic_gradient(u)` has entries
`integral(u_h^3 phi_i)`. Thus `R_beta(u)` is twice the physical GP energy of
the M-normalized state. At a normalized stationary point,

    K*u + beta*cubic_gradient(u) = lambda*M*u.
"""
struct GrossPitaevskiiRayleighQuotient{
  T,
  MK<:AbstractMatrix{T},
  MM<:AbstractMatrix{T},
  FQ<:Function,
  FG<:Function,
} <: AbstractEnergy{T}
  K::MK
  M::MM
  beta::T
  quartic::FQ
  cubic_gradient::FG
end

function GrossPitaevskiiRayleighQuotient(
  K::AbstractMatrix{T},
  M::AbstractMatrix{T},
  beta::Real,
  quartic::FQ,
  cubic_gradient::FG,
) where {T,FQ<:Function,FG<:Function}
  beta_T = T(beta)
  beta_T >= zero(T) || throw(ArgumentError("beta must be nonnegative"))
  size(K) == size(M) || throw(DimensionMismatch("K and M must have equal size"))
  return GrossPitaevskiiRayleighQuotient{
    T,typeof(K),typeof(M),FQ,FG,
  }(K, M, beta_T, quartic, cubic_gradient)
end

dimension(e::GrossPitaevskiiRayleighQuotient) = size(e.K, 1)

function energy(
  e::GrossPitaevskiiRayleighQuotient{T},
  u::AbstractVector{T},
) where {T}
  Mu = e.M * u
  mass = dot(u, Mu)
  mass > eps(T) || throw(ArgumentError("GP quotient is undefined at a zero-mass vector"))
  return dot(u, e.K * u) / mass +
         (e.beta / 2) * e.quartic(u) / mass^2
end

physical_energy(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector) =
  energy(e, u) / 2

function gradient(
  e::GrossPitaevskiiRayleighQuotient{T},
  u::AbstractVector{T},
) where {T}
  Ku = e.K * u
  Mu = e.M * u
  mass = dot(u, Mu)
  mass > eps(T) || throw(
    ArgumentError("GP quotient gradient is undefined at a zero-mass vector"),
  )
  kinetic = dot(u, Ku)
  quartic = e.quartic(u)
  cubic = e.cubic_gradient(u)
  return 2 .* (
    Ku ./ mass .+
    e.beta .* cubic ./ mass^2 .-
    (kinetic / mass^2 + e.beta * quartic / mass^3) .* Mu
  )
end

function chemical_potential(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  mass = dot(u, e.M * u)
  mass > eps(eltype(u)) ||
    throw(ArgumentError("chemical potential is undefined at zero mass"))
  return dot(u, e.K * u) / mass + e.beta * e.quartic(u) / mass^2
end

function projected_residual(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  mass = dot(u, e.M * u)
  mass > eps(eltype(u)) ||
    throw(ArgumentError("GP residual is undefined at zero mass"))
  v = u ./ sqrt(mass)
  lambda = chemical_potential(e, v)
  return e.K * v .+ e.beta .* e.cubic_gradient(v) .- lambda .* (e.M * v)
end

residual_norm(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector) =
  norm(projected_residual(e, u))



# 4) Generic Nonlinear Energy: E(u) for general nonlinear problems
#    Can represent PDE problems like p-Laplacian: E(u) = ∫(|∇u|^p/p)dΩ - ∫f*u dΩ
#    Or simple algebraic systems: E(x) = ½||F(x)||² where F(x) = 0 is the root problem
struct NonlinearEnergy{T,F1<:Function,F2<:Function} <: AbstractEnergy{T}
  name::String   # Descriptive name (e.g., "p-Laplacian", "Circle-Cubic System")
  assembler::F1  # Function that assembles the energy given u: (u) -> energy_value
  grad_assembler::F2  # Function that assembles the gradient: (u) -> gradient_vector
  N::Int        # Problem dimension
end

NonlinearEnergy(
  name::String,
  assembler::F1,
  grad_assembler::F2,
  N::Int;
) where {F1,F2} =
  NonlinearEnergy{Float64,F1,F2}(name, assembler, grad_assembler, N)

# E(u) - energy evaluation
energy(e::NonlinearEnergy{T}, u::AbstractVector{T}) where {T} = e.assembler(u)

# dimension
dimension(e::NonlinearEnergy) = e.N

# ∇E(u) - gradient evaluation
gradient(e::NonlinearEnergy{T}, u::AbstractVector{T}) where {T} =
  e.grad_assembler(u)



# 6) Linear Regression Energy: E(x) = ||Ax - b||² for least squares problems
#    Minimizing this energy leads to solving the normal equations A'Ax = A'b
struct LinearRegressionEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractEnergy{T}
  A::M           # Design matrix (m × n where m ≥ n typically)
  b::V           # Observation vector (length m)
end

# E(x) = ||Ax - b||² = (Ax - b)ᵀ(Ax - b)
energy(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T} = begin
  residual = e.A * x .- e.b
  return dot(residual, residual)
end

# dimension (number of parameters to fit)
dimension(e::LinearRegressionEnergy) = size(e.A, 2)

# ∇E(x) = 2AᵀAx - 2Aᵀb = 2Aᵀ(Ax - b)
gradient(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T} = begin
  residual = e.A * x .- e.b
  return 2 * (e.A' * residual)
end

# ∇²E(x) = 2AᵀA (constant Hessian)
hessian(e::LinearRegressionEnergy{T}) where {T} = 2 * (e.A' * e.A)
hessian(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T} =
  hessian(e)




# ---------- Utilities ----------
# Promote to Symmetric if you know A is symmetric to avoid accidental double work.
as_symmetric(A) = Symmetric(A)  # no-op if already Symmetric

function normalize_M!(u::Vector{Float64}, M::AbstractMatrix)
  nu = sqrt(dot(u, M * u))
  @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
  u ./= nu
end
end # module
