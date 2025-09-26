# # TODO
# abstract type Energy

#   solve_1st_order()

# end

# struct QuadraticForm(Q,b) <: Energy
#   solve_1st_order(u_cur)
#     # Solve the quadratic form minimization problem
#     # min 0.5*u'*Q*u + b'*u
#     u_next = -Q \ b
# end

# struct LinearSystem(A,B) <: Energy
#   solve_1st_order()
#     # Solve Ax = b
#     x = A \ B
# end
# return x

# struct RayleighQuotient <: energy
#   A :: AbstractArray
#   B :: AbstractArray
#   solve_1st_order(K,M,u_cur,subspaces,i,nev)
#     # Solve the generalized eigenvalue problem Kx = λMx
#     # using the inverse power method
#     u_next_i = inf_step(u_cur, K, M, subspaces[i], nev)
# end





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
hessian(e::AbstractEnergy, x)  = error("hessian not implemented for $(typeof(e))")

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
struct QuadraticEnergy{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractEnergy{T}
    A::M           # can be Dense, Sparse, or Symmetric wrapper
    b::V
    c::T
end

# Make a convenient constructor; wrap A as Symmetric if you know it.
QuadraticEnergy(A::AbstractMatrix{T}, b::AbstractVector{T}; c::T=zero(T)) where {T} =
    QuadraticEnergy{T, typeof(A), typeof(b)}(A, b, c)

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
struct RayleighQuotient{T, M<:AbstractMatrix{T}} <: AbstractEnergy{T}
    A::M           # typically symmetric/hermitian for real-valued quotient
end

RayleighQuotient(A::AbstractMatrix{T}) where {T} =
    RayleighQuotient{T, typeof(A)}(A)

# ρ(x)
energy(e::RayleighQuotient{T}, x::AbstractVector{T}) where {T} = begin
    num = dot(x, e.A * x)
    den = dot(x, x)
    @assert den != zero(T) "Rayleigh quotient undefined at x=0"
    num / den
end

# ∇ρ(x) = 2 * ( (Ax)(x⋅x) - x(x⋅Ax) ) / (x⋅x)^2
gradient(e::RayleighQuotient{T}, x::AbstractVector{T}) where {T} = begin
    Ax  = e.A * x
    xx  = dot(x, x)
    xAx = dot(x, Ax)
    @assert xx != zero(T) "Rayleigh quotient gradient undefined at x=0"
    (2 / (xx*xx)) * (Ax .* xx .- x .* xAx)
end

# A helper: in-place gradient for performance
function gradient!(g::AbstractVector, e::RayleighQuotient, x::AbstractVector)
    Ax  = e.A * x
    xx  = dot(x, x)
    xAx = dot(x, Ax)
    @assert xx != 0 "Rayleigh quotient gradient undefined at x=0"
    # g = 2 * (Ax*xx - x*xAx) / xx^2
    @. g = 2 * (Ax*xx - x*xAx) / (xx*xx)
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

# Check if Hessian correct
using FiniteDiff
using LinearAlgebra

# Make matrix exactly symmetric to avoid issues
A_sym = Symmetric(A)
RQ_sym = Energies.RayleighQuotient(A_sym)

# Check gradient first
FiniteDiff.finite_difference_gradient(z -> Energies.energy(RQ_sym, z), x; absstep=1e-8) - Energies.gradient(RQ_sym, x) |> norm |> println

# Check Hessian
FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - Energies.hessian(RQ_sym, x) |> norm |> println

# Check Hessian in-place
H = zeros(length(x), length(x))
Energies.hessian!(H, RQ_sym, x)
FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - H |> norm |> println





