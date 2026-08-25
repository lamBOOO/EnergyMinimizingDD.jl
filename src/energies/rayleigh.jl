"""
    RayleighQuotient(A[, B])

The quotient `rho(x) = (x' A x) / (x' B x)`. Omitting `B` uses the identity.
`A` and `B` are assumed symmetric and `B` positive definite.

`GeneralizedRayleighQuotient` is a compatibility alias for this same type.
"""
struct RayleighQuotient{T,MA<:AbstractMatrix{T},MB<:AbstractMatrix{T}} <:
       AbstractEnergy{T}
  A::MA
  B::MB
end

function RayleighQuotient(A::AbstractMatrix{T}) where {T}
  size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
  B = Diagonal(fill(one(T), size(A, 1)))
  return RayleighQuotient(A, B)
end

const GeneralizedRayleighQuotient = RayleighQuotient

dimension(e::RayleighQuotient) = size(e.A, 1)

function _rayleigh_parts(e::RayleighQuotient, x::AbstractVector)
  Ax = e.A * x
  Bx = e.B * x
  denominator = dot(x, Bx)
  iszero(denominator) &&
    throw(ArgumentError("Rayleigh quotient is undefined when x'B*x is zero"))
  numerator = dot(x, Ax)
  return Ax, Bx, numerator, denominator
end

function energy(e::RayleighQuotient, x::AbstractVector)
  _, _, numerator, denominator = _rayleigh_parts(e, x)
  return numerator / denominator
end

function gradient(e::RayleighQuotient, x::AbstractVector)
  Ax, Bx, numerator, denominator = _rayleigh_parts(e, x)
  rho = numerator / denominator
  return (2 / denominator) .* (Ax .- rho .* Bx)
end

function hessian(e::RayleighQuotient, x::AbstractVector)
  _, Bx, numerator, denominator = _rayleigh_parts(e, x)
  rho = numerator / denominator
  shifted = e.A - rho * e.B
  shifted_x = shifted * x
  return (2 / denominator) .* shifted .-
         (4 / denominator^2) .* (shifted_x * Bx' + Bx * shifted_x')
end
