"""
    QuadraticEnergy(A, b; c=0)

The energy `E(x) = 1/2 x' A x - b' x + c`. `A` is assumed symmetric.
"""
struct QuadraticEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractEnergy{T}
  A::M
  b::V
  c::T
end

function QuadraticEnergy(
  A::AbstractMatrix{T}, b::AbstractVector{T}; c::T=zero(T)
) where {T}
  size(A, 1) == size(A, 2) == length(b) ||
    throw(DimensionMismatch("A must be square with size(A, 1) == length(b)"))
  return QuadraticEnergy{T,typeof(A),typeof(b)}(A, b, c)
end

function energy(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T}
  return T(0.5) * dot(x, e.A * x) - dot(e.b, x) + e.c
end

dimension(e::QuadraticEnergy) = size(e.A, 1)

gradient(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = e.A * x .- e.b

hessian(e::QuadraticEnergy) = e.A
hessian(e::QuadraticEnergy, x::AbstractVector) = hessian(e)

"""
    quadratic_model(e, u)

Return the second-order Taylor model of `e` at `u` as a `QuadraticEnergy`.
The constant is retained so that the model agrees with `e` at `u`; it does
not affect any minimization step.
"""
function quadratic_model(e::AbstractEnergy{T}, u::AbstractVector{T}) where {T}
  length(u) == dimension(e) || throw(
    DimensionMismatch("iterate length must equal the energy dimension"),
  )
  A = hessian(e, u)
  g = gradient(e, u)
  Au = A * u
  b = Au - g
  c = energy(e, u) + T(0.5) * dot(u, Au) - dot(g, u)
  return QuadraticEnergy(A, b; c=c)
end

"""
    LinearRegressionEnergy(A, b)

Least-squares energy `E(x) = norm(A*x - b)^2`.
"""
struct LinearRegressionEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractEnergy{T}
  A::M
  b::V
end

function energy(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T}
  residual = e.A * x .- e.b
  return dot(residual, residual)
end

dimension(e::LinearRegressionEnergy) = size(e.A, 2)

function gradient(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T}
  return 2 .* (e.A' * (e.A * x .- e.b))
end

hessian(e::LinearRegressionEnergy) = 2 .* (e.A' * e.A)
hessian(e::LinearRegressionEnergy, x::AbstractVector) = hessian(e)
