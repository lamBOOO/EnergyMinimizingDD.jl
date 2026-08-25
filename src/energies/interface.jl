"""
    AbstractEnergy{T}

Interface for a scalar-valued energy. Subtypes implement `energy`, `gradient`,
and `dimension`; second-order models additionally implement `hessian`.
"""
abstract type AbstractEnergy{T} end

(e::AbstractEnergy)(x) = energy(e, x)

energy(e::AbstractEnergy, x) = throw(MethodError(energy, (e, x)))

gradient(e::AbstractEnergy, x) = throw(MethodError(gradient, (e, x)))

hessian(e::AbstractEnergy, x) = throw(MethodError(hessian, (e, x)))

dimension(e::AbstractEnergy) = throw(MethodError(dimension, (e,)))

function gradient!(storage, e::AbstractEnergy, x)
  copyto!(storage, gradient(e, x))
  return storage
end

function hessian!(storage, e::AbstractEnergy, x)
  copyto!(storage, hessian(e, x))
  return storage
end

residual_norm(e::AbstractEnergy, x) = norm(gradient(e, x))

as_symmetric(A) = Symmetric(A)

function normalize_M!(u::AbstractVector, M::AbstractMatrix)
  norm_squared = real(dot(u, M * u))
  norm_squared > 100 * eps(real(one(eltype(u)))) ||
    throw(ArgumentError("cannot normalize a vector with near-zero M-norm"))
  u ./= sqrt(norm_squared)
  return u
end
