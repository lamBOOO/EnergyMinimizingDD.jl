"""
    GrossPitaevskiiRayleighQuotient(K, M, beta, quartic, cubic_gradient)

Scale-invariant Gross--Pitaevskii objective

    R_beta(u) = (u'Ku)/(u'Mu) + (beta/2) quartic(u)/(u'Mu)^2.

It is twice the physical GP energy of the M-normalized state.
"""
struct GrossPitaevskiiRayleighQuotient{
  T,MK<:AbstractMatrix{T},MM<:AbstractMatrix{T},FQ,FG
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
  quartic,
  cubic_gradient,
) where {T}
  beta_T = T(beta)
  beta_T >= zero(T) || throw(ArgumentError("beta must be nonnegative"))
  size(K) == size(M) || throw(DimensionMismatch("K and M must have equal size"))
  size(K, 1) == size(K, 2) || throw(DimensionMismatch("K and M must be square"))
  return GrossPitaevskiiRayleighQuotient{
    T,typeof(K),typeof(M),typeof(quartic),typeof(cubic_gradient)
  }(
    K, M, beta_T, quartic, cubic_gradient
  )
end

dimension(e::GrossPitaevskiiRayleighQuotient) = size(e.K, 1)

function _gp_mass(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  Mu = e.M * u
  mass = dot(u, Mu)
  mass > eps(real(one(eltype(u)))) ||
    throw(ArgumentError("GP quotient is undefined at a zero-mass vector"))
  return Mu, mass
end

function energy(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  Mu, mass = _gp_mass(e, u)
  return dot(u, e.K * u) / mass + (e.beta / 2) * e.quartic(u) / mass^2
end

function physical_energy(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  return energy(e, u) / 2
end

function gradient(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  Ku = e.K * u
  Mu, mass = _gp_mass(e, u)
  kinetic = dot(u, Ku)
  quartic = e.quartic(u)
  return 2 .* (
    Ku ./ mass .+ e.beta .* e.cubic_gradient(u) ./ mass^2 .-
    (kinetic / mass^2 + e.beta * quartic / mass^3) .* Mu
  )
end

function chemical_potential(
  e::GrossPitaevskiiRayleighQuotient, u::AbstractVector
)
  _, mass = _gp_mass(e, u)
  return dot(u, e.K * u) / mass + e.beta * e.quartic(u) / mass^2
end

function projected_residual(
  e::GrossPitaevskiiRayleighQuotient, u::AbstractVector
)
  _, mass = _gp_mass(e, u)
  v = u ./ sqrt(mass)
  lambda = chemical_potential(e, v)
  return e.K * v .+ e.beta .* e.cubic_gradient(v) .- lambda .* (e.M * v)
end

function residual_norm(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  return norm(projected_residual(e, u))
end
