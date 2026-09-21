"""
    GrossPitaevskiiRayleighQuotient(
      K, M, beta, quartic, cubic_gradient; density_matrix=nothing
    )

Scale-invariant Gross--Pitaevskii objective

    R_beta(u) = (u'Ku)/(u'Mu) + (beta/2) quartic(u)/(u'Mu)^2.

It is twice the physical GP energy of the M-normalized state. When supplied,
`density_matrix(u)` returns the matrix `D(u)` satisfying
`D(u) * u == cubic_gradient(u)` and is used by the SCF solvers. The matrix can
be omitted for backwards compatibility; in that case its action is recovered
from the cubic map by polarization.
"""
struct GrossPitaevskiiRayleighQuotient{
  T,
  MK<:AbstractMatrix{T},
  MM<:AbstractMatrix{T},
  FQ,
  FG,
  FD,
} <: AbstractEnergy{T}
  K::MK
  M::MM
  beta::T
  quartic::FQ
  cubic_gradient::FG
  density_matrix::FD
end

function GrossPitaevskiiRayleighQuotient(
  K::AbstractMatrix{T},
  M::AbstractMatrix{T},
  beta::Real,
  quartic,
  cubic_gradient,
  ;
  density_matrix = nothing,
) where {T}
  beta_T = T(beta)
  beta_T >= zero(T) || throw(ArgumentError("beta must be nonnegative"))
  size(K) == size(M) || throw(DimensionMismatch("K and M must have equal size"))
  size(K, 1) == size(K, 2) || throw(DimensionMismatch("K and M must be square"))
  return GrossPitaevskiiRayleighQuotient{
    T,
    typeof(K),
    typeof(M),
    typeof(quartic),
    typeof(cubic_gradient),
    typeof(density_matrix),
  }(
    K,
    M,
    beta_T,
    quartic,
    cubic_gradient,
    density_matrix,
  )
end

dimension(e::GrossPitaevskiiRayleighQuotient) = size(e.K, 1)

"""
Quadratic Taylor model of the physical Gross--Pitaevskii energy at a
mass-normalized state. The model is minimized only over tangent corrections;
the full nonlinear GP quotient remains responsible for the combination step.
"""
struct GrossPitaevskiiTangentQuadraticModel{
  T,
  MH<:AbstractMatrix{T},
  MM<:AbstractMatrix{T},
  V<:AbstractVector{T},
} <: AbstractEnergy{T}
  H::MH
  M::MM
  u::V
  residual::V
  chemical_potential::T
end

dimension(model::GrossPitaevskiiTangentQuadraticModel) = length(model.u)

"""
Riemannian quadratic model of the physical Gross--Pitaevskii energy on the
unit-mass sphere. In contrast to `GrossPitaevskiiTangentQuadraticModel`, its
Hessian includes the curvature term `-chemical_potential*M` arising from
normalizing a tangent step.
"""
struct GrossPitaevskiiProjectedNewtonModel{
  T,
  MH<:AbstractMatrix{T},
  MM<:AbstractMatrix{T},
  MK<:AbstractMatrix{T},
  V<:AbstractVector{T},
} <: AbstractEnergy{T}
  H_lagrangian::MH
  M::MM
  metric::MK
  u::V
  residual::V
  chemical_potential::T
end

dimension(model::GrossPitaevskiiProjectedNewtonModel) = length(model.u)

"""
    tangent_quadratic_model(e, u)

Build the second-order model of the physical GP energy at `u`, restricted to
the linearized mass constraint. For the normalized state `u_hat`, its tangent
linear term and positive-definite quadratic term are

    r = K*u_hat + beta*C(u_hat) - mu*M*u_hat,
    H = K + 3beta*D(u_hat),

where `mu` is the chemical potential. Local solvers minimize
`dot(r,d) + 0.5dot(d,H*d)` subject to the linearized mass constraint
`dot(u_hat,M*d)=0`, then retract the candidate to the mass sphere.
"""
function tangent_quadratic_model(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  isnothing(e.density_matrix) && throw(
    ArgumentError("tangent_quadratic_model requires a density_matrix callback"),
  )
  length(u) == dimension(e) ||
    throw(DimensionMismatch("iterate length must equal the energy dimension"))
  u_normalized = copy(u)
  normalize_M!(u_normalized, e.M)
  mu = chemical_potential(e, u_normalized)
  residual =
    e.K * u_normalized .+ e.beta .* e.cubic_gradient(u_normalized) .-
    mu .* (e.M * u_normalized)
  H = e.K + 3e.beta .* e.density_matrix(u_normalized)
  return GrossPitaevskiiTangentQuadraticModel(
    H,
    e.M,
    u_normalized,
    residual,
    mu,
  )
end

"""
    projected_newton_model(e, u)

Build the second-order model of the physical GP energy after retraction to the
unit-mass sphere. If `d` is tangent at the normalized state `u_hat`, then

    E(normalize_M(u_hat + t*d)) = E(u_hat) + t*r'd
      + 0.5t^2*d'*(H - mu*M)*d + O(t^3),

where `H = K + 3beta*D(u_hat)`. Local solvers use the projected spaces
`P*V_i`, with `P = I - u_hat*(M*u_hat)'`.
"""
function projected_newton_model(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  isnothing(e.density_matrix) && throw(
    ArgumentError("projected_newton_model requires a density_matrix callback"),
  )
  length(u) == dimension(e) ||
    throw(DimensionMismatch("iterate length must equal the energy dimension"))
  u_normalized = copy(u)
  normalize_M!(u_normalized, e.M)
  mu = chemical_potential(e, u_normalized)
  residual =
    e.K * u_normalized .+ e.beta .* e.cubic_gradient(u_normalized) .-
    mu .* (e.M * u_normalized)
  H_lagrangian = e.K + 3e.beta .* e.density_matrix(u_normalized) - mu .* e.M
  return GrossPitaevskiiProjectedNewtonModel(
    H_lagrangian,
    e.M,
    e.K,
    u_normalized,
    residual,
    mu,
  )
end

"""
    frozen_density_model(e, u; previous=nothing, alpha=1)

Return the generalized Rayleigh quotient obtained by freezing the
Gross--Pitaevskii density at the normalized state represented by `u`:

    (K + beta * D(u_hat), M),    u_hat' M u_hat = 1.

This is the linear eigenproblem used by one SCF step. An explicit
`density_matrix` callback is required because an assembled matrix, rather than
only its action on selected vectors, is needed by the linear EVP solver. If a
`previous` iterate is supplied, the frozen density is charge-mixed as
`alpha * D(u_hat) + (1-alpha) * D(previous_hat)`. Both states are normalized
independently before assembling their density matrices.
"""
function frozen_density_model(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector;
  previous::Union{Nothing,AbstractVector} = nothing,
  alpha::Real = 1,
)
  isnothing(e.density_matrix) && throw(
    ArgumentError("frozen_density_model requires a density_matrix callback"),
  )
  length(u) == dimension(e) ||
    throw(DimensionMismatch("iterate length must equal the energy dimension"))
  0 < alpha <= 1 || throw(ArgumentError("alpha must satisfy 0 < alpha <= 1"))
  !isnothing(previous) &&
    length(previous) != dimension(e) &&
    throw(
      DimensionMismatch(
        "previous iterate length must equal the energy dimension",
      ),
    )
  u_normalized = copy(u)
  normalize_M!(u_normalized, e.M)
  density = e.density_matrix(u_normalized)
  if !isnothing(previous) && alpha != 1
    previous_normalized = copy(previous)
    normalize_M!(previous_normalized, e.M)
    density =
      alpha .* density .+ (1 - alpha) .* e.density_matrix(previous_normalized)
  end

  A = e.K + e.beta .* density
  return GeneralizedRayleighQuotient(A, e.M)
end

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
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  _, mass = _gp_mass(e, u)
  return dot(u, e.K * u) / mass + e.beta * e.quartic(u) / mass^2
end

function projected_residual(
  e::GrossPitaevskiiRayleighQuotient,
  u::AbstractVector,
)
  _, mass = _gp_mass(e, u)
  v = u ./ sqrt(mass)
  lambda = chemical_potential(e, v)
  return e.K * v .+ e.beta .* e.cubic_gradient(v) .- lambda .* (e.M * v)
end

function residual_norm(e::GrossPitaevskiiRayleighQuotient, u::AbstractVector)
  return norm(projected_residual(e, u))
end
