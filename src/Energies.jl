module Energies

using LinearAlgebra

export AbstractEnergy, QuadraticEnergy, RayleighQuotient
export GeneralizedRayleighQuotient, GrossPitaevskiiRayleighQuotient
export GrossPitaevskiiTangentQuadraticModel
export NonlinearEnergy, LinearRegressionEnergy
export energy, gradient, gradient!, hessian, hessian!, dimension, residual_norm
export quadratic_model
export physical_energy, chemical_potential, projected_residual
export frozen_density_model
export tangent_quadratic_model

include("energies/interface.jl")
include("energies/quadratic.jl")
include("energies/rayleigh.jl")
include("energies/gross_pitaevskii.jl")
include("energies/nonlinear.jl")

end # module
