module Energies

using LinearAlgebra

export AbstractEnergy, QuadraticEnergy, RayleighQuotient
export GeneralizedRayleighQuotient, GrossPitaevskiiRayleighQuotient
export GrossPitaevskiiProjectedNewtonModel
export NonlinearEnergy
export energy, gradient, gradient!, hessian, hessian!, dimension, residual_norm
export quadratic_model
export physical_energy, chemical_potential, projected_residual
export projected_newton_model

include("energies/interface.jl")
include("energies/quadratic.jl")
include("energies/rayleigh.jl")
include("energies/gross_pitaevskii.jl")
include("energies/nonlinear.jl")

end # module
