"""
    NonlinearEnergy(name, energy, gradient[, hessian], dimension)

An energy backed by user-provided value and derivative functions. The Hessian
is optional; solvers use a first-order fallback when it is absent.
"""
struct NonlinearEnergy{T,F,G,H} <: AbstractEnergy{T}
  name::String
  assembler::F
  grad_assembler::G
  hess_assembler::H
  N::Int
end

function NonlinearEnergy(name::String, assembler, grad_assembler, N::Int;)
  N >= 0 || throw(ArgumentError("dimension must be nonnegative"))
  return NonlinearEnergy{
    Float64,typeof(assembler),typeof(grad_assembler),Nothing
  }(
    name, assembler, grad_assembler, nothing, N
  )
end

function NonlinearEnergy(
  name::String, assembler, grad_assembler, hess_assembler, N::Int;
)
  N >= 0 || throw(ArgumentError("dimension must be nonnegative"))
  return NonlinearEnergy{
    Float64,typeof(assembler),typeof(grad_assembler),typeof(hess_assembler)
  }(
    name, assembler, grad_assembler, hess_assembler, N
  )
end

energy(e::NonlinearEnergy, u::AbstractVector) = e.assembler(u)
gradient(e::NonlinearEnergy, u::AbstractVector) = e.grad_assembler(u)
dimension(e::NonlinearEnergy) = e.N

function hessian(e::NonlinearEnergy, u::AbstractVector)
  isnothing(e.hess_assembler) && throw(
    ArgumentError(
      "analytic Hessian not available for nonlinear energy $(e.name)"
    ),
  )
  return e.hess_assembler(u)
end
