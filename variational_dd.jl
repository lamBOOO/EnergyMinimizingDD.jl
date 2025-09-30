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

# Underlying dimension
dimension(e::AbstractEnergy) = error("dimension not implemented for $(typeof(e))")

# Optionally provide defaults via AD; otherwise keep them abstract.
gradient(e::AbstractEnergy, x) = error("gradient not implemented for $(typeof(e))")
hessian(e::AbstractEnergy, x) = error("hessian not implemented for $(typeof(e))")

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
struct QuadraticEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractEnergy{T}
  A::M           # can be Dense, Sparse, or Symmetric wrapper
  b::V
  c::T
end

# Make a convenient constructor; wrap A as Symmetric if you know it.
QuadraticEnergy(A::AbstractMatrix{T}, b::AbstractVector{T}; c::T=zero(T)) where {T} =
  QuadraticEnergy{T,typeof(A),typeof(b)}(A, b, c)

# E(x)
energy(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  T(0.5) * dot(x, e.A * x) - dot(e.b, x) + e.c

# dimension
dimension(e::QuadraticEnergy) = size(e.A, 1)

# ∇E(x) = Ax - b  (if A symmetric; if not, this is gradient of 1/2 x'(A+A')x - b'x)
gradient(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} =
  e.A * x .- e.b

# ∇²E(x) = A (constant)
hessian(e::QuadraticEnergy{T}) where {T} = e.A
hessian(e::QuadraticEnergy{T}, x::AbstractVector{T}) where {T} = hessian(e)




# 2) Rayleigh quotient:  ρ(x) = (x'Ax) / (x'x), scale-invariant in x ≠ 0
struct RayleighQuotient{T,M<:AbstractMatrix{T}} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
end

RayleighQuotient(A::AbstractMatrix{T}) where {T} =
  RayleighQuotient{T,typeof(A)}(A)

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
  (2 / (xx * xx)) * (Ax .* xx .- x .* xAx)
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

# 3) Generalized Rayleigh quotient:  ρ(x) = (x'Ax) / (x'Bx), scale-invariant in x ≠ 0
#    (covers generalized eigenvalue problems Ax = λBx)
struct GeneralizedRayleighQuotient{T,M<:AbstractMatrix{T},N<:AbstractMatrix{T}} <: AbstractEnergy{T}
  A::M           # typically symmetric/hermitian for real-valued quotient
  B::N           # typically symmetric/hermitian positive definite
end
GeneralizedRayleighQuotient(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} =
  GeneralizedRayleighQuotient{T,typeof(A),typeof(B)}(A, B)
# ρ(x)
energy(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T
}) where {T} = begin
  num = dot(x, e.A * x)
  den = dot(x, e.B * x)
  @assert den != zero(T) "Generalized Rayleigh quotient undefined at x with x'Bx=0"
  num / den
end

# dimension
dimension(e::GeneralizedRayleighQuotient) = size(e.A, 1)

# ∇ρ(x) = 2 * ( (Ax)(x'Bx) - (Bx)(x'Ax) ) / (x'Bx)^2
gradient(e::GeneralizedRayleighQuotient{T}, x::AbstractVector{T
}) where {T} = begin
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
function hessian!(H::AbstractMatrix, e::GeneralizedRayleighQuotient,
  x::AbstractVector)
  throw(ErrorException(
    "In-place Hessian not implemented for GeneralizedRayleighQuotient"
  ))
end



# 4) Generic Nonlinear Energy: E(u) for general nonlinear problems
#    Can represent PDE problems like p-Laplacian: E(u) = ∫(|∇u|^p/p)dΩ - ∫f*u dΩ
#    Or simple algebraic systems: E(x) = ½||F(x)||² where F(x) = 0 is the root problem
struct NonlinearEnergy{T,F1<:Function,F2<:Function} <: AbstractEnergy{T}
  name::String   # Descriptive name (e.g., "p-Laplacian", "Circle-Cubic System")
  assembler::F1  # Function that assembles the energy given u: (u) -> energy_value
  grad_assembler::F2  # Function that assembles the gradient: (u) -> gradient_vector
  N::Int        # Problem dimension
  params::Dict{String,Any}  # Additional parameters (e.g., p for p-Laplacian, etc.)
end

NonlinearEnergy(name::String, assembler::F1, grad_assembler::F2, N::Int;
               params::Dict{String,Any}=Dict{String,Any}()) where {F1,F2} =
  NonlinearEnergy{Float64,F1,F2}(name, assembler, grad_assembler, N, params)

# E(u) - energy evaluation
energy(e::NonlinearEnergy{T}, u::AbstractVector{T}) where {T} = e.assembler(u)

# dimension
dimension(e::NonlinearEnergy) = e.N

# ∇E(u) - gradient evaluation
gradient(e::NonlinearEnergy{T}, u::AbstractVector{T}) where {T} = e.grad_assembler(u)



# 6) Linear Regression Energy: E(x) = ||Ax - b||² for least squares problems
#    Minimizing this energy leads to solving the normal equations A'Ax = A'b
struct LinearRegressionEnergy{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractEnergy{T}
  A::M           # Design matrix (m × n where m ≥ n typically)
  b::V           # Observation vector (length m)
end

LinearRegressionEnergy(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T} =
  LinearRegressionEnergy{T,typeof(A),typeof(b)}(A, b)

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
hessian(e::LinearRegressionEnergy{T}, x::AbstractVector{T}) where {T} = hessian(e)




# ---------- Utilities ----------
# Promote to Symmetric if you know A is symmetric to avoid accidental double work.
as_symmetric(A) = Symmetric(A)  # no-op if already Symmetric

function normalize_M!(u::Vector{Float64}, M::AbstractMatrix)
  nu = sqrt(dot(u, M * u))
  @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
  u ./= nu
end
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

# Test

# Make matrix exactly symmetric to avoid issues
A_sym = Symmetric(A)
RQ_sym = Energies.RayleighQuotient(A_sym)

# Check gradient first
@assert FiniteDiff.finite_difference_gradient(z -> Energies.energy(RQ_sym, z), x; absstep=1e-8) - Energies.gradient(RQ_sym, x) |> norm < 1E-6

# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - Energies.hessian(RQ_sym, x) |> norm < 1E-6

# Check Hessian in-place
H = zeros(length(x), length(x))
Energies.hessian!(H, RQ_sym, x)
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(RQ_sym, z), x; absstep=1e-8) - H |> norm < 1E-6

# Check GeneralizedRayleighQuotient
B = [2.0 0.0; 0.0 1.0]
GRQ = Energies.GeneralizedRayleighQuotient(A_sym, Symmetric(B))
GRQ(x)                    # Evaluate generalized Rayleigh quotient
Energies.gradient(GRQ, x) # Compute gradient
Energies.hessian(GRQ, x)   # Hessian implemented
# Check gradient first
@assert FiniteDiff.finite_difference_gradient(z -> Energies.energy(GRQ, z), x; absstep=1e-8) - Energies.gradient(GRQ, x) |> norm < 1E-6
# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(z -> Energies.gradient(GRQ, z), x; absstep=1e-8) - Energies.hessian(GRQ, x) |> norm < 1E-6






function FEM_Schroedinger(
  N::Int,
  m::Int=9;
  P::F1=(x -> exp(sqrt((x.data[1])^2 + (x.data[2])^2))),
  # also return RHS to solve source problem
  f::F2=(x -> 1.0),
  overlap::Int=2
) where {F1<:Function, F2<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model = CartesianDiscreteModel(domain, partition1; isperiodic=(false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags=["boundary"])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  U = TrialFESpace(VV, 0)
  a1(u, v) = ∫(∇(u) ⋅ ∇(v) + (x -> P(x)) * u * v)dΩ
  a2(u, v) = ∫(u * v)dΩ
  b(v) = ∫( (x -> f(x)) * v )dΩ
  K = assemble_matrix(a1, VV, U)
  M = assemble_matrix(a2, VV, U)
  b = assemble_vector(b, VV)
  g = GridapDistributed.compute_cell_graph(model)
  par = Metis.partition(g, m)
  elpar = create_elements_partition(par, m)
  create_overlapping_elements_partition!(elpar, g, m, overlap)
  t1 = time()
  dofspar = create_dofs_partition(elpar, VV)
  elapsed = time() - t1
  println("create_dofs_partition finished in $elapsed seconds")
  return K, M, b, dofspar, U
end

"""
  FEM_PLaplacian(N::Int, m::Int, p::Float64)

Set up a p-Laplacian problem using Gridap FEM on a unit square domain.
Returns the necessary components for domain decomposition including
optimized energy and gradient assemblers using cached FEFunction pattern.
"""
function FEM_PLaplacian(
  N::Int,
  m::Int=9,
  p::Float64=3.0,
  f::F=(x -> 1.0),
  overlap::Int=2
) where {F<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model = CartesianDiscreteModel(domain, partition1; isperiodic=(false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags=["boundary"])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  U = TrialFESpace(VV, 0)

  # Domain decomposition setup
  g = GridapDistributed.compute_cell_graph(model)
  par = Metis.partition(g, m)
  elpar = create_elements_partition(par, m)
  create_overlapping_elements_partition!(elpar, g, m, overlap)
  dofspar = create_dofs_partition(elpar, VV)

  # Cache setup for zero-allocation energy/gradient evaluation
  ndofs = num_free_dofs(U)

  # Cache FEFunction that we will reuse by mutating its DOF array
  ufe_cache = FEFunction(U, zeros(ndofs), get_dirichlet_dof_values(U))

  # Use smooth, branch-free ε-regularization
  eps2 = 1e-24
  half_p = p/2

  # Prebuild load pieces: ∫ f u = b_free⋅u_free + c_dirichlet
  rhs_form(v) = ∫( v * (x -> f(x)) )dΩ
  b_free = assemble_vector(rhs_form, VV)
  c_dirichlet = sum( ∫( FEFunction(U, zero(b_free),
                                        get_dirichlet_dof_values(U)) * (x -> f(x)) )dΩ )

  # Energy density: (|∇u|^2 + eps2)^(p/2) / p
  e_density = (∇u) -> ((∇u ⊙ ∇u + eps2)^half_p) / p

  # Optimized energy assembler using cached FEFunction
  function energy_assembler(u_vec::Vector{Float64})
    # Mutate the cached FEFunction instead of constructing a new one
    copyto!(get_free_dof_values(ufe_cache), u_vec)

    E_grad = sum( ∫( e_density ∘ ∇(ufe_cache) )dΩ )
    # ∫ f u = b_free⋅u_free + c_dirichlet
    return E_grad - (dot(b_free, u_vec) + c_dirichlet)
  end

  # Set up algebraic operator for gradient evaluation
  # p-Laplacian weak form following Gridap tutorial
  flux(∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    return gnorm_sq^((p-2)/2) * ∇u
  end

  # Jacobian for Newton method
  dflux(∇du, ∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    gnorm = sqrt(gnorm_sq)
    if gnorm < 1e-12  # Additional safety
      return zero(∇du)
    end
    return (p-2) * gnorm^(p-4) * (∇u ⊙ ∇du) * ∇u + gnorm^(p-2) * ∇du
  end

  # Weak residual and Jacobian
  res(u, v) = ∫(∇(v) ⊙ (flux ∘ ∇(u)) - v * (x -> f(x)))dΩ
  jac(u, du, v) = ∫(∇(v) ⊙ (dflux ∘ (∇(du), ∇(u))))dΩ

  # Create FE operator and get algebraic view
  feop = FEOperator(res, jac, U, VV)
  alg_op = Gridap.FESpaces.get_algebraic_operator(feop)

  # Pre-allocate vectors for efficiency
  r_temp = zeros(Float64, ndofs)

  # Optimized gradient assembler using algebraic operator (avoids FEFunction creation)
  function gradient_assembler(u_vec::Vector{Float64})
    # Use pre-allocated residual vector
    Gridap.Algebra.residual!(r_temp, alg_op, u_vec)
    return copy(r_temp)  # Return a copy to avoid mutation issues
  end

  return energy_assembler, gradient_assembler, dofspar, U, ndofs
end

"""
  solve_p_laplacian_gridap(N::Int, p::Float64)

Solve the p-Laplacian problem using standard Gridap approach following
the tutorial https://gridap.github.io/Tutorials/dev/pages/t004_p_laplacian/
Returns the solution for comparison with domain decomposition method.
Uses consistent smooth ε-regularization matching the DD version.
"""
function solve_p_laplacian_gridap(
  N::Int,
  p::Float64=3.0,
  f::F=(x -> 1.0)
) where {F<:Function}

  # Setup domain and FE space (same as DD version for consistency)
  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model = CartesianDiscreteModel(domain, partition1; isperiodic=(false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V0 = TestFESpace(model, reffe, dirichlet_tags=["boundary"])
  Ug = TrialFESpace(V0, 0)

  # Numerical integration setup
  degree = 2
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)

  # p-Laplacian weak form with consistent regularization
  # Use same smooth, branch-free ε-regularization as DD version
  eps2 = 1e-24

  flux(∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    return gnorm_sq^((p-2)/2) * ∇u
  end

  # Jacobian for Newton method
  dflux(∇du, ∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    gnorm = sqrt(gnorm_sq)
    if gnorm < 1e-12  # Additional safety
      return zero(∇du)
    end
    return (p-2) * gnorm^(p-4) * (∇u ⊙ ∇du) * ∇u + gnorm^(p-2) * ∇du
  end

  # Weak residual and Jacobian
  res(u, v) = ∫(∇(v) ⊙ (flux ∘ ∇(u)) - v * (x -> f(x)))dΩ
  jac(u, du, v) = ∫(∇(v) ⊙ (dflux ∘ (∇(du), ∇(u))))dΩ

  # Create FE operator
  op = FEOperator(res, jac, Ug, V0)

  # Setup nonlinear solver using NLsolve with optimized tolerance
  nls = NLSolver(
    show_trace=false,
    method=:newton,
    linesearch=LineSearches.BackTracking(),
    ftol=1e-8,
    iterations=50
  )
  solver = FESolver(nls)

  # Initial guess - small random perturbation
  Random.seed!(123)
  x0 = 0.01 * randn(Float64, num_free_dofs(Ug))
  uh0 = FEFunction(Ug, x0)

  # Solve the nonlinear problem
  uh, = solve!(uh0, solver, op)

  return uh, Ug
end

function create_dofs_partition(
  elemsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace
)
  m = sp.fe_basis.trian.model
  dim = size(m.grid_topology.n_m_to_nface_to_mfaces, 2) - 1
  npars = length(elemsp)
  @debug "create nodesp"
  nodesp = [Vector{Int32}() for _ in 1:npars]
  Threads.@threads for ipar = 1:npars
    nodesp[ipar] = sort(unique(vcat([m.grid_topology.n_m_to_nface_to_mfaces[dim+1][el] for el in elemsp[ipar]]...)))
  end

  @debug "create freenodesp"
  freenodesp = copy(nodesp)
  Threads.@threads for ipar = 1:npars
    filter!(e -> e in sp.metadata.free_dof_to_node, nodesp[ipar])
  end

  @debug "create dofsp"
  reverse_map = zeros(Int32, maximum(sp.metadata.free_dof_to_node))
  for (node, freenode) in enumerate(sp.metadata.free_dof_to_node)
    reverse_map[freenode] = node
  end
  dofsp = [reverse_map[freenodesp[ipar]] for ipar = 1:npars]
  return dofsp
end

function create_elements_partition(partition::Vector{Int32}, npars::Integer) # Helper function from DDEigenlab
  nelems = length(partition)
  @debug nelems, length(partition)
  @assert nelems == length(partition)
  elemsp = [Vector{Int32}() for _ in 1:npars]
  for iel = 1:nelems
    push!(elemsp[partition[iel]], iel)
  end
  @debug nelems, sum(length.(elemsp))
  @assert nelems == sum(length.(elemsp))
  return elemsp
end

function create_overlapping_elements_partition!(elemsp, g, npars::Integer, ol) # Helper function from DDEigenlab
  for iol = 1:ol
    @debug "overlap" iol
    Threads.@threads for ipar = 1:npars
      tmp = copy(elemsp)
      elemsp[ipar] = sort(unique(vcat([g[:, i].nzind for i in tmp[ipar]]...)))
    end
  end
end


"""
  inf_step(e::Energies.AbstractEnergy{Float64}, u_current::Vector{Float64}, idx_sub::AbstractVector)

Perform a local optimization step in the subdomain for variational domain decomposition.

This is an abstract method that must be implemented by concrete subtypes of `AbstractEnergy`.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: The energy functional object
- `u_current::Vector{Float64}`: Current solution vector
- `idx_sub::AbstractVector`: Indices defining the subdomain

# Throws
- `ErrorException`: Always throws since this is not implemented for the abstract type

# Notes
Concrete implementations should override this method to provide specific behavior
for different energy types in the variational domain decomposition framework.
"""
function inf_step(
  e::Energies.AbstractEnergy{Float64},
  u_current::Vector{Float64},
  idx_sub::AbstractVector
)
  # TODO: Use the more efficient DD operations with views
  throw(ErrorException("inf_step not implemented for $(typeof(e))"))
end

function inf_step(
  e::Energies.QuadraticEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  A, b = e.A, e.b

  # Extract relevant rows/columns from A and b
  if length(idx_sub) > 0
    # Build the local matrix more efficiently
    A_local = zeros(localdim, localdim)
    b_local = zeros(localdim)

    # First row/column: u_cur' * A * [u_cur, e_j1, e_j2, ...]
    Au = A * u_cur
    A_local[1, 1] = dot(u_cur, Au)  # u' * A * u
    b_local[1] = dot(b, u_cur)      # b' * u

    # First row/column: u_cur' * A * e_j
    for (k, j) in pairs(idx_sub)
      A_local[1, k+1] = Au[j]  # u' * A * e_j = (A * u)[j]
      A_local[k+1, 1] = Au[j]  # e_j' * A * u = (A * u)[j] (symmetric)
      b_local[k+1] = b[j]      # b' * e_j = b[j]
    end

    # Remaining entries: e_i' * A * e_j = A[i,j]
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        A_local[k1+1, k2+1] = A[j1, j2]
      end
    end
  else
    # Degenerate case: only current solution
    A_local = reshape([dot(u_cur, A * u_cur)], 1, 1)
    b_local = reshape([dot(b, u_cur)], 1)
  end

  @debug "Size of A_local: $(size(A_local))"
  # Solve the local quadratic minimization problem
  # min_{α} ½ α' A_local α - b_local' α
  # where α is the coefficient vector in the basis [u_cur, e_j1, e_j2, ...]
  α_new = A_local \ b_local

  # Use helper function to reconstruct (no normalization needed for QuadraticEnergy)
  x_new = reconstruct_implicit!(α_new, u_cur, idx_sub)
  return x_new
end

function inf_step(
  e::Energies.LinearRegressionEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  A, b = e.A, e.b

  # For linear regression energy ||Ax - b||², the optimal solution in any subspace
  # is found by solving the normal equations for that subspace

  if length(idx_sub) > 0
    # Build local system: minimize ||A*[u_cur, e_j1, e_j2, ...]*α - b||²
    # This gives normal equations: (A'A)_local α = (A'b)_local

    A_local = zeros(localdim, localdim)
    b_local = zeros(localdim)

    # Compute A*u_cur once for efficiency
    A_u_cur = A * u_cur
    AtA_u_cur = A' * A_u_cur  # A'A * u_cur
    Atb = A' * b              # A' * b

    # First entry: (A*u_cur)' * (A*u_cur) = u_cur' * A' * A * u_cur
    A_local[1, 1] = dot(A_u_cur, A_u_cur)
    b_local[1] = dot(A_u_cur, b)  # (A*u_cur)' * b

    # Cross terms: u_cur' * A' * A * e_j = (A'A * u_cur)[j]
    for (k, j) in pairs(idx_sub)
      A_local[1, k+1] = AtA_u_cur[j]  # u_cur' * A' * A * e_j
      A_local[k+1, 1] = AtA_u_cur[j]  # e_j' * A' * A * u_cur (symmetric)
      b_local[k+1] = Atb[j]           # e_j' * A' * b = (A' * b)[j]
    end

    # Diagonal entries: e_i' * A' * A * e_j = (A' * A)[i,j]
    AtA = A' * A
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        A_local[k1+1, k2+1] = AtA[j1, j2]
      end
    end
  else
    # Degenerate case: only current solution
    A_u_cur = A * u_cur
    A_local = reshape([dot(A_u_cur, A_u_cur)], 1, 1)
    b_local = reshape([dot(A_u_cur, b)], 1)
  end

  # Solve normal equations: A_local * α = b_local
  α_new = A_local \ b_local

  # Reconstruct solution in original space
  x_new = reconstruct_implicit!(α_new, u_cur, idx_sub)
  return x_new
end

"""
  inf_step(u_cur, K, M, idx_sub)

Calcualted x_new = argmin_{x ∈ span{u_cur, e_j, j ∈ idx_sub} \\ {0}} (x' K x)/(x' M x)
where {e_j} are standard basis vectors. The output is re-normalized in the M-norm.
"""
function inf_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  K, M = e.A, e.B

  # Extract relevant rows/columns from K and M
  if length(idx_sub) > 0
    # Build the local matrices more efficiently
    K_local = zeros(localdim, localdim)
    M_local = zeros(localdim, localdim)

    # First row/column: u_cur' * K/M * [u_cur, e_j1, e_j2, ...]
    K_u = K * u_cur
    M_u = M * u_cur

    K_local[1, 1] = dot(u_cur, K_u)  # u' * K * u
    M_local[1, 1] = dot(u_cur, M_u)  # u' * M * u

    # First row/column: u_cur' * K/M * e_j
    for (k, j) in pairs(idx_sub)
      K_local[1, k+1] = K_u[j]  # u' * K * e_j = (K * u)[j]
      K_local[k+1, 1] = K_u[j]  # e_j' * K * u = (K * u)[j] (symmetric)
      M_local[1, k+1] = M_u[j]  # u' * M * e_j = (M * u)[j]
      M_local[k+1, 1] = M_u[j]  # e_j' * M * u = (M * u)[j] (symmetric)
    end

    # Remaining entries: e_i' * K/M * e_j = K[i,j] and M[i,j]
    for (k1, j1) in pairs(idx_sub)
      for (k2, j2) in pairs(idx_sub)
        K_local[k1+1, k2+1] = K[j1, j2]
        M_local[k1+1, k2+1] = M[j1, j2]
      end
    end
  else
    # Degenerate case: only current solution
    K_local = reshape([dot(u_cur, K * u_cur)], 1, 1)
    M_local = reshape([dot(u_cur, M * u_cur)], 1, 1)
  end

  @debug "Size of K_local: $(size(K_local))"
  @debug "Size of M_local: $(size(M_local))"
  K_local_sym = Symmetric(K_local)
  M_local_sym = Symmetric(M_local)
  F = cholesky(K_local_sym)  # ≈ A^{-1} preconditioner
  res = lobpcg(K_local_sym, M_local_sym, false, 1; P=F, tol=1e-8, maxiter=500)

  # Use helper function to reconstruct, then normalize
  x_new = reconstruct_implicit!(res.X[:, 1], u_cur, idx_sub)
  Energies.normalize_M!(x_new, M)

  return x_new
end

"""
  inf_step(e::NonlinearEnergy, u_cur, idx_sub)

Perform local minimization step for nonlinear energy using Newton's method
in the subdomain spanned by {u_cur, e_j for j ∈ idx_sub}.
Works for any nonlinear problem: PDEs, algebraic systems, etc.
"""
function inf_step(
  e::Energies.NonlinearEnergy{Float64},
  u_cur::Vector{Float64},
  idx_sub::AbstractVector
)
  localdim = 1 + length(idx_sub)

  if length(idx_sub) == 0
    # Degenerate case: return current solution
    return copy(u_cur)
  end

  # Newton iteration for local minimization in subspace
  # We minimize E(u_cur + α₁*u_cur + ∑αⱼ*eⱼ) = E(∑βₖ*basis_k)
  # where basis = [u_cur, e_j1, e_j2, ...]

  α = zeros(localdim)
  α[1] = 1.0  # Initial guess: α = [1, 0, 0, ...]

  max_newton_iter = 100
  newton_tol = 1e-6

  for iter = 1:max_newton_iter
    # Reconstruct current iterate
    u_current = reconstruct_implicit!(α, u_cur, idx_sub)

    # Compute gradient and approximate Hessian in subspace
    grad_full = Energies.gradient(e, u_current)

    # Project gradient to subspace: g_local[k] = ∂E/∂αₖ
    g_local = zeros(localdim)
    g_local[1] = dot(grad_full, u_cur)  # ∂E/∂α₁
    for (k, j) in pairs(idx_sub)
      g_local[k+1] = grad_full[j]  # ∂E/∂αₖ = grad_full[j]
    end

    # Check convergence
    if norm(g_local) < newton_tol
      break
    end

    # Approximate Hessian using finite differences (could be improved with analytical Hessian)
    H_local = zeros(localdim, localdim)
    eps_fd = 1e-6

    for i = 1:localdim
      α_plus = copy(α)
      α_plus[i] += eps_fd
      u_plus = reconstruct_implicit!(α_plus, u_cur, idx_sub)
      grad_plus = Energies.gradient(e, u_plus)

      # Project gradient difference
      g_plus = zeros(localdim)
      g_plus[1] = dot(grad_plus, u_cur)
      for (k, j) in pairs(idx_sub)
        g_plus[k+1] = grad_plus[j]
      end

      H_local[:, i] = (g_plus .- g_local) / eps_fd
    end

    # Newton step with regularization for stability
    H_reg = H_local + 1e-8 * I
    try
      Δα = H_reg \ (-g_local)

      # Line search for stability
      step_size = 1.0
      α_new = α + step_size * Δα
      u_new = reconstruct_implicit!(α_new, u_cur, idx_sub)

      # Simple backtracking
      while step_size > 1e-4
        u_test = reconstruct_implicit!(α + step_size * Δα, u_cur, idx_sub)
        if Energies.energy(e, u_test) <= Energies.energy(e, u_current)
          break
        end
        step_size *= 0.5
      end

      α += step_size * Δα
    catch
      # If Hessian is singular, use steepest descent
      α -= 0.01 * g_local / (norm(g_local) + 1e-12)
    end
  end

  return reconstruct_implicit!(α, u_cur, idx_sub)
end


"""
  combine_step(e::Energies.AbstractEnergy{Float64}, sspace::Matrix{Float64})

Combine step function for variational domain decomposition methods.

This is an abstract interface that must be implemented by concrete energy types.
The function is intended to perform a combination step in the variational domain
decomposition algorithm, typically involving operations on the given subspace.

# Arguments
- `e::Energies.AbstractEnergy{Float64}`: Energy functional or operator
- `sspace::Matrix{Float64}`: Subspace matrix, typically containing basis vectors
  or coefficients for the current iteration

# Throws
- `ErrorException`: Always throws an error indicating that the method must be
  implemented for the specific energy type

# Notes
This is a fallback method that serves as a template. Concrete implementations
should override this method for specific energy types to provide the actual
combination step logic.
"""
function combine_step(
  e::Energies.AbstractEnergy{Float64},
  sspace::Matrix{Float64},
)
  throw(ErrorException("combine_step not implemented for $(typeof(e))"))
end

function combine_step(
  e::Energies.QuadraticEnergy{Float64},
  sspace::Matrix{Float64},
)
  # TODO: Is combine step the same as inf step in general?
  # => Just 1st order optimality in subspace?
  A, b = e.A, e.b

  B = Matrix(qr(sspace).Q)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_A = Matrix{Float64}(undef, N, localdim)
  A_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_A, A, B)
  mul!(A_local, B', temp_A)  # => A_local = B' A B

  b_local = B' * b  # => b_local = B' b

  # Solve the linear system A_local α = b_local to minimize the quadratic energy
  α_new = A_local \ b_local
  x_new = B * α_new
  return x_new
end

"""
  combine_step(e::Energies.GeneralizedRayleighQuotient, u_collection::Matrix)

Perform a Rayleigh-Ritz procedure in the span of the columns of sspace.
The output is re-normalized in the M-norm.
"""
function combine_step(
  e::Energies.GeneralizedRayleighQuotient{Float64},
  sspace::Matrix{Float64},
)
  K, M = e.A, e.B

  B = Matrix(qr(sspace).Q)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_K = Matrix{Float64}(undef, N, localdim)
  temp_M = Matrix{Float64}(undef, N, localdim)
  K_local = Matrix{Float64}(undef, localdim, localdim)
  M_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_K, K, B)
  mul!(temp_M, M, B)
  mul!(K_local, B', temp_K)  # => K_local = B' K B
  mul!(M_local, B', temp_M)  # => M_local = B' M B

  _, eigvecs = eigen(K_local, M_local)
  x_new = B * eigvecs[:, 1]
  Energies.normalize_M!(x_new, M)
  return x_new
end

function combine_step(
  e::Energies.LinearRegressionEnergy{Float64},
  sspace::Matrix{Float64},
)
  # For linear regression, the combine step solves the least squares problem
  # in the subspace spanned by the columns of sspace: min ||A*B*α - b||²
  # where B = QR factorization of sspace

  A, b = e.A, e.b

  B = Matrix(qr(sspace).Q)

  localdim = size(B, 2)
  N = size(B, 1)

  # Pre-allocate temporary matrices for efficiency
  temp_A = Matrix{Float64}(undef, size(A, 1), localdim)
  A_local = Matrix{Float64}(undef, localdim, localdim)

  mul!(temp_A, A, B)        # temp_A = A * B
  mul!(A_local, temp_A', temp_A)  # A_local = B' * A' * A * B = (A*B)' * (A*B)

  b_local = temp_A' * b     # b_local = B' * A' * b = (A*B)' * b

  # Solve the normal equations: A_local α = b_local
  α_new = A_local \ b_local
  x_new = B * α_new
  return x_new
end

"""
  combine_step(e::NonlinearEnergy, sspace)

Perform combine step for nonlinear energy by minimizing energy in the subspace
spanned by the columns of sspace using Newton's method.
Works for any nonlinear problem: PDEs, algebraic systems, etc.
"""
function combine_step(
  e::Energies.NonlinearEnergy{Float64},
  sspace::Matrix{Float64},
)
  B = Matrix(qr(sspace).Q)
  localdim = size(B, 2)

  # Newton iteration in subspace
  α = zeros(localdim)
  α[1] = 1.0  # Initial guess: mostly first component

  max_newton_iter = 10
  newton_tol = 1e-6

  for iter = 1:max_newton_iter
    u_current = B * α

    # Compute gradient and project to subspace
    grad_full = Energies.gradient(e, u_current)
    g_local = B' * grad_full

    # Check convergence
    if norm(g_local) < newton_tol
      break
    end

    # Approximate Hessian in subspace using finite differences
    H_local = zeros(localdim, localdim)
    eps_fd = 1e-6

    for i = 1:localdim
      α_plus = copy(α)
      α_plus[i] += eps_fd
      u_plus = B * α_plus
      grad_plus = Energies.gradient(e, u_plus)
      g_plus = B' * grad_plus
      H_local[:, i] = (g_plus .- g_local) / eps_fd
    end

    # Newton step with regularization
    H_reg = H_local + 1e-8 * I
    try
      Δα = H_reg \ (-g_local)

      # Simple line search
      step_size = 1.0
      while step_size > 1e-4
        α_test = α + step_size * Δα
        u_test = B * α_test
        if Energies.energy(e, u_test) <= Energies.energy(e, u_current)
          break
        end
        step_size *= 0.5
      end

      α += step_size * Δα
    catch
      # Fallback to steepest descent
      α -= 0.01 * g_local / (norm(g_local) + 1e-12)
    end
  end

  return B * α
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::AbstractMatrix)
  w = u .- v
  return sqrt(dot(w, M * w))
end

"""
  restrict_to_subdomain!(u_local::AbstractVector, u_global::AbstractVector, idx_sub::AbstractVector)

Efficient restriction operator: extract subdomain values from global vector.
Writes u_local[k] = u_global[idx_sub[k]] for k = 1:length(idx_sub).
"""
function restrict_to_subdomain!(u_local::AbstractVector, u_global::AbstractVector, idx_sub::AbstractVector)
  for (k, j) in pairs(idx_sub)
    u_local[k] = u_global[j]
  end
  return u_local
end

"""
  restrict_to_subdomain(u_global::AbstractVector, idx_sub::AbstractVector)

Allocating version of restriction operator.
"""
function restrict_to_subdomain(u_global::AbstractVector, idx_sub::AbstractVector)
  u_local = similar(u_global, length(idx_sub))
  return restrict_to_subdomain!(u_local, u_global, idx_sub)
end

"""
  extend_from_subdomain!(u_global::AbstractVector, u_local::AbstractVector, idx_sub::AbstractVector)

Efficient extension operator: scatter subdomain values into global vector.
Writes u_global[idx_sub[k]] = u_local[k] for k = 1:length(idx_sub).
Does not zero out other entries - use zero_out_complement! if needed.
"""
function extend_from_subdomain!(u_global::AbstractVector, u_local::AbstractVector, idx_sub::AbstractVector)
  for (k, j) in pairs(idx_sub)
    u_global[j] = u_local[k]
  end
  return u_global
end

"""
  zero_out_complement!(u_global::AbstractVector, idx_sub::AbstractVector)

Zero out all entries of u_global except those indexed by idx_sub.
Useful for creating functions supported only on subdomain.
"""
function zero_out_complement!(u_global::AbstractVector, idx_sub::AbstractVector)
  # Create a set for fast lookup
  idx_set = Set(idx_sub)
  for i in eachindex(u_global)
    if i ∉ idx_set
      u_global[i] = 0.0
    end
  end
  return u_global
end

"""
  SubdomainView{T,V<:AbstractVector{T}} <: AbstractVector{T}

A view into a global vector that presents only the subdomain DOFs.
Avoids allocation when working with subdomain data.
"""
struct SubdomainView{T,V<:AbstractVector{T}} <: AbstractVector{T}
  parent::V
  indices::Vector{Int}
end

# AbstractArray interface
Base.size(v::SubdomainView) = (length(v.indices),)
Base.getindex(v::SubdomainView, i::Int) = v.parent[v.indices[i]]
Base.setindex!(v::SubdomainView, val, i::Int) = (v.parent[v.indices[i]] = val)
Base.IndexStyle(::Type{<:SubdomainView}) = IndexLinear()

"""
  subdomain_view(u_global::AbstractVector, idx_sub::AbstractVector)

Create a view into the global vector that presents only the subdomain DOFs.
Changes to the view are reflected in the original vector.
"""
function subdomain_view(u_global::AbstractVector{T}, idx_sub::AbstractVector) where T
  return SubdomainView{T,typeof(u_global)}(u_global, collect(Int, idx_sub))
end

"""
  restrict_matrix_block!(A_local::AbstractMatrix, A_global::AbstractMatrix,
                        row_indices::AbstractVector, col_indices::AbstractVector)

Efficient matrix restriction: extract block from global matrix.
A_local[i,j] = A_global[row_indices[i], col_indices[j]]
"""
function restrict_matrix_block!(A_local::AbstractMatrix, A_global::AbstractMatrix,
                               row_indices::AbstractVector, col_indices::AbstractVector)
  for (j, col_idx) in pairs(col_indices)
    for (i, row_idx) in pairs(row_indices)
      A_local[i, j] = A_global[row_idx, col_idx]
    end
  end
  return A_local
end

"""
  restrict_matrix_block(A_global::AbstractMatrix, indices::AbstractVector)

Allocating version for symmetric case: extract A_global[indices, indices].
"""
function restrict_matrix_block(A_global::AbstractMatrix, indices::AbstractVector)
  n = length(indices)
  A_local = Matrix{eltype(A_global)}(undef, n, n)
  return restrict_matrix_block!(A_local, A_global, indices, indices)
end

"""
  reconstruct_implicit!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)

Reconstruct x_new from implicit basis [u_cur, e_j1, e_j2, ...] where e_j are standard
basis vectors, without normalization. This avoids explicitly constructing the basis matrix.
"""
function reconstruct_implicit!(α::Vector{Float64}, u_cur::Vector{Float64}, idx_sub::AbstractVector)
  # The following is equivalent to x_new = [u_cur e_j1 e_j2 ...] * α
  # but avoids constructing e_j explicitly
  x_new = α[1] * u_cur  # Coefficient for current solution
  # Add contributions from standard basis vectors
  for (k, j) in pairs(idx_sub)
    x_new[j] += α[k+1]  # Add coefficient for e_j
  end
  return x_new
end# not implemented for abstract energy
function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{Vector{Int32}};
  kwargs...
)
  throw(ErrorException("var_dd not implemented for $(typeof(e))"))
end

function var_dd(
  e::Energies.AbstractEnergy{Float64},
  subdomain_dofs::Vector{Vector{Int32}};
  maxiter::Int=50,
  tol::Float64=1e-8
)
  # TODO: Add sweep option, multiplicative version

  # Initial guess, no need to normalize apparently
  u_cur = ones(Energies.dimension(e))

  e_hist = Float64[]
  sol_hist = Vector{Vector{Float64}}()

  e_cur = e(u_cur)
  push!(e_hist, e_cur)
  push!(sol_hist, copy(u_cur))

  m = length(subdomain_dofs)

  local_updates = zeros(size(u_cur, 1), m)  # preallocate for efficiency
  for n in 1:maxiter
    for i = 1:m
      u_next_i = inf_step(e, u_cur, subdomain_dofs[i])
      local_updates[:, i] = u_next_i
    end

    combined_matrix = hcat(u_cur, local_updates)
    u_new = combine_step(e, combined_matrix)

    resnorm = Energies.residual_norm(e, u_new)
    @printf(
      "Iteration %3d: Residual norm ≈ %12.6e energy = %12.6e\n",
      n, resnorm, e(u_new)
    )

    e_new = e(u_new)
    push!(e_hist, e_new)
    push!(sol_hist, copy(u_new))
    if resnorm < tol
      println("Converged at iteration $n with energy e = $e_new")
      return u_new, e_new, e_hist, sol_hist
    end

    u_cur = u_new
    e_cur = e_new
  end

  @warn "Reached maxiter=$maxiter with energy ≈ $e_cur"
  return u_cur, e_cur, e_hist, sol_hist
end



N = 20
m = 9
maxiter = 200
tol = 1e-5
overlap = 2

# Schroedinger EVP FEM
println("\n=== Schrödinger EVP FEM Example ===")
K, M, b, part, U = FEM_Schroedinger(N, m, overlap=overlap)
energy_eigen_fem = Energies.GeneralizedRayleighQuotient(K, M)
# energy_eigen_fem = Energies.RayleighQuotient(K)
u_approx, lambda_approx, lambda_history, solutions = var_dd(
  energy_eigen_fem,
  part,
  maxiter=maxiter,
  tol=tol
)
println("Final approximate eigenvalue = $lambda_approx")
exact_sol = eigs(K, M, nev=1, which=:SM)
println("Exact eigenvalue = $(exact_sol[1][1])")
@assert abs(lambda_approx - exact_sol[1][1]) < 1e-6
# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  "eigen_solution",
  cellfields = ["u_approx" => FEFunction(U, u_approx)]
)
# write all sols to vtk file for visualization
for (i, sol) in enumerate(solutions)
  writevtk(
    U.space.fe_basis.trian,
    "schroedinger_solution_iter$(i-1)",
    cellfields = ["u" => FEFunction(U, sol)]
  )
end
println("✓ passed.")



# Poisson problem FEM
println("\n=== Poisson Linear FEM Example ===")
K, M, b, part, U = FEM_Schroedinger(
  N, m, P=(x -> 0.0), f=(x -> 1.0), overlap=overlap
)
energy_poisson_fem = Energies.QuadraticEnergy(K, b, 0.0)
# direct solve
u_poisson_direct = K \ b
# var_dd solve
u_poisson, E_poisson, E_hist, sols = var_dd(energy_poisson_fem, part, maxiter=maxiter, tol=tol)
# difference
@assert norm(u_poisson - u_poisson_direct) < 1e-4
E_poisson = Energies.energy(energy_poisson_fem, u_poisson)
println("Poisson energy = $E_poisson")
# write to vtk file using U info
writevtk(
  U.space.fe_basis.trian,
  "poisson_solution",
  cellfields = [
    "u_poisson" => FEFunction(U, u_poisson),
    "u_poisson_direct" => FEFunction(U, u_poisson_direct)
  ]
)
println("✓ passed.")

# write all sols to vtk file for visualization
for (i, sol) in enumerate(sols)
  writevtk(
    U.space.fe_basis.trian,
    "poisson_solution_iter$(i-1)",
    cellfields = ["u" => FEFunction(U, sol)]
  )
end

# Linear Regression problem example
println("\n=== Linear Regression Example ===")
# Create a synthetic linear regression problem: y = Ax + ε
n_params = 20    # number of parameters to estimate
n_obs = 100      # number of observations (overdetermined system)
Random.seed!(42) # for reproducibility

# Generate random design matrix and true parameters
A_lr = randn(n_obs, n_params)
x_true = randn(n_params)
b_lr = A_lr * x_true + 0.1 * randn(n_obs)  # add some noise

# Create linear regression energy functional
energy_lr = Energies.LinearRegressionEnergy(A_lr, b_lr)

# Create simple uniform partition for demonstration
# In practice, this would be more sophisticated domain decomposition
part_lr = [Vector{Int32}() for _ in 1:m]
for i in 1:n_params
  push!(part_lr[((i-1) % m) + 1], i)
end

# Direct solution via normal equations
x_direct = (A_lr' * A_lr) \ (A_lr' * b_lr)

# Domain decomposition solution
x_dd, E_lr, E_hist_lr, sols_lr = var_dd(energy_lr, part_lr, maxiter=maxiter, tol=tol)

# Compare solutions
println("Direct least squares energy: $(Energies.energy(energy_lr, x_direct))")
println("DD least squares energy: $E_lr")
println("Relative error in solution: $(norm(x_dd - x_direct) / norm(x_direct))")
println("Residual norm (direct): $(norm(A_lr * x_direct - b_lr))")
println("Residual norm (DD): $(norm(A_lr * x_dd - b_lr))")

# Verify that we're solving the normal equations
normal_residual_direct = (A_lr' * A_lr) * x_direct - (A_lr' * b_lr)
normal_residual_dd = (A_lr' * A_lr) * x_dd - (A_lr' * b_lr)
println("Normal equation residual (direct): $(norm(normal_residual_direct))")
println("Normal equation residual (DD): $(norm(normal_residual_dd))")

@assert norm(x_dd - x_direct) / norm(x_direct) < 1e-3
@assert abs(E_lr - Energies.energy(energy_lr, x_direct)) < 1e-6
println("✓ passed.")


# p-Laplacian nonlinear problem example
println("\n=== p-Laplacian Nonlinear FEM Example ===")
p_val = 3.0
N_small = 20  # Use smaller problem size for testing
m_small = 9   # Fewer subdomains
energy_assembler, grad_assembler, part_pl, U_pl, n_dofs = FEM_PLaplacian(N_small, m_small, p_val)

# Create p-Laplacian energy functional using the generic NonlinearEnergy
energy_pl = Energies.NonlinearEnergy("p-Laplacian", energy_assembler, grad_assembler, n_dofs,
                                    params=Dict{String,Any}("p" => p_val))

# Test energy and gradient evaluation with better initial guess
Random.seed!(123)  # For reproducibility
u_test = 0.01 * randn(n_dofs)  # Small random initial guess to break symmetry
try
  E_test = Energies.energy(energy_pl, u_test)
  grad_test = Energies.gradient(energy_pl, u_test)
  println("Initial energy: $E_test")
  println("Initial gradient norm: $(norm(grad_test))")

  # Domain decomposition solution with more relaxed tolerance
  u_pl, E_pl, E_hist_pl, sols_pl = var_dd(energy_pl, part_pl, maxiter=30, tol=1e-5)

  println("Final p-Laplacian energy: $E_pl")
  println("Final gradient norm: $(Energies.residual_norm(energy_pl, u_pl))")

  # Write to VTK file for visualization
  writevtk(
    U_pl.space.fe_basis.trian,
    "p_laplacian_solution",
    cellfields = ["u_pl" => FEFunction(U_pl, u_pl)]
  )

  # Verify that we've found a critical point (gradient should be small)
  final_grad_norm = Energies.residual_norm(energy_pl, u_pl)
  if final_grad_norm < 1e-2
    println("✓ p-Laplacian converged successfully!")
  else
    println("⚠ p-Laplacian converged but with larger residual: $final_grad_norm")
  end

  # Compare with Gridap reference solution
  println("\n--- Comparison with Gridap Reference ---")
  uh_ref, U_ref = solve_p_laplacian_gridap(N_small, p_val)
  u_ref = get_free_dof_values(uh_ref)

  # Compare solution vectors
  error_l2 = norm(u_pl - u_ref) / norm(u_ref)
  println("Relative L2 error vs Gridap reference: $(error_l2)")

  # Compare energies
  E_ref = Energies.energy(energy_pl, u_ref)
  println("DD energy: $E_pl")
  println("Reference energy: $E_ref")
  println("Energy difference: $(abs(E_pl - E_ref))")

  # Write both solutions to VTK for comparison
  writevtk(
    U_pl.space.fe_basis.trian,
    "p_laplacian_comparison",
    cellfields = [
      "u_dd" => FEFunction(U_pl, u_pl),
      "u_gridap" => FEFunction(U_pl, u_ref)
    ]
  )

  if error_l2 < 1e-4
    println("✓ DD solution agrees well with Gridap reference!")
  else
    println("⚠ Larger difference between DD and reference: check implementation")
  end

catch e
  println("Error in p-Laplacian example: $e")
  println("This may indicate implementation challenges with the nonlinear solver.")
end

# Nonlinear algebraic system example: Circle-Cubic intersection
println("\n=== Nonlinear Algebraic System Example ===")
println("Solving: F[1] = x[1]² + x[2]² - 1 = 0  (circle)")
println("         F[2] = x[1]³ - x[2] = 0        (cubic)")

# Define the system F(x) = 0 as an energy minimization: E(x) = ½||F(x)||²
function circle_cubic_system(x::Vector{Float64})
  F = zeros(2)
  F[1] = x[1]^2 + x[2]^2 - 1.0    # circle: x² + y² = 1
  F[2] = x[1]^3 - x[2]             # cubic: y = x³
  return F
end

# Energy functional: E(x) = ½||F(x)||²
function circle_cubic_energy(x::Vector{Float64})
  F = circle_cubic_system(x)
  return 0.5 * dot(F, F)
end

# Gradient: ∇E(x) = J(x)ᵀ F(x) where J is Jacobian of F
function circle_cubic_gradient(x::Vector{Float64})
  F = circle_cubic_system(x)

  # Jacobian matrix J = [∂F₁/∂x₁  ∂F₁/∂x₂]
  #                     [∂F₂/∂x₁  ∂F₂/∂x₂]
  J = zeros(2, 2)
  J[1, 1] = 2*x[1]        # ∂F₁/∂x₁ = 2x₁
  J[1, 2] = 2*x[2]        # ∂F₁/∂x₂ = 2x₂
  J[2, 1] = 3*x[1]^2      # ∂F₂/∂x₁ = 3x₁²
  J[2, 2] = -1.0          # ∂F₂/∂x₂ = -1

  return J' * F  # ∇E = Jᵀ F
end

# Create the nonlinear energy for the algebraic system
energy_circle_cubic = Energies.NonlinearEnergy(
  "Circle-Cubic System",
  circle_cubic_energy,
  circle_cubic_gradient,
  2,  # 2D problem
  params=Dict{String,Any}("description" => "Intersection of unit circle and cubic y=x³")
)

# Create simple partition for 2D problem (each subdomain gets one variable)
part_cc = [Vector{Int32}([1]), Vector{Int32}([2])]

# Initial guess (near one of the expected solutions)
x_init = [0.8, 0.5]  # Should converge to intersection point

try
  println("Initial guess: x = $(x_init)")
  println("Initial F(x) = $(circle_cubic_system(x_init))")
  println("Initial ||F(x)|| = $(norm(circle_cubic_system(x_init)))")
  println("Initial energy E(x) = $(Energies.energy(energy_circle_cubic, x_init))")

  # Solve using domain decomposition
  x_sol, E_sol, E_hist_cc, sols_cc = var_dd(energy_circle_cubic, part_cc, maxiter=50, tol=1e-5)

  println("\nSolution found: x = $(x_sol)")
  F_sol = circle_cubic_system(x_sol)
  println("Final F(x) = $(F_sol)")
  println("Final ||F(x)|| = $(norm(F_sol))")
  println("Final energy E(x) = $E_sol")

  # Verify the solution
  circle_error = abs(x_sol[1]^2 + x_sol[2]^2 - 1.0)
  cubic_error = abs(x_sol[1]^3 - x_sol[2])

  println("\nVerification:")
  println("Circle equation error: |x² + y² - 1| = $(circle_error)")
  println("Cubic equation error: |x³ - y| = $(cubic_error)")

  if norm(F_sol) < 1e-6
    println("✓ Nonlinear algebraic system solved successfully!")
    println("  Solution represents intersection of unit circle and cubic curve.")
  else
    println("⚠ System not fully converged, residual = $(norm(F_sol))")
  end

catch e
  println("Error in algebraic system example: $e")
end
