module FEMDiscretizations

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

function FEM_Schroedinger(
  N::Int,
  m::Int = 9;
  P::F1 = (x -> exp(sqrt((x.data[1])^2 + (x.data[2])^2))),
  # also return RHS to solve source problem
  f::F2 = (x -> 1.0),
  overlap::Int = 2,
) where {F1<:Function,F2<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model =
    CartesianDiscreteModel(domain, partition1; isperiodic = (false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  U = TrialFESpace(VV, 0)
  a1(u, v) = ∫(∇(u) ⋅ ∇(v) + (x -> P(x)) * u * v)dΩ
  a2(u, v) = ∫(u * v)dΩ
  b(v) = ∫((x -> f(x)) * v)dΩ
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
  m::Int = 9,
  p::Float64 = 3.0,
  f::F = (x -> 1.0),
  overlap::Int = 2,
) where {F<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model =
    CartesianDiscreteModel(domain, partition1; isperiodic = (false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
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
  half_p = p / 2

  # Prebuild load pieces: ∫ f u = b_free⋅u_free + c_dirichlet
  rhs_form(v) = ∫(v * (x -> f(x)))dΩ
  b_free = assemble_vector(rhs_form, VV)
  c_dirichlet = sum(
    ∫(FEFunction(U, zero(b_free), get_dirichlet_dof_values(U)) * (x -> f(x)))dΩ,
  )

  # Energy density: (|∇u|^2 + eps2)^(p/2) / p
  e_density = (∇u) -> ((∇u ⊙ ∇u + eps2)^half_p) / p

  # Optimized energy assembler using cached FEFunction
  function energy_assembler(u_vec::Vector{Float64})
    # Mutate the cached FEFunction instead of constructing a new one
    copyto!(get_free_dof_values(ufe_cache), u_vec)

    E_grad = sum(∫(e_density ∘ ∇(ufe_cache))dΩ)
    # ∫ f u = b_free⋅u_free + c_dirichlet
    return E_grad - (dot(b_free, u_vec) + c_dirichlet)
  end

  # Set up algebraic operator for gradient evaluation
  # p-Laplacian weak form following Gridap tutorial
  flux(∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    return gnorm_sq^((p - 2) / 2) * ∇u
  end

  # Jacobian for Newton method
  dflux(∇du, ∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    gnorm = sqrt(gnorm_sq)
    if gnorm < 1e-12  # Additional safety
      return zero(∇du)
    end
    return (p - 2) * gnorm^(p - 4) * (∇u ⊙ ∇du) * ∇u + gnorm^(p - 2) * ∇du
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
  p::Float64 = 3.0,
  f::F = (x -> 1.0),
) where {F<:Function}

  # Setup domain and FE space (same as DD version for consistency)
  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model =
    CartesianDiscreteModel(domain, partition1; isperiodic = (false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V0 = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
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
    return gnorm_sq^((p - 2) / 2) * ∇u
  end

  # Jacobian for Newton method
  dflux(∇du, ∇u) = begin
    gnorm_sq = ∇u ⊙ ∇u + eps2
    gnorm = sqrt(gnorm_sq)
    if gnorm < 1e-12  # Additional safety
      return zero(∇du)
    end
    return (p - 2) * gnorm^(p - 4) * (∇u ⊙ ∇du) * ∇u + gnorm^(p - 2) * ∇du
  end

  # Weak residual and Jacobian
  res(u, v) = ∫(∇(v) ⊙ (flux ∘ ∇(u)) - v * (x -> f(x)))dΩ
  jac(u, du, v) = ∫(∇(v) ⊙ (dflux ∘ (∇(du), ∇(u))))dΩ

  # Create FE operator
  op = FEOperator(res, jac, Ug, V0)

  # Setup nonlinear solver using NLsolve with optimized tolerance
  nls = NLSolver(
    show_trace = false,
    method = :newton,
    linesearch = LineSearches.BackTracking(),
    ftol = 1e-8,
    iterations = 50,
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
  elemsp::Vector{Vector{Int32}},
  sp::Gridap.FESpaces.UnconstrainedFESpace,
)
  m = sp.fe_basis.trian.model
  dim = size(m.grid_topology.n_m_to_nface_to_mfaces, 2) - 1
  npars = length(elemsp)
  @debug "create nodesp"
  nodesp = [Vector{Int32}() for _ = 1:npars]
  Threads.@threads for ipar = 1:npars
    nodesp[ipar] = sort(
      unique(
        vcat(
          [
            m.grid_topology.n_m_to_nface_to_mfaces[dim+1][el] for
            el in elemsp[ipar]
          ]...,
        ),
      ),
    )
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

function create_elements_partition(partition::Vector{Int32}, npars::Integer) # Helper function from VariationalDD
  nelems = length(partition)
  @debug nelems, length(partition)
  @assert nelems == length(partition)
  elemsp = [Vector{Int32}() for _ = 1:npars]
  for iel = 1:nelems
    push!(elemsp[partition[iel]], iel)
  end
  @debug nelems, sum(length.(elemsp))
  @assert nelems == sum(length.(elemsp))
  return elemsp
end

function create_overlapping_elements_partition!(elemsp, g, npars::Integer, ol) # Helper function from VariationalDD
  for iol = 1:ol
    @debug "overlap" iol
    Threads.@threads for ipar = 1:npars
      tmp = copy(elemsp)
      elemsp[ipar] = sort(unique(vcat([g[:, i].nzind for i in tmp[ipar]]...)))
    end
  end
end

end # module
