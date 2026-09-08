module FEMDiscretizations

using LinearAlgebra
using SparseArrays
using Gridap
using GridapDistributed
using Metis
using IterativeSolvers
using Arpack
using Printf
using Random

function FEM_Schroedinger(
  N::Int,
  m::Int = 9;
  P::F1 = (x -> exp(sqrt((x.data[1])^2 + (x.data[2])^2))),
  # also return RHS to solve source problem
  f::F2 = (x -> 1.0),
  diffusion::F3 = (x -> 1.0),
  overlap::Int = 2,
  partitioning::Symbol = :metis,
  cell_owners::Union{Nothing,AbstractVector{<:Integer}} = nothing,
  return_core_partition::Bool = false,
) where {F1<:Function,F2<:Function,F3<:Function}

  domain = (0, 1.0, 0, 1.0)
  partition1 = (1.0 * N, 1.0 * N)
  model =
    CartesianDiscreteModel(domain, partition1; isperiodic = (false, false))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VV = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  U = TrialFESpace(VV, 0)
  a1(u, v) =
    ∫((x -> diffusion(x)) * (∇(u) ⋅ ∇(v)) + (x -> P(x)) * u * v)dΩ
  a2(u, v) = ∫(u * v)dΩ
  b(v) = ∫((x -> f(x)) * v)dΩ
  K = assemble_matrix(a1, VV, U)
  M = assemble_matrix(a2, VV, U)
  b = assemble_vector(b, VV)
  g = GridapDistributed.compute_cell_graph(model)
  par = if !isnothing(cell_owners)
    length(cell_owners) == N^2 || throw(DimensionMismatch(
      "cell_owners must contain one owner for each of the $(N^2) cells",
    ))
    owners = Int32.(cell_owners)
    all(owner -> 1 <= owner <= m, owners) || throw(ArgumentError(
      "cell_owners entries must lie in 1:$m",
    ))
    all(i -> i in owners, 1:m) || throw(ArgumentError(
      "cell_owners must assign at least one cell to every subdomain",
    ))
    owners
  elseif partitioning == :metis
    Metis.partition(g, m)
  elseif partitioning == :cartesian
    nsub_direction = round(Int, sqrt(m))
    nsub_direction^2 == m || throw(ArgumentError(
      "partitioning=:cartesian requires m to be a perfect square",
    ))
    owners = Vector{Int32}(undef, N^2)
    for cell = 1:N^2
      xcell = mod1(cell, N)
      ycell = cld(cell, N)
      xowner = min(nsub_direction, (xcell - 1) * nsub_direction ÷ N + 1)
      yowner = min(nsub_direction, (ycell - 1) * nsub_direction ÷ N + 1)
      owners[cell] = xowner + nsub_direction * (yowner - 1)
    end
    owners
  else
    throw(ArgumentError("partitioning must be :metis or :cartesian"))
  end
  elpar = create_elements_partition(par, m)
  core_dofs =
    return_core_partition ? create_dofs_partition(elpar, VV) : nothing
  create_overlapping_elements_partition!(elpar, g, m, overlap)
  t1 = time()
  dofspar = create_dofs_partition(elpar, VV)
  elapsed = time() - t1
  println("create_dofs_partition finished in $elapsed seconds")
  result = (K, M, b, dofspar, U)
  return return_core_partition ? (result..., core_dofs) : result
end

"""
    FEM_GrossPitaevskii(N, m=9; P, overlap=2, domain, quadrature_degree=4)

Assemble the linear Schrödinger matrices and thread-local finite-element
evaluators required by `GrossPitaevskiiRayleighQuotient`. Returns
`K, M, quartic, cubic_gradient, density_matrix, dofspar, U`, where

    quartic(u) = integral(u_h^4),
    cubic_gradient(u)_i = integral(u_h^3 phi_i),
    density_matrix(u)_ij = integral(u_h^2 phi_i phi_j).

`domain` sets the rectangular computational domain. The default quadrature
degree exactly integrates the quartic term for affine `P1` elements; higher
orders can be selected for nonpolynomial trapping potentials.
"""
function FEM_GrossPitaevskii(
  N::Int,
  m::Int = 9;
  P::F = (x -> exp(sqrt((x.data[1])^2 + (x.data[2])^2))),
  overlap::Int = 2,
  domain::NTuple{4,<:Real} = (0.0, 1.0, 0.0, 1.0),
  quadrature_degree::Int = 4,
) where {F<:Function}
  partition = (1.0 * N, 1.0 * N)
  model = CartesianDiscreteModel(
    domain,
    partition;
    isperiodic = (false, false),
  )
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
  U = TrialFESpace(V, 0)
  omega = Triangulation(model)
  dOmega = Measure(omega, quadrature_degree)

  linear_form(u, v) =
    ∫(∇(u) ⋅ ∇(v) + (x -> P(x)) * u * v)dOmega
  mass_form(u, v) = ∫(u * v)dOmega
  K = assemble_matrix(linear_form, V, U)
  M = assemble_matrix(mass_form, V, U)

  graph = GridapDistributed.compute_cell_graph(model)
  owners = Metis.partition(graph, m)
  element_partition = create_elements_partition(owners, m)
  create_overlapping_elements_partition!(element_partition, graph, m, overlap)
  dofspar = create_dofs_partition(element_partition, V)

  ndofs = num_free_dofs(U)
  dirichlet_values = get_dirichlet_dof_values(U)
  caches = [
    FEFunction(U, zeros(ndofs), dirichlet_values) for _ = 1:Threads.nthreads()
  ]

  function cached_fe_function(u::AbstractVector)
    uh = caches[Threads.threadid()]
    copyto!(get_free_dof_values(uh), u)
    return uh
  end

  function quartic(u::AbstractVector)
    uh = cached_fe_function(u)
    return sum(∫(uh * uh * uh * uh)dOmega)
  end

  function cubic_gradient(u::AbstractVector)
    uh = cached_fe_function(u)
    cubic_form(v) = ∫(uh * uh * uh * v)dOmega
    return assemble_vector(cubic_form, V)
  end

  function density_matrix(u::AbstractVector)
    uh = cached_fe_function(u)
    density_form(w, v) = ∫(uh * uh * w * v)dOmega
    return assemble_matrix(density_form, V, U)
  end

  return K, M, quartic, cubic_gradient, density_matrix, dofspar, U
end

"""
    FEM_LShapeModel(N; grading=0.0)

Build a triangular mesh of

    (-1,1)^2 \\ ([-1,0] x [0,1])

with `N` cells per unit length. For `0 <= grading < 1`, vertices are moved
toward the re-entrant corner by the radial-in-the-max-norm map
`x -> x*norm(x,Inf)^(grading/(1-grading))`. Thus local mesh widths near the
origin scale like `h*r^grading`. In particular, `grading=0.4` uses the same
corner-grading exponent as Spicher--Wihler (2026), Section 6.1.
"""
function FEM_LShapeModel(N::Int; grading::Real = 0.0)
  N > 0 || throw(ArgumentError("N must be positive"))
  0 <= grading < 1 || throw(ArgumentError(
    "grading must satisfy 0 <= grading < 1",
  ))

  background = CartesianDiscreteModel(
    (-1.0, 1.0, -1.0, 1.0),
    (2N, 2N);
    isperiodic = (false, false),
  )
  cell_coordinates = get_cell_coordinates(get_grid(background))
  keep_cell = map(cell_coordinates) do coordinates
    center = sum(coordinates) / length(coordinates)
    !(center[1] < 0 && center[2] > 0)
  end
  portion = Gridap.Geometry.DiscreteModelPortion(background, keep_cell)
  triangular = simplexify(portion)

  old_grid = get_grid(triangular)
  exponent = grading / (1 - grading)
  grade(point) = begin
    radius = max(abs(point[1]), abs(point[2]))
    iszero(radius) ? point : point * radius^exponent
  end
  grid = Gridap.Geometry.UnstructuredGrid(
    grade.(Gridap.Geometry.get_node_coordinates(old_grid)),
    Gridap.Geometry.get_cell_node_ids(old_grid),
    Gridap.Geometry.get_reffes(old_grid),
    Gridap.Geometry.get_cell_type(old_grid),
  )
  topology = Gridap.Geometry.UnstructuredGridTopology(grid)
  # Rebuild the labels after removing the upper-left quadrant. Restricting
  # the background labels alone would leave the two new cut edges unmarked.
  labels = Gridap.Geometry.FaceLabeling(topology)
  return Gridap.Geometry.UnstructuredDiscreteModel(grid, topology, labels)
end

"Build the common triangular P1 space and METIS core/overlap partitions."
function _partitioned_triangular_p1_space(
  N::Int,
  m::Int,
  overlap::Int;
  model = nothing,
)
  m > 0 || throw(ArgumentError("m must be positive"))
  overlap >= 0 || throw(ArgumentError("overlap must be nonnegative"))

  model = isnothing(model) ? simplexify(CartesianDiscreteModel(
    (0, 1.0, 0, 1.0), (N, N); isperiodic = (false, false)
  )) : model
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])
  U = TrialFESpace(V, 0)
  omega = Triangulation(model)

  triangle_graph = GridapDistributed.compute_cell_graph(model, 1)
  triangle_owners = Metis.partition(triangle_graph, m)
  element_partition = create_elements_partition(triangle_owners, m)
  core_dofs = create_dofs_partition(element_partition, V)
  create_overlapping_elements_partition!(
    element_partition, triangle_graph, m, overlap
  )
  overlapping_dofs = create_dofs_partition(element_partition, V)

  return (
    model=model,
    V=V,
    U=U,
    omega=omega,
    overlapping_dofs=overlapping_dofs,
    core_dofs=core_dofs,
    ndofs=num_free_dofs(U),
  )
end


"""
    FEM_SemilinearPoisson(N, m=9; potential, potential_gradient,
                          potential_hessian, forcing=x->1.0, overlap=2,
                          quadrature_degree=6, initial_guess=x->0.0,
                          return_mass_matrix=false, model=nothing,
                          quadrature_rule=nothing)

Assemble the triangular P1 discretization of the generic semilinear energy

    E(u) = integral(1/2 * |grad u|^2 + V(u) - f*u),

whose Euler equation is `-Delta u + V'(u) = f`. The three potential callbacks
provide `V`, `V'`, and `V''`; consequently the returned `NonlinearEnergy`
assemblers use an analytic residual and sparse Hessian. Local overlapping DOFs
and the original METIS core DOFs are returned in the result tuple. With
`return_mass_matrix=true`, the consistent mass matrix
is appended to the return tuple; this supports stabilized pseudo-time
linearizations without changing the common nine-value interface. Pass a
two-dimensional Gridap `model` to assemble on a mesh other than the default
unit-square mesh; [`FEM_LShapeModel`](@ref) supplies the model used by the
Section 6.1 example. A Gridap `quadrature_rule` overrides
`quadrature_degree`, allowing that example to use the paper's edge-midpoint
triangle rule exactly.
"""
function FEM_SemilinearPoisson(
  N::Int,
  m::Int = 9;
  potential::V,
  potential_gradient::DV,
  potential_hessian::DDV,
  forcing::F = (x -> 1.0),
    overlap::Int = 2,
    quadrature_degree::Int = 6,
    initial_guess::I = (x -> 0.0),
    return_mass_matrix::Bool = false,
    model = nothing,
    quadrature_rule = nothing,
) where {V<:Function,DV<:Function,DDV<:Function,F<:Function,I<:Function}
  quadrature_degree > 0 ||
    throw(ArgumentError("quadrature_degree must be positive"))

  setup = _partitioned_triangular_p1_space(N, m, overlap; model=model)
  Vh, Uh, omega = setup.V, setup.U, setup.omega
  dOmega = isnothing(quadrature_rule) ?
           Measure(omega, quadrature_degree) :
           Measure(omega, quadrature_rule)
  ndofs = setup.ndofs
  dirichlet_values = get_dirichlet_dof_values(Uh)
  caches = [
    FEFunction(Uh, zeros(ndofs), dirichlet_values) for _ = 1:Threads.nthreads()
  ]
  function cached_fe_function(values)
    uh = caches[Threads.threadid()]
    copyto!(get_free_dof_values(uh), values)
    return uh
  end

  load(v) = ∫((x -> forcing(x)) * v)dOmega
  b = assemble_vector(load, Vh)
  stiffness(du, v) = ∫(∇(du) ⋅ ∇(v))dOmega
  K = sparse(assemble_matrix(stiffness, Vh, Uh))
  mass(du, v) = ∫(du * v)dOmega
  M = return_mass_matrix ? sparse(assemble_matrix(mass, Vh, Uh)) : nothing

  function energy_assembler(values::Vector{Float64})
    uh = cached_fe_function(values)
    return 0.5 * dot(values, K * values) +
           sum(∫(potential ∘ uh)dOmega) - dot(b, values)
  end

  function gradient_assembler(values::Vector{Float64})
    uh = cached_fe_function(values)
    reaction(v) = ∫((potential_gradient ∘ uh) * v)dOmega
    return K * values + assemble_vector(reaction, Vh) - b
  end

  function hessian_assembler(values::Vector{Float64})
    uh = cached_fe_function(values)
    reaction_tangent(du, v) =
      ∫((potential_hessian ∘ uh) * du * v)dOmega
    return K + sparse(assemble_matrix(reaction_tangent, Vh, Uh))
  end

  initial_fe = interpolate_everywhere(initial_guess, Uh)
  initial = collect(get_free_dof_values(initial_fe))
  result = (
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    setup.overlapping_dofs,
    Uh,
    ndofs,
    K,
    initial,
    setup.core_dofs,
  )
  return return_mass_matrix ? (result..., M) : result
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

"""
    create_balanced_disjoint_dofs_partition(core_dofs, ndofs)

Assign every degree of freedom to exactly one adjacent nonoverlapping element
core. Interface ties are assigned to the currently least-loaded admissible
core, retaining the METIS partition while avoiding a subdomain-index bias.
"""
function create_balanced_disjoint_dofs_partition(core_dofs, ndofs::Int)
  memberships = [Int[] for _ = 1:ndofs]
  for (subdomain, degrees) in enumerate(core_dofs), degree in degrees
    push!(memberships[degree], subdomain)
  end

  owned = [Int[] for _ in core_dofs]
  loads = zeros(Int, length(core_dofs))
  # Assign forced/interior DOFs first. Flexible interface DOFs are then used
  # to equalize the loads instead of inheriting the ordering of global DOFs.
  for degree in sortperm(length.(memberships))
    candidates = memberships[degree]
    isempty(candidates) && error("DOF $degree has no METIS core owner")
    owner = candidates[argmin(view(loads, candidates))]
    push!(owned[owner], degree)
    loads[owner] += 1
  end
  return owned
end

function create_elements_partition(partition::Vector{Int32}, npars::Integer) # Helper function from EnergyMinimizingDD
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

function create_overlapping_elements_partition!(elemsp, g, npars::Integer, ol) # Helper function from EnergyMinimizingDD
  for iol = 1:ol
    @debug "overlap" iol
    Threads.@threads for ipar = 1:npars
      tmp = copy(elemsp)
      elemsp[ipar] = sort(unique(vcat([g[:, i].nzind for i in tmp[ipar]]...)))
    end
  end
end

end # module
