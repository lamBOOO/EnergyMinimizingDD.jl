# Shared one-level nonlinear-source baselines and triangular partition output.
# This remains example infrastructure: AS/RAS are comparison methods, not
# part of the VariationalDD package solver API.

const NONLINEAR_SOURCE_METHODS = (
  :nonlinear_as,
  :nonlinear_ras,
  :var_dd,
  :var_dd_history,
)

function nonlinear_source_schwarz_baseline(
  energy,
  subdomains;
  method,
  core_subdomains=nothing,
  u0,
  maxiter,
  tolerance,
)
  method in (:nonlinear_as, :nonlinear_ras) ||
    throw(ArgumentError("unsupported nonlinear source baseline $method"))
  dofs = [collect(Int, indices) for indices in subdomains]
  restricted_dofs = if method == :nonlinear_ras
    isnothing(core_subdomains) && throw(ArgumentError(
      "nonlinear RAS requires the nonoverlapping METIS core partition",
    ))
    FEMDiscretizations.create_balanced_disjoint_dofs_partition(
      core_subdomains, length(u0)
    )
  else
    nothing
  end

  u = copy(u0)
  energy_history = [Energies.energy(energy, u)]
  residual_history = [norm(Energies.gradient(energy, u))]
  initial_residual = residual_history[1]
  for _ = 1:maxiter
    candidates = hcat([
      Solvers.inf_step(energy, u, indices) for indices in dofs
    ]...)
    corrections = candidates .- u
    direction = if method == :nonlinear_as
      vec(sum(corrections; dims=2))
    else
      restricted = zeros(length(u))
      for i in eachindex(dofs), degree in restricted_dofs[i]
        restricted[degree] = corrections[degree, i]
      end
      restricted
    end
    u = Solvers.minimize_affine_corrections(
      energy, u, reshape(direction, :, 1)
    ).u
    push!(energy_history, Energies.energy(energy, u))
    push!(residual_history, norm(Energies.gradient(energy, u)))
    residual_history[end] <= tolerance * initial_residual && break
  end
  return (u=u, energy_history=energy_history, residual_history=residual_history)
end

function nonlinear_source_vardd(
  energy,
  subdomains;
  u0,
  maxiter,
  tolerance,
  history_depth,
)
  initial_residual = norm(Energies.gradient(energy, u0))
  u, _, energy_history, _, residuals = Solvers.var_dd(
    energy,
    subdomains;
    u0=u0,
    maxiter=maxiter,
    tol=tolerance * initial_residual,
    history_depth=history_depth,
    verbose=false,
  )
  return (
    u=u,
    energy_history=energy_history,
    residual_history=vcat(initial_residual, residuals),
  )
end

function triangle_partition_rows(N, ms, overlap)
  rows = (
    m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[],
    x1=Float64[], y1=Float64[], x2=Float64[], y2=Float64[],
    x3=Float64[], y3=Float64[],
  )
  model = simplexify(CartesianDiscreteModel(
    (0, 1.0, 0, 1.0), (N, N); isperiodic=(false, false)
  ))
  coordinates = get_cell_coordinates(Triangulation(model))
  graph = GridapDistributed.compute_cell_graph(model, 1)
  for m in ms
    owner = Int.(Metis.partition(graph, m))
    element_partition = FEMDiscretizations.create_elements_partition(
      Int32.(owner), m
    )
    FEMDiscretizations.create_overlapping_elements_partition!(
      element_partition, graph, m, overlap
    )
    mult = zeros(Int, length(owner))
    for elements in element_partition, element in elements
      mult[element] += 1
    end
    for index in eachindex(owner)
      points = coordinates[index]
      push!(rows.m, m)
      push!(rows.N, N)
      push!(rows.idx, index)
      push!(rows.owner, owner[index])
      push!(rows.mult, mult[index])
      push!(rows.x1, points[1][1])
      push!(rows.y1, points[1][2])
      push!(rows.x2, points[2][1])
      push!(rows.y2, points[2][2])
      push!(rows.x3, points[3][1])
      push!(rows.y3, points[3][2])
    end
  end
  return rows
end
