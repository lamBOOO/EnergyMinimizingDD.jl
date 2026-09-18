# Fig. 17b duplicate for the non-monotone semilinear L-shaped problem from
# Spicher--Wihler (2026), Section 6.1. This intentionally uses the same
# focused method set, stopping criterion, partitions, and output schema as
# Study 11 so the corresponding paper figures are directly comparable.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :SEMILINEAR_SOURCE_METHODS) ||
  include("nonlinear_source_common.jl")
isdefined(Main, :REACTION_AMPLITUDE) ||
  include(joinpath(@__DIR__, "..", "semilinear_l_shape_section_61.jl"))

const LSHAPE_FIG17B_METHODS = (
  :var_dd,
  :var_dd_history,
  :var_dd_quadratic,
  :var_dd_quadratic_history,
  :anderson_ras,
  :nonlinear_cg_optim_as,
  :newton_pcg_as_1,
  :newton_pcg_as_2,
  :newton_pcg_as_4,
  :newton_pcg_as_8,
)
const LSHAPE_NONLINEARITY_CASES = (
  (id="exponential", beta=12.0),
)

function lshape_fig17b_method(
  method, energy, dofs, core, initial; maxiter, tolerance
)
  if method == :anderson_ras
    return nonlinear_source_anderson_ras(
      energy,
      dofs,
      core;
      u0=initial,
      maxiter=maxiter,
      tolerance=tolerance,
      history_depth=4,
    )
  elseif method == :nonlinear_cg_optim_as
    return nonlinear_source_optim_ncg_as(
      energy,
      dofs;
      u0=initial,
      maxiter=maxiter,
      tolerance=tolerance,
    )
  elseif method in (
    :newton_pcg_as_1, :newton_pcg_as_2, :newton_pcg_as_4, :newton_pcg_as_8
  )
    inner_iterations = parse(Int, split(string(method), "_")[end])
    return nonlinear_source_newton_pcg_as(
      energy,
      dofs;
      u0=initial,
      maxiter=maxiter,
      tolerance=tolerance,
      inner_iterations=inner_iterations,
    )
  elseif method in (
    :var_dd, :var_dd_history, :var_dd_quadratic, :var_dd_quadratic_history
  )
    history_depth =
      method in (:var_dd_history, :var_dd_quadratic_history) ? 1 : 0
    quadratic_model = method in (:var_dd_quadratic, :var_dd_quadratic_history)
    return nonlinear_source_vardd(
      energy,
      dofs;
      u0=initial,
      maxiter=maxiter,
      tolerance=tolerance,
      history_depth=history_depth,
      quadratic_model=quadratic_model,
    )
  end
  return error("unsupported Fig. 17b L-shaped method $method")
end

function lshape_triangle_rows(model, N)
  rows = (
    N=Int[],
    idx=Int[],
    x1=Float64[],
    y1=Float64[],
    value1=Float64[],
    x2=Float64[],
    y2=Float64[],
    value2=Float64[],
    x3=Float64[],
    y3=Float64[],
    value3=Float64[],
  )
  coordinates = get_cell_coordinates(Triangulation(model))
  for (index, points) in enumerate(coordinates)
    values = exact_solution.(points)
    push!(rows.N, N)
    push!(rows.idx, index)
    push!(rows.x1, points[1][1])
    push!(rows.y1, points[1][2])
    push!(rows.value1, values[1])
    push!(rows.x2, points[2][1])
    push!(rows.y2, points[2][2])
    push!(rows.value2, values[2])
    push!(rows.x3, points[3][1])
    push!(rows.y3, points[3][2])
    push!(rows.value3, values[3])
  end
  return rows
end

function lshape_partition_rows(model, N, ms, overlap)
  rows = (
    m=Int[],
    N=Int[],
    idx=Int[],
    owner=Int[],
    mult=Int[],
    x1=Float64[],
    y1=Float64[],
    x2=Float64[],
    y2=Float64[],
    x3=Float64[],
    y3=Float64[],
  )
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

function run_study11b()
  files = (
    "study11b_semilinear_l_shape_conv.csv",
    "study11b_semilinear_l_shape_summary.csv",
    "study11b_semilinear_l_shape_solution.csv",
    "study11b_semilinear_l_shape_partitions.csv",
    "study11b_semilinear_l_shape_work.csv",
  )
  if !needs_run(files...)
    cached = loadtable(files[1])
    expected_cases = Set(
      (case.id, case.beta, string(method)) for case in LSHAPE_NONLINEARITY_CASES,
      method in LSHAPE_FIG17B_METHODS
    )
    cached_cases = if hasproperty(cached, :case) &&
                      hasproperty(cached, :beta) &&
                      hasproperty(cached, :method)
      Set(zip(cached.case, cached.beta, cached.method))
    else
      Set()
    end
    if cached_cases == expected_cases
      println("study11b: cached, skipping")
      return nothing
    end
  end
  println("study11b: Section 6.1 semilinear L-shaped Fig. 17b duplicate")

  N = SMALL ? 4 : 16
  ms = SMALL ? [2] : [2, 4, 8]
  overlap = 2
  maxiter = SMALL ? 4 : 100
  tolerance = SMALL ? 1e-5 : 1e-7
  model = FEMDiscretizations.FEM_LShapeModel(N; grading=MESH_GRADING)

  convergence = (
    case=String[],
    beta=Float64[],
    m=Int[],
    method=String[],
    outer=Int[],
    energy_gap=Float64[],
    relative_residual=Float64[],
  )
  summary = (
    case=String[],
    beta=Float64[],
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    final_relative_residual=Float64[],
    relative_nodal_error=Float64[],
  )
  work = (
    case=String[],
    beta=Float64[],
    m=Int[],
    method=String[],
    outer_iterations=Int[],
    nonlinear_local_batches=Int[],
    linear_as_batches=Int[],
    global_jacobian_products=Int[],
  )

  exact_values = nothing
  for case in LSHAPE_NONLINEARITY_CASES, m in ms
    if case.id == "exponential"
      case_potential = s -> -case.beta * sqrt(pi) * erf(s) / 2
      case_gradient = s -> -case.beta * exp(-s^2)
      case_hessian = s -> 2 * case.beta * s * exp(-s^2)
      case_forcing =
        x -> minus_laplacian_exact(x) - case.beta * exp(-exact_solution(x)^2)
    else
      case_potential = s -> case.beta * s^4 / 4
      case_gradient = s -> case.beta * s^3
      case_hessian = s -> 3 * case.beta * s^2
      case_forcing =
        x -> minus_laplacian_exact(x) + case.beta * exact_solution(x)^3
    end
    ea, ga, ha, dofs, U, ndofs, _, initial, core = FEMDiscretizations.FEM_SemilinearPoisson(
      N,
      m;
      potential=case_potential,
      potential_gradient=case_gradient,
      potential_hessian=case_hessian,
      forcing=case_forcing,
      overlap=overlap,
      quadrature_rule=MIDPOINT_QUADRATURE,
      initial_guess=x -> 0.0,
      model=model,
    )
    energy = Energies.NonlinearEnergy(
      "Section 6.1 semilinear L-shaped problem", ea, ga, ha, ndofs
    )
    if isnothing(exact_values)
      exact_fe = interpolate_everywhere(exact_solution, U)
      exact_values = collect(get_free_dof_values(exact_fe))
    end

    for method in LSHAPE_FIG17B_METHODS
      result = lshape_fig17b_method(
        method, energy, dofs, core, initial; maxiter, tolerance
      )
      initial_residual = result.residual_history[1]
      for outer in eachindex(result.energy_history)
        push!(convergence.case, case.id)
        push!(convergence.beta, case.beta)
        push!(convergence.m, m)
        push!(convergence.method, string(method))
        push!(convergence.outer, outer - 1)
        push!(convergence.energy_gap, NaN)
        push!(
          convergence.relative_residual,
          result.residual_history[outer] / initial_residual,
        )
      end
      push!(summary.case, case.id)
      push!(summary.beta, case.beta)
      push!(summary.m, m)
      push!(summary.method, string(method))
      push!(summary.outer_iterations, length(result.energy_history) - 1)
      push!(
        summary.final_relative_residual,
        result.residual_history[end] / initial_residual,
      )
      push!(
        summary.relative_nodal_error,
        norm(result.u - exact_values) / norm(exact_values),
      )
      push!(work.case, case.id)
      push!(work.beta, case.beta)
      push!(work.m, m)
      push!(work.method, string(method))
      push!(work.outer_iterations, length(result.energy_history) - 1)
      push!(work.nonlinear_local_batches, result.nonlinear_local_batches)
      push!(work.linear_as_batches, result.linear_as_batches)
      push!(work.global_jacobian_products, result.global_jacobian_products)
      @printf(
        "  %-11s beta = %.0f, m = %d, %-25s: %3d outer, relres %.2e, nodal error %.2e\n",
        case.id,
        case.beta,
        m,
        string(method),
        summary.outer_iterations[end],
        summary.final_relative_residual[end],
        summary.relative_nodal_error[end],
      )
    end
  end

  savetable(files[1], convergence)
  savetable(files[2], summary)
  savetable(files[3], lshape_triangle_rows(model, N))
  savetable(files[4], lshape_partition_rows(model, N, ms, overlap))
  return savetable(files[5], work)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study11b()
end
