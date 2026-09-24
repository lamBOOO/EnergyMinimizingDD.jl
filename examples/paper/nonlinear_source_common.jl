# Baselines used by the two semilinear figures in the final paper.

const SEMILINEAR_SOURCE_METHODS = (
  :anderson_ras,
  :nonlinear_cg_optim_as,
  :newton_pcg_as_4,
  :var_dd,
  :var_dd_history,
  :var_dd_quadratic,
  :newton_pcg_as_1,
  :var_dd_quadratic_history,
  :newton_pcg_as_2,
)

Base.@kwdef mutable struct NonlinearSourceWork
  nonlinear_local_batches::Int = 0
  linear_as_batches::Int = 0
  global_jacobian_products::Int = 0
end

nonlinear_source_result(u, energies, residuals, work) = (
  u=u,
  energy_history=energies,
  residual_history=residuals,
  nonlinear_local_batches=work.nonlinear_local_batches,
  linear_as_batches=work.linear_as_batches,
  global_jacobian_products=work.global_jacobian_products,
)

function nonlinear_energy_line_search(energy, u, direction; c=1e-4)
  gradient = Energies.gradient(energy, u)
  slope = dot(gradient, direction)
  if !isfinite(slope) || slope >= 0
    direction = -gradient
    slope = -dot(gradient, gradient)
  end
  initial = Energies.energy(energy, u)
  step = 1.0
  while step >= 2.0^-30
    trial = u + step * direction
    Energies.energy(energy, trial) <= initial + c * step * slope && return trial
    step *= 0.5
  end
  return copy(u)
end

function nonlinear_apply_as(factors, dofs, residual)
  result = zeros(length(residual))
  for (factor, indices) in zip(factors, dofs)
    result[indices] .+= factor \ residual[indices]
  end
  return result
end

mutable struct NonlinearOptimASPreconditioner{E,D}
  energy::E
  dofs::D
  factors::Vector{Any}
  hessian::Any
  applications::Int
  metric_products::Int
end

function NonlinearOptimASPreconditioner(energy, dofs, initial)
  preconditioner = NonlinearOptimASPreconditioner(
    energy, dofs, Any[], nothing, 0, 0
  )
  nonlinear_optim_as_prepare!(preconditioner, initial)
  return preconditioner
end

function nonlinear_optim_as_prepare!(preconditioner, u)
  H = Energies.hessian(preconditioner.energy, u)
  preconditioner.hessian = H
  preconditioner.factors = Any[
    cholesky(Symmetric(sparse(H[d, d]))) for d in preconditioner.dofs
  ]
  return preconditioner
end

function LinearAlgebra.ldiv!(result, preconditioner::NonlinearOptimASPreconditioner, rhs)
  result .= nonlinear_apply_as(preconditioner.factors, preconditioner.dofs, rhs)
  preconditioner.applications += 1
  return result
end

function LinearAlgebra.dot(x::AbstractVector,
                           preconditioner::NonlinearOptimASPreconditioner,
                           y::AbstractVector)
  preconditioner.metric_products += 1
  return dot(x, preconditioner.hessian * y)
end

function nonlinear_source_optim_ncg_as(
  energy, subdomains; u0, maxiter, tolerance,
)
  dofs = [collect(Int, indices) for indices in subdomains]
  initial_residual = norm(Energies.gradient(energy, u0))
  energies = Float64[]
  residuals = Float64[]
  last_recorded = Ref(copy(u0))
  function record_iterate(state)
    u = state.metadata["x"]
    push!(energies, Energies.energy(energy, u))
    residual = norm(Energies.gradient(energy, u))
    push!(residuals, residual)
    last_recorded[] = copy(u)
    return residual <= tolerance * initial_residual
  end
  objective(u) = Energies.energy(energy, u)
  gradient!(storage, u) = copyto!(storage, Energies.gradient(energy, u))
  preconditioner = NonlinearOptimASPreconditioner(energy, dofs, u0)
  method = Optim.ConjugateGradient(
    P=preconditioner, precondprep=nonlinear_optim_as_prepare!
  )
  options = Optim.Options(
    iterations=maxiter, x_abstol=0.0, x_reltol=0.0, f_abstol=NaN,
    f_reltol=NaN, g_abstol=0.0, allow_f_increases=true,
    extended_trace=true, callback=record_iterate, show_warnings=false,
  )
  result = Optim.optimize(objective, gradient!, copy(u0), method, options)
  u = copy(Optim.minimizer(result))
  if isempty(energies) || u != last_recorded[]
    push!(energies, Energies.energy(energy, u))
    push!(residuals, norm(Energies.gradient(energy, u)))
  end
  work = NonlinearSourceWork(
    linear_as_batches=preconditioner.applications,
    global_jacobian_products=preconditioner.metric_products,
  )
  return nonlinear_source_result(u, energies, residuals, work)
end

function nonlinear_pcg_as(A, rhs, dofs; maxiter, relative_tolerance=0.0)
  factors = [cholesky(Symmetric(sparse(A[d, d]))) for d in dofs]
  x = zeros(length(rhs))
  r = copy(rhs)
  initial = norm(r)
  initial == 0 && return (x=x, iterations=0, matvecs=0)
  z = nonlinear_apply_as(factors, dofs, r)
  p = copy(z)
  rz = dot(r, z)
  iterations = 0
  for iteration = 1:maxiter
    Ap = A * p
    denominator = dot(p, Ap)
    (!isfinite(denominator) || denominator <= 0) && break
    alpha = rz / denominator
    x .+= alpha .* p
    r .-= alpha .* Ap
    iterations = iteration
    norm(r) <= relative_tolerance * initial && break
    z = nonlinear_apply_as(factors, dofs, r)
    rz_new = dot(r, z)
    (!isfinite(rz_new) || rz_new <= 0) && break
    p .= z .+ (rz_new / rz) .* p
    rz = rz_new
  end
  return (x=x, iterations=iterations, matvecs=iterations)
end

function nonlinear_ras_map(energy, u, dofs, restricted_dofs)
  candidates = [Solvers.nonlinear_local_minimize(
    energy, u, indices;
    relative_tolerance=1e-11, absolute_tolerance=1e-12, maxiter=80,
  ).u for indices in dofs]
  direction = zeros(length(u))
  for i in eachindex(dofs), degree in restricted_dofs[i]
    direction[degree] = candidates[i][degree] - u[degree]
  end
  next = Solvers.minimize_affine_corrections(
    energy, u, reshape(direction, :, 1);
    relative_tolerance=1e-12, absolute_tolerance=1e-13,
  ).u
  return next
end

function nonlinear_source_vardd(
  energy, subdomains; u0, maxiter, tolerance, history_depth,
  quadratic_model=false,
)
  initial = norm(Energies.gradient(energy, u0))
  result = Solvers.var_dd(
    energy, subdomains;
    u0, maxiter, tol=tolerance * initial, history_depth,
    quadratic_model, verbose=false,
  )
  u, energies, residuals = result.u, result.energy_history,
    result.residual_history
  batches = length(energies) - 1
  work = NonlinearSourceWork(
    nonlinear_local_batches=quadratic_model ? 0 : batches,
    linear_as_batches=quadratic_model ? batches : 0,
  )
  return nonlinear_source_result(u, energies, vcat(initial, residuals), work)
end

function nonlinear_source_anderson_ras(
  energy, subdomains, core_subdomains;
  u0, maxiter, tolerance, history_depth,
)
  dofs = [collect(Int, indices) for indices in subdomains]
  restricted = FEMDiscretizations.create_balanced_disjoint_dofs_partition(
    core_subdomains, length(u0)
  )
  initial = norm(Energies.gradient(energy, u0))
  fixed_point_residual(u, _) = nonlinear_ras_map(energy, u, dofs, restricted) - u
  problem = NonlinearProblem(fixed_point_residual, copy(u0))
  algorithm = NonlinearSolve.FixedPointAccelerationJL(
    algorithm=:Anderson, m=history_depth
  )
  solution = NonlinearSolve.solve(
    problem, algorithm; abstol=0.01 * tolerance * initial, maxiters=maxiter
  )
  package_result = solution.original
  inputs = package_result.Inputs_
  u = ismissing(package_result.FixedPoint_) ?
      copy(package_result.Outputs_[:, end]) : copy(package_result.FixedPoint_)
  iterates = [copy(inputs[:, j]) for j in axes(inputs, 2)]
  push!(iterates, u)
  energies = [Energies.energy(energy, iterate) for iterate in iterates]
  residuals = [norm(Energies.gradient(energy, iterate)) for iterate in iterates]
  work = NonlinearSourceWork(
    nonlinear_local_batches=package_result.Iterations_
  )
  return nonlinear_source_result(u, energies, residuals, work)
end

function nonlinear_source_newton_pcg_as(
  energy, subdomains; u0, maxiter, tolerance, inner_iterations,
)
  dofs = [collect(Int, indices) for indices in subdomains]
  u = copy(u0)
  energies = [Energies.energy(energy, u)]
  residuals = [norm(Energies.gradient(energy, u))]
  initial = residuals[1]
  work = NonlinearSourceWork()
  for _ = 1:maxiter
    residuals[end] <= tolerance * initial && break
    linear = nonlinear_pcg_as(
      Energies.hessian(energy, u), -Energies.gradient(energy, u), dofs;
      maxiter=inner_iterations,
    )
    work.linear_as_batches += linear.iterations
    work.global_jacobian_products += linear.matvecs
    u = nonlinear_energy_line_search(energy, u, linear.x)
    push!(energies, Energies.energy(energy, u))
    push!(residuals, norm(Energies.gradient(energy, u)))
  end
  return nonlinear_source_result(u, energies, residuals, work)
end

function triangle_partition_rows(N, ms, overlap)
  model = simplexify(CartesianDiscreteModel(
    (0, 1.0, 0, 1.0), (N, N); isperiodic=(false, false)
  ))
  coordinates = get_cell_coordinates(Triangulation(model))
  graph = GridapDistributed.compute_cell_graph(model, 1)
  rows = (
    m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[],
    x1=Float64[], y1=Float64[], x2=Float64[], y2=Float64[],
    x3=Float64[], y3=Float64[],
  )
  for m in ms
    owner = Int.(Metis.partition(graph, m))
    partition = FEMDiscretizations.create_elements_partition(Int32.(owner), m)
    FEMDiscretizations.create_overlapping_elements_partition!(
      partition, graph, m, overlap
    )
    multiplicity = zeros(Int, length(owner))
    for elements in partition, element in elements
      multiplicity[element] += 1
    end
    for index in eachindex(owner)
      points = coordinates[index]
      push!(rows.m, m); push!(rows.N, N); push!(rows.idx, index)
      push!(rows.owner, owner[index]); push!(rows.mult, multiplicity[index])
      push!(rows.x1, points[1][1]); push!(rows.y1, points[1][2])
      push!(rows.x2, points[2][1]); push!(rows.y2, points[2][2])
      push!(rows.x3, points[3][1]); push!(rows.y3, points[3][2])
    end
  end
  return rows
end
