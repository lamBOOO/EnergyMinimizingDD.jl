# Linear-source comparisons used by Figure 12b of the paper.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

study8_diffusion(x) = 2 + sinpi(2 * x.data[1] + 3 * x.data[2])

function study8_sign_changing_exact_solution(x)
  x_coordinate, y_coordinate = x.data
  return exp(3 * x_coordinate * y_coordinate) *
         sinpi(x_coordinate) * sinpi(2 * y_coordinate)
end

function study8_sign_changing_forcing(x)
  x_coordinate, y_coordinate = x.data
  exponential = exp(3 * x_coordinate * y_coordinate)
  sine_x, sine_y = sinpi(x_coordinate), sinpi(2 * y_coordinate)
  cosine_x, cosine_y = cospi(x_coordinate), cospi(2 * y_coordinate)
  derivative_x = exponential * sine_y *
                 (3 * y_coordinate * sine_x + pi * cosine_x)
  derivative_y = exponential * sine_x *
                 (3 * x_coordinate * sine_y + 2pi * cosine_y)
  laplacian = exponential * (
    (9 * (x_coordinate^2 + y_coordinate^2) - 5pi^2) * sine_x * sine_y +
    6pi * y_coordinate * cosine_x * sine_y +
    12pi * x_coordinate * sine_x * cosine_y
  )
  diffusion_gradient = cospi(2 * x_coordinate + 3 * y_coordinate) *
                       (2pi * derivative_x + 3pi * derivative_y)
  return -study8_diffusion(x) * laplacian - diffusion_gradient
end

function study8_sign_changing_problem_setup(N, m, overlap; kwargs...)
  return FEMDiscretizations.FEM_Schroedinger(
    N, m;
    P=x -> 0.0,
    f=study8_sign_changing_forcing,
    diffusion=study8_diffusion,
    overlap,
    kwargs...,
  )
end

function ras_stationary(K, b, schwarz; maxsweeps, tol, x_sol)
  x = ones(size(K, 1))
  history = Tuple{Int,Float64,Float64}[]
  for iteration = 0:maxsweeps
    residual = norm(b - K * x)
    push!(history, (
      iteration * nsub(schwarz), residual, norm(x - x_sol) / norm(x_sol)
    ))
    residual < tol && break
    iteration == maxsweeps && break
    x .+= apply_RAS(schwarz, b - K * x)
  end
  return history
end

function iterative_solver_history(x, x0, initial_residual, residuals, m, x_sol)
  if isnothing(x_sol)
    return vcat(
      [(0, initial_residual)],
      [(iteration * m, residual) for
       (iteration, residual) in enumerate(residuals)],
    )
  end

  relative_error(y) = norm(y - x_sol) / norm(x_sol)
  final_iteration = length(residuals)
  # IterativeSolvers exposes every residual, but GMRES updates its solution
  # only at the end of an unrestarted cycle. Retain the exact endpoint errors
  # used by the study and mark unavailable intermediate errors explicitly.
  return vcat(
    [(0, initial_residual, relative_error(x0))],
    [(iteration * m, residual,
      iteration == final_iteration ? relative_error(x) : NaN) for
     (iteration, residual) in enumerate(residuals)],
  )
end

function pcg_as(K, b, schwarz; maxiter, tol, x_sol=nothing)
  m = nsub(schwarz)
  x0 = ones(size(K, 1))
  x = copy(x0)
  initial_residual = norm(b - K * x)
  _, convergence = cg!(
    x, K, b;
    Pl=ASPreconditioner(schwarz),
    maxiter,
    abstol=tol,
    reltol=0.0,
    log=true,
  )
  return iterative_solver_history(
    x, x0, initial_residual, convergence[:resnorm], m, x_sol,
  )
end

"Unrestarted right-preconditioned GMRES with the RAS preconditioner."
function gmres_ras(K, b, schwarz; maxiter, tol, x_sol=nothing)
  m = nsub(schwarz)
  x0 = ones(length(b))
  x = copy(x0)
  initial_residual = norm(b - K * x)
  _, convergence = gmres!(
    x, K, b;
    Pr=RASPreconditioner(schwarz),
    restart=maxiter,
    maxiter,
    abstol=tol,
    reltol=0.0,
    orth_meth=DGKS(),
    log=true,
  )
  return iterative_solver_history(
    x, x0, initial_residual, convergence[:resnorm], m, x_sol,
  )
end

function var_dd_linear_history(K, b, dofs; maxiter, tol, x_sol=nothing, kwargs...)
  _, _, _, iterates, residuals = Solvers.var_dd(
    Energies.QuadraticEnergy(K, b), dofs;
    maxiter,
    tol,
    verbose=false,
    kwargs...,
  )
  m = length(dofs)
  initial = norm(b - K * ones(length(b)))
  if isnothing(x_sol)
    return vcat(
      [(0, initial)],
      [(k * m, residual) for (k, residual) in enumerate(residuals)],
    )
  end
  relative_errors = [norm(x - x_sol) / norm(x_sol) for x in iterates]
  return vcat(
    [(0, initial, relative_errors[1])],
    [(k * m, residual, relative_errors[k+1]) for
      (k, residual) in enumerate(residuals)],
  )
end

function run_study8_problem(file, solution_file, label, setup)
  curves_needed = needs_run(file)
  solution_needed = needs_run(solution_file)
  if !curves_needed && !solution_needed
    println("study8 $label: cached, skipping")
    return
  end
  Random.seed!(1)
  N = SMALL ? 16 : 64
  ms = SMALL ? [4] : [4, 16, 64]
  overlap = 2
  relative_tolerance = SMALL ? 1e-7 : 1e-10
  maxiterations = SMALL ? 50 : 400
  owners = Dict(m => metis_cell_owners(N, m) for m in ms)
  rows = (
    method=String[], m=Int[], solves=Int[], resnorm=Float64[],
    relative_residual=Float64[], relative_error=Float64[],
  )
  solution_rows = (N=Int[], idx=Int[], value=Float64[])
  function record(method, m, history)
    initial = first(history)[2]
    for (solves, residual, error) in history
      push!(rows.method, method)
      push!(rows.m, m)
      push!(rows.solves, solves)
      push!(rows.resnorm, residual)
      push!(rows.relative_residual, residual / initial)
      push!(rows.relative_error, error)
    end
  end
  for m in ms
    !curves_needed && m != first(ms) && continue
    K, _, b, dofs, _, core = setup(
      N, m, overlap; cell_owners=owners[m], return_core_partition=true
    )
    solution = K \ b
    if solution_needed && m == first(ms)
      for index in eachindex(solution)
        push!(solution_rows.N, N)
        push!(solution_rows.idx, index)
        push!(solution_rows.value, solution[index])
      end
    end
    curves_needed || continue
    schwarz = schwarz_setup(K, dofs; core_dofs=core)
    tolerance = relative_tolerance * norm(b - K * ones(length(b)))
    record("var_dd_additive", m, var_dd_linear_history(
      K, b, dofs; maxiter=maxiterations, tol=tolerance, x_sol=solution
    ))
    record("var_dd_additive_history", m, var_dd_linear_history(
      K, b, dofs; maxiter=maxiterations, tol=tolerance, x_sol=solution,
      history_depth=1,
    ))
    record("ras", m, ras_stationary(
      K, b, schwarz; maxsweeps=maxiterations, tol=tolerance, x_sol=solution
    ))
    record("pcg_as", m, pcg_as(
      K, b, schwarz; maxiter=maxiterations, tol=tolerance, x_sol=solution
    ))
    record("gmres_ras", m, gmres_ras(
      K, b, schwarz; maxiter=maxiterations, tol=tolerance, x_sol=solution
    ))
    println("  $label m = $m done")
  end
  curves_needed && savetable(file, rows)
  solution_needed && savetable(solution_file, solution_rows)
end

function write_study8_partitions()
  file = "study8_partitions.csv"
  !needs_run(file) && return
  N = SMALL ? 16 : 64
  ms = SMALL ? [4] : [4, 16, 64]
  rows = (m=Int[], N=Int[], idx=Int[], owner=Int[], mult=Int[])
  for m in ms
    owner, multiplicity = metis_cell_partition(N, m, 2)
    for index in eachindex(owner)
      push!(rows.m, m)
      push!(rows.N, N)
      push!(rows.idx, index)
      push!(rows.owner, owner[index])
      push!(rows.mult, multiplicity[index])
    end
  end
  savetable(file, rows)
end

function run_study8()
  run_study8_problem(
    "study8_linear_cmp_poisson.csv", "study8_linear_solution_poisson.csv",
    "Poisson", laplace_setup,
  )
  run_study8_problem(
    "study8_linear_cmp_sign_changing.csv",
    "study8_linear_solution_sign_changing.csv", "variable diffusion",
    study8_sign_changing_problem_setup,
  )
  write_study8_partitions()
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_study8()
