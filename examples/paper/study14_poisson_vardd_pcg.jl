# Diagnostic control: varDD versus PCG+AS on identical structured quadrants.
#
# Both methods use u0=0 and the same local matrices on a fixed geometric 2x2
# partition. We record the relative Euclidean residual, relative quadratic
# energy gap, and relative A-norm error. The A-norm error is the convergence
# quantity directly controlled by the classical PCG condition-number estimate.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :structured_quadrant_dofs) ||
  include("study13_poisson_structured_overlap.jl")

function diagnostic_metrics(K, b, solution, reference, initial_residual, initial_gap)
  residual = norm(b - K * solution) / initial_residual
  error = solution - reference
  gap = 0.5 * dot(error, K * error)
  relative_gap = gap / initial_gap
  relative_A_error = sqrt(max(relative_gap, 0.0))
  return residual, relative_gap, relative_A_error
end

function pcg_as_diagnostic(K, b, S, reference; maxiter, tolerance)
  x = zeros(size(K, 1))
  initial_residual = norm(b)
  initial_error = x - reference
  initial_gap = 0.5 * dot(initial_error, K * initial_error)

  iterations = Int[0]
  residuals = Float64[1.0]
  energy_gaps = Float64[1.0]
  A_errors = Float64[1.0]

  r = copy(b)
  z = apply_AS(S, r)
  p = copy(z)
  rz = dot(r, z)

  for iteration in 1:maxiter
    Kp = K * p
    alpha = rz / dot(p, Kp)
    x .+= alpha .* p
    r .-= alpha .* Kp

    relative_residual, relative_gap, relative_A_error = diagnostic_metrics(
      K, b, x, reference, initial_residual, initial_gap
    )
    push!(iterations, iteration)
    push!(residuals, relative_residual)
    push!(energy_gaps, relative_gap)
    push!(A_errors, relative_A_error)
    relative_residual <= tolerance && break

    z = apply_AS(S, r)
    rz_new = dot(r, z)
    p .= z .+ (rz_new / rz) .* p
    rz = rz_new
  end
  return iterations, residuals, energy_gaps, A_errors
end

function vardd_diagnostic(K, b, dofspar, reference; maxiter, tolerance)
  u0 = zeros(size(K, 1))
  initial_residual = norm(b)
  initial_error = u0 - reference
  initial_gap = 0.5 * dot(initial_error, K * initial_error)

  _, _, _, solutions, _ = Solvers.var_dd(
    Energies.QuadraticEnergy(K, b),
    dofspar;
    maxiter=maxiter,
    tol=tolerance * initial_residual,
    u0=u0,
    verbose=false,
  )

  iterations = collect(0:length(solutions)-1)
  residuals = Float64[]
  energy_gaps = Float64[]
  A_errors = Float64[]
  for solution in solutions
    relative_residual, relative_gap, relative_A_error = diagnostic_metrics(
      K, b, solution, reference, initial_residual, initial_gap
    )
    push!(residuals, relative_residual)
    push!(energy_gaps, relative_gap)
    push!(A_errors, relative_A_error)
  end
  return iterations, residuals, energy_gaps, A_errors
end

first_below(values, tolerance) = begin
  hit = findfirst(<=(tolerance), values)
  isnothing(hit) ? -1 : hit - 1
end

function run_study14()
  history_name = "study14_poisson_vardd_pcg_history.csv"
  summary_name = "study14_poisson_vardd_pcg_summary.csv"
  if !needs_run(history_name, summary_name)
    println("study14: cached, skipping")
    return
  end

  println("study14: structured Poisson, varDD versus PCG+AS")
  Ns = SMALL ? [20, 40] : [20, 40, 80]
  physical_overlaps = SMALL ? [0.1] : [0.05, 0.1]
  tolerance = 1e-8
  maxiter = 150
  H = 0.5
  m = 4

  history = (
    method=String[],
    N=Int[],
    delta=Float64[],
    H_over_delta=Float64[],
    iteration=Int[],
    relative_residual=Float64[],
    relative_energy_gap=Float64[],
    relative_A_error=Float64[],
  )
  summary = (
    method=String[],
    N=Int[],
    delta=Float64[],
    H_over_delta=Float64[],
    residual_iterations=Int[],
    energy_gap_iterations=Int[],
    A_error_iterations=Int[],
    final_relative_residual=Float64[],
    final_relative_energy_gap=Float64[],
    final_relative_A_error=Float64[],
  )

  for delta in physical_overlaps, N in Ns
    layers = round(Int, delta * N)
    isapprox(layers, delta * N; atol=100eps(Float64)) ||
      error("delta*N must be integral")

    K, _, b, _, _ = laplace_setup(N, m, 0)
    dofspar = structured_quadrant_dofs(N, layers)
    schwarz = schwarz_setup(K, dofspar)
    reference = K \ b

    for (method, runner) in (
      ("varDD", () -> vardd_diagnostic(
        K, b, dofspar, reference; maxiter=maxiter, tolerance=tolerance
      )),
      ("PCG+AS", () -> pcg_as_diagnostic(
        K, b, schwarz, reference; maxiter=maxiter, tolerance=tolerance
      )),
    )
      iterations, residuals, gaps, A_errors = runner()
      for index in eachindex(iterations)
        push!(history.method, method)
        push!(history.N, N)
        push!(history.delta, delta)
        push!(history.H_over_delta, H / delta)
        push!(history.iteration, iterations[index])
        push!(history.relative_residual, residuals[index])
        push!(history.relative_energy_gap, gaps[index])
        push!(history.relative_A_error, A_errors[index])
      end

      push!(summary.method, method)
      push!(summary.N, N)
      push!(summary.delta, delta)
      push!(summary.H_over_delta, H / delta)
      push!(summary.residual_iterations, first_below(residuals, tolerance))
      push!(summary.energy_gap_iterations, first_below(gaps, tolerance))
      push!(summary.A_error_iterations, first_below(A_errors, tolerance))
      push!(summary.final_relative_residual, last(residuals))
      push!(summary.final_relative_energy_gap, last(gaps))
      push!(summary.final_relative_A_error, last(A_errors))

      @printf(
        "  %-6s delta=%.2f N=%3d: residual=%d, energy gap=%d, A-error=%d iterations\n",
        method,
        delta,
        N,
        first_below(residuals, tolerance),
        first_below(gaps, tolerance),
        first_below(A_errors, tolerance),
      )
    end
  end

  savetable(history_name, history)
  savetable(summary_name, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study14()
end
