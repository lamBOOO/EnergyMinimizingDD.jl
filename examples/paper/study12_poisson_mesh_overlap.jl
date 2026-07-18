# Poisson mesh-size versus overlap study.
#
# Runs the variational DD solver for a fixed number of subdomains while varying
# the Cartesian mesh resolution N and overlap width in element layers. The
# summary CSV contains the first outer iteration at which the residual has been
# reduced by RELTOL relative to the zero-start residual norm ‖r₀‖₂=‖b‖₂.
#
# Run locally:
#   julia --project=. examples/paper/study12_poisson_mesh_overlap.jl
#
# Optional environment variables:
#   SMALL=1       smoke-test parameter set
#   FORCE=1       overwrite cached results
#   POISSON_N=... comma-separated mesh sizes, e.g. 20,40,80
#   POISSON_O=... comma-separated overlaps, e.g. 1,2,4,8

isdefined(Main, :PAPER_COMMON) || include("common.jl")

parse_int_list(name, default) =
  haskey(ENV, name) ? parse.(Int, split(ENV[name], ',')) : default

function run_study12()
  summary_name = "study12_poisson_mesh_overlap_summary.csv"
  history_name = "study12_poisson_mesh_overlap_history.csv"
  if !needs_run(summary_name, history_name)
    println("study12: cached, skipping")
    return
  end

  println("study12: Poisson mesh size versus overlap")
  Random.seed!(1)

  Ns = parse_int_list("POISSON_N", SMALL ? [20, 40] : [20, 40, 80])
  overlaps = parse_int_list("POISSON_O", SMALL ? [1, 2] : [1, 2, 4, 8])
  m = 4
  reltol = 1e-8
  maxiter = SMALL ? 80 : 250

  summary = (
    N = Int[],
    h = Float64[],
    m = Int[],
    overlap_layers = Int[],
    overlap_physical = Float64[],
    relative_overlap = Float64[],
    ndofs = Int[],
    iterations = Int[],
    converged = Bool[],
    final_relative_residual = Float64[],
    elapsed_seconds = Float64[],
  )
  history = (
    N = Int[],
    m = Int[],
    overlap_layers = Int[],
    iter = Int[],
    relative_residual = Float64[],
  )

  for N in Ns, overlap in overlaps
    K, _, b, dofspar, _ = laplace_setup(N, m, overlap)
    energy = Energies.QuadraticEnergy(K, b)
    u0 = zeros(size(K, 1))
    initial_residual = norm(K * u0 - b)

    elapsed = @elapsed begin
      _, _, _, _, residuals = Solvers.var_dd(
        energy,
        dofspar;
        maxiter = maxiter,
        tol = reltol * initial_residual,
        u0 = u0,
        verbose = false,
      )
    end

    relative_residuals = residuals ./ initial_residual
    hit = findfirst(<=(reltol), relative_residuals)
    converged = !isnothing(hit)
    iterations = converged ? hit : -1

    for (iteration, residual) in enumerate(relative_residuals)
      push!(history.N, N)
      push!(history.m, m)
      push!(history.overlap_layers, overlap)
      push!(history.iter, iteration)
      push!(history.relative_residual, residual)
    end

    push!(summary.N, N)
    push!(summary.h, 1 / N)
    push!(summary.m, m)
    push!(summary.overlap_layers, overlap)
    push!(summary.overlap_physical, overlap / N)
    push!(summary.relative_overlap, overlap / N)
    push!(summary.ndofs, size(K, 1))
    push!(summary.iterations, iterations)
    push!(summary.converged, converged)
    push!(summary.final_relative_residual, last(relative_residuals))
    push!(summary.elapsed_seconds, elapsed)

    iteration_text = converged ? string(iterations) : ">$maxiter"
    @printf(
      "  N=%3d, overlap=%2d layers (delta=%.4f): %s iterations, final relres=%.3e, %.2f s\n",
      N,
      overlap,
      overlap / N,
      iteration_text,
      last(relative_residuals),
      elapsed,
    )
  end

  savetable(summary_name, summary)
  savetable(history_name, history)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study12()
end
