# Control study: Poisson h-robustness with a fixed structured 2x2 partition.
#
# Unlike study12, the four nonoverlapping cores are identical geometric
# quadrants for every N. For each prescribed physical overlap delta, the number
# of graph-overlap layers is delta*N, so both H=1/2 and delta remain fixed.
#
# Run:
#   julia --project=. examples/paper/study13_poisson_structured_overlap.jl

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function structured_quadrant_dofs(N::Int, overlap_layers::Int)
  iseven(N) || throw(ArgumentError("N must be even for the 2x2 partition"))

  model = CartesianDiscreteModel(
    (0, 1.0, 0, 1.0),
    (N, N);
    isperiodic = (false, false),
  )
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = TestFESpace(model, reffe, dirichlet_tags = ["boundary"])

  # Cartesian cell numbering is x-fastest. These owners define the same four
  # quadrants for every refinement level.
  owners = Int32[
    1 + (ix > N ÷ 2) + 2 * (iy > N ÷ 2)
    for iy in 1:N for ix in 1:N
  ]
  graph = GridapDistributed.compute_cell_graph(model)
  element_partition =
    FEMDiscretizations.create_elements_partition(owners, 4)
  FEMDiscretizations.create_overlapping_elements_partition!(
    element_partition,
    graph,
    4,
    overlap_layers,
  )
  return FEMDiscretizations.create_dofs_partition(element_partition, V)
end

function run_study13()
  summary_name = "study13_poisson_structured_overlap_summary.csv"
  history_name = "study13_poisson_structured_overlap_history.csv"
  if !needs_run(summary_name, history_name)
    println("study13: cached, skipping")
    return
  end

  println("study13: Poisson h-robustness, fixed structured 2x2 partition")
  Random.seed!(1)

  Ns = SMALL ? [20, 40] : [20, 40, 80]
  physical_overlaps = SMALL ? [0.1] : [0.05, 0.1]
  reltol = 1e-8
  maxiter = 250
  m = 4
  H = 0.5

  summary = (
    N = Int[],
    h = Float64[],
    m = Int[],
    H = Float64[],
    overlap_layers = Int[],
    delta = Float64[],
    H_over_delta = Float64[],
    ndofs = Int[],
    iterations = Int[],
    converged = Bool[],
    final_relative_residual = Float64[],
    elapsed_seconds = Float64[],
  )
  history = (
    N = Int[],
    delta = Float64[],
    overlap_layers = Int[],
    iter = Int[],
    relative_residual = Float64[],
  )

  for delta in physical_overlaps, N in Ns
    layers_exact = delta * N
    layers = round(Int, layers_exact)
    isapprox(layers, layers_exact; atol = 100eps(Float64)) ||
      error("delta*N must be integral, got delta=$delta and N=$N")

    # Assembly is independent of the partition. Ignore the METIS dofs returned
    # by the existing helper and replace them with fixed quadrant dofs.
    K, _, b, _, _ = laplace_setup(N, m, 0)
    dofspar = structured_quadrant_dofs(N, layers)
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
      push!(history.delta, delta)
      push!(history.overlap_layers, layers)
      push!(history.iter, iteration)
      push!(history.relative_residual, residual)
    end

    push!(summary.N, N)
    push!(summary.h, 1 / N)
    push!(summary.m, m)
    push!(summary.H, H)
    push!(summary.overlap_layers, layers)
    push!(summary.delta, delta)
    push!(summary.H_over_delta, H / delta)
    push!(summary.ndofs, size(K, 1))
    push!(summary.iterations, iterations)
    push!(summary.converged, converged)
    push!(summary.final_relative_residual, last(relative_residuals))
    push!(summary.elapsed_seconds, elapsed)

    iteration_text = converged ? string(iterations) : ">$maxiter"
    @printf(
      "  delta=%.2f, N=%3d, layers=%2d, H/delta=%.1f: %s iterations, relres=%.3e, %.2f s\n",
      delta,
      N,
      layers,
      H / delta,
      iteration_text,
      last(relative_residuals),
      elapsed,
    )
  end

  savetable(summary_name, summary)
  savetable(history_name, history)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study13()
end
