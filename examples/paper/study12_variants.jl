# Study 12: focused comparison of EMDD algorithmic variants for Poisson.
#
# The experiment follows the weak-scaling setup of Study 8c: a Cartesian
# subdomain grid is refined together with the global mesh, keeping H/h and
# H/delta fixed while the number of subdomains grows.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

"Geometric mean of the final residual contractions (an observed linear rate)."
function terminal_contraction_rate(residuals; window=5)
  positive = filter(r -> isfinite(r) && r > 0, residuals)
  length(positive) >= 2 || return NaN
  count = min(window + 1, length(positive))
  tail = positive[(end - count + 1):end]
  return exp((log(tail[end]) - log(tail[1])) / (count - 1))
end

function run_study12()
  output = "study12_variants.csv"
  if !needs_run(output)
    println("study12: cached, skipping")
    return nothing
  end
  println("study12: weak scaling of method variants")

  roots = SMALL ? (2, 4) : (2, 4, 8)
  cells_per_subdomain_side = SMALL ? 5 : 10
  overlap = SMALL ? 1 : 2
  maxiter = SMALL ? 80 : 200
  relative_tolerance = SMALL ? 1e-7 : 1e-10

  # q=2 is used throughout so the comparison changes only the named flavor.
  methods = (
    ("emdd", "EMDD", :additive, :none, :none),
    ("memdd", "multiplicative EMDD", :multiplicative, :none, :none),
    ("remdd", "REMDD", :additive, :partition_of_unity, :none),
    (
      "emdd_multiplicity",
      "EMDD + multiplicity PoU",
      :additive,
      :none,
      :multiplicity,
    ),
    (
      "emdd_nicolaides",
      "EMDD + harmonic Nicolaides",
      :additive,
      :none,
      :harmonic,
    ),
    (
      "remdd_nicolaides",
      "REMDD + harmonic Nicolaides",
      :additive,
      :partition_of_unity,
      :harmonic,
    ),
  )

  rows = (
    method=String[],
    label=String[],
    N=Int[],
    m=Int[],
    overlap=Int[],
    cells_per_subdomain_side=Int[],
    H_over_delta=Float64[],
    iteration=Int[],
    relative_residual=Float64[],
    contraction_rate=Float64[],
  )

  for root in roots
    m = root^2
    N = cells_per_subdomain_side * root
    K, _, b, dofspar, _, core_dofs = laplace_setup(
      N, m, overlap; partitioning=:cartesian, return_core_partition=true
    )
    multiplicity_basis = Solvers.partition_of_unity_weights(dofspar, length(b))
    harmonic_basis = Solvers.nicolaides_coarse_basis(K, core_dofs, dofspar)
    initial = ones(length(b))
    initial_residual = norm(b - K * initial)
    tolerance = relative_tolerance * initial_residual

    for (method, label, sweep, restriction, coarse_kind) in methods
      coarse_basis = if coarse_kind == :multiplicity
        multiplicity_basis
      elseif coarse_kind == :harmonic
        harmonic_basis
      else
        nothing
      end
      result = Solvers.var_dd(
        Energies.QuadraticEnergy(K, b),
        dofspar;
        u0=initial,
        maxiter,
        tol=tolerance,
        history_depth=1,
        sweep,
        restriction,
        coarse_basis,
        verbose=false,
      )
      residuals = vcat(initial_residual, result[5])
      rate = terminal_contraction_rate(residuals)
      for (iteration, residual) in enumerate(residuals)
        push!(rows.method, method)
        push!(rows.label, label)
        push!(rows.N, N)
        push!(rows.m, m)
        push!(rows.overlap, overlap)
        push!(rows.cells_per_subdomain_side, cells_per_subdomain_side)
        push!(rows.H_over_delta, cells_per_subdomain_side / overlap)
        push!(rows.iteration, iteration - 1)
        push!(rows.relative_residual, residual / initial_residual)
        push!(rows.contraction_rate, rate)
      end
      @printf(
        "  m=%3d, N=%3d, %-29s: %3d sweeps, tail factor %.3f\n",
        m,
        N,
        label,
        length(residuals) - 1,
        rate,
      )
    end
  end
  return savetable(output, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study12()
end
