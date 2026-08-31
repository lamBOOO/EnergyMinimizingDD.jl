# Study 8c: weak scaling of one- and two-level EMDD/REMDD for Poisson.
#
# A sqrt(m) by sqrt(m) Cartesian subdomain grid is refined together with the
# global mesh. Each core keeps the same number of cells per direction and the
# overlap keeps the same number of fine-cell layers. Consequently H/h and
# H/delta remain fixed while the global problem and coarse space grow.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :var_dd_linear_history) || include("study8_linear_cmp.jl")

function run_study8_weak_scaling()
  output = "study8_weak_scaling.csv"
  needs_run(output) || return println("study8 weak scaling: cached, skipping")
  println("study8 weak scaling: Poisson with harmonic Nicolaides coarse space")

  roots = SMALL ? (2, 4) : (2, 4, 8)
  cells_per_subdomain_side = SMALL ? 5 : 10
  overlap = SMALL ? 1 : 2
  relative_tolerance = SMALL ? 1e-7 : 1e-10
  maxiter = SMALL ? 100 : 200
  methods = (
    ("emdd_q1", "EMDD", 1, :none, "none"),
    ("emdd_q1_multiplicity", "EMDD", 1, :none, "multiplicity"),
    ("emdd_q1_nicolaides", "EMDD", 1, :none, "harmonic"),
    ("emdd_q2", "EMDD", 2, :none, "none"),
    ("emdd_q2_multiplicity", "EMDD", 2, :none, "multiplicity"),
    ("emdd_q2_nicolaides", "EMDD", 2, :none, "harmonic"),
    ("remdd_q1", "REMDD", 1, :partition_of_unity, "none"),
    ("remdd_q1_multiplicity", "REMDD", 1, :partition_of_unity, "multiplicity"),
    ("remdd_q1_nicolaides", "REMDD", 1, :partition_of_unity, "harmonic"),
    ("remdd_q2", "REMDD", 2, :partition_of_unity, "none"),
    ("remdd_q2_multiplicity", "REMDD", 2, :partition_of_unity, "multiplicity"),
    ("remdd_q2_nicolaides", "REMDD", 2, :partition_of_unity, "harmonic"),
  )
  rows = (
    method=String[],
    family=String[],
    q=Int[],
    coarse=Bool[],
    coarse_kind=String[],
    m=Int[],
    N=Int[],
    ndofs=Int[],
    overlap=Int[],
    cells_per_subdomain_side=Int[],
    H_over_delta=Float64[],
    iteration=Int[],
    local_batches=Int[],
    local_solves=Int[],
    resnorm=Float64[],
    relative_residual=Float64[],
  )

  for root in roots
    m = root^2
    N = cells_per_subdomain_side * root
    K, _, b, dofspar, _, core_dofs = laplace_setup(
      N, m, overlap; partitioning=:cartesian, return_core_partition=true
    )
    multiplicity_basis = Solvers.partition_of_unity_weights(dofspar, length(b))
    harmonic_basis = Solvers.nicolaides_coarse_basis(K, core_dofs, dofspar)
    initial_residual = norm(b - K * ones(length(b)))
    tolerance = relative_tolerance * initial_residual

    for (method, family, q, restriction, coarse_kind) in methods
      coarse_basis = if coarse_kind == "harmonic"
        harmonic_basis
      elseif coarse_kind == "multiplicity"
        multiplicity_basis
      else
        nothing
      end
      history = var_dd_linear_history(
        K,
        b,
        dofspar;
        maxiter,
        tol=tolerance,
        history_depth=q - 1,
        restriction,
        coarse_basis,
      )
      for (iteration, (solves, residual)) in enumerate(history)
        push!(rows.method, method)
        push!(rows.family, family)
        push!(rows.q, q)
        push!(rows.coarse, coarse_kind != "none")
        push!(rows.coarse_kind, coarse_kind)
        push!(rows.m, m)
        push!(rows.N, N)
        push!(rows.ndofs, length(b))
        push!(rows.overlap, overlap)
        push!(rows.cells_per_subdomain_side, cells_per_subdomain_side)
        push!(rows.H_over_delta, cells_per_subdomain_side / overlap)
        push!(rows.iteration, iteration - 1)
        push!(rows.local_batches, solves ÷ m)
        push!(rows.local_solves, solves)
        push!(rows.resnorm, residual)
        push!(rows.relative_residual, residual / initial_residual)
      end
      @printf(
        "  m=%3d, N=%3d, %-25s: %3d batches\n",
        m,
        N,
        method,
        length(history) - 1,
      )
    end
  end
  return savetable(output, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study8_weak_scaling()
end
