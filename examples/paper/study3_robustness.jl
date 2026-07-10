# Study 3: robustness and timing.
# Sweep N x m x overlap for the Laplace EVP and the Poisson problem; record
# iteration counts everywhere and median wall-clock timings on the overlap=2
# slice (the one shown in the timing figure).
#
# EVP tolerance 1e-4 matches the historical matrix_study.jl runs (Overlap*.csv)
# so the iteration counts are directly comparable.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function run_study3()
  if !needs_run("study3_robustness.csv")
    println("study3: cached, skipping")
    return
  end
  println("study3: robustness sweep (the long one)")
  Random.seed!(1)

  Ns = SMALL ? [10, 20] : [20, 40, 80]
  ms = SMALL ? [2] : [2, 4, 8]
  olaps = SMALL ? [1, 2] : [1, 2, 4]
  repeats = SMALL ? 1 : 3

  rows = (
    problem = String[],
    N = Int[],
    m = Int[],
    overlap = Int[],
    ndofs = Int[],
    min_sub = Int[],
    max_sub = Int[],
    iters = Int[],
    time_s = Float64[],
  )

  for N in Ns, m in ms, olap in olaps
    K, M, b, dofspar, U = laplace_setup(N, m, olap)
    ndofs = size(K, 1)
    subs = length.(dofspar)

    solvers = (
      evp = () -> Solvers.var_dd(
        Energies.GeneralizedRayleighQuotient(K, M),
        dofspar;
        maxiter = 200,
        tol = 1e-4,
        verbose = false,
      ),
      poisson = () -> Solvers.var_dd(
        Energies.QuadraticEnergy(K, b),
        dofspar;
        maxiter = 200,
        tol = 1e-8,
        verbose = false,
      ),
    )

    for (problem, solve) in pairs(solvers)
      # Timings only on the overlap = 2 slice (used in the timing figure);
      # other configs run once for the iteration counts.
      if olap == 2
        t, result = timed_median(solve; repeats = repeats)
      else
        t, result = NaN, solve()
      end
      iters = length(result[5])
      push!(rows.problem, String(problem))
      push!(rows.N, N)
      push!(rows.m, m)
      push!(rows.overlap, olap)
      push!(rows.ndofs, ndofs)
      push!(rows.min_sub, minimum(subs))
      push!(rows.max_sub, maximum(subs))
      push!(rows.iters, iters)
      push!(rows.time_s, t)
      @printf(
        "  %-8s N=%3d m=%2d overlap=%d: %3d iters%s\n",
        problem,
        N,
        m,
        olap,
        iters,
        isnan(t) ? "" : @sprintf(", %.3f s", t)
      )
    end
  end
  savetable("study3_robustness.csv", rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study3()
end
