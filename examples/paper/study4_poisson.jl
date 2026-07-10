# Study 4: Poisson source problem (QuadraticEnergy).
# Convergence histories (energy gap to the direct solution and residual norm)
# for several subdomain counts.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function run_study4()
  if !needs_run("study4_poisson.csv")
    println("study4: cached, skipping")
    return
  end
  println("study4: Poisson convergence histories")
  Random.seed!(1)

  N = SMALL ? 20 : 40
  ms = SMALL ? [2] : [2, 4, 8]
  maxiter = SMALL ? 20 : 60

  rows = (
    m = Int[],
    iter = Int[],
    quantity = String[],
    value = Float64[],
  )

  for m in ms
    K, M, b, dofspar, U = laplace_setup(N, m, 2)
    ustar = K \ b
    e = Energies.QuadraticEnergy(K, b)
    Jstar = e(ustar)
    _, _, e_hist, _, resnorm_hist = Solvers.var_dd(
      e,
      dofspar;
      maxiter = maxiter,
      tol = 1e-12,
      verbose = false,
    )
    for (k, ev) in enumerate(e_hist)
      push!(rows.m, m)
      push!(rows.iter, k - 1)
      push!(rows.quantity, "energy_gap")
      push!(rows.value, ev - Jstar)
    end
    for (k, rn) in enumerate(resnorm_hist)
      push!(rows.m, m)
      push!(rows.iter, k)
      push!(rows.quantity, "resnorm")
      push!(rows.value, rn)
    end
    println("  m = $m done ($(length(resnorm_hist)) iters)")
  end
  savetable("study4_poisson.csv", rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study4()
end
