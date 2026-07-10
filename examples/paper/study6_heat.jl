# Study 6: heat equation as minimizing movements (backward Euler, each step a
# QuadraticEnergy solved with var_dd).
# (a) warm start (u0 = previous step) vs cold start (all-ones default)
# (b) sweep over the time step tau: per-step iterations, energy dissipation,
#     accuracy against a directly factorized backward Euler reference.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

# Two-bump initial condition (same as examples/heat_equation_dd.ipynb)
heat_ic(x, y) =
  exp(-100 * ((x - 0.25)^2 + (y - 0.35)^2)) +
  0.5 * exp(-100 * ((x - 0.7)^2 + (y - 0.6)^2))

function heat_trajectory(K, M, b, dofspar, u0, tau, nsteps; warm = true)
  A_tau = K + M / tau
  Af = factorize(A_tau)
  J(u) = 0.5 * dot(u, K * u) - dot(b, u)

  u = copy(u0)
  u_direct = copy(u0)
  iters = Int[]
  jvals = Float64[]
  relerrs = Float64[]
  for n = 1:nsteps
    e_step = Energies.QuadraticEnergy(A_tau, b + M * u / tau)
    result = Solvers.var_dd(
      e_step,
      dofspar;
      maxiter = 100,
      tol = 1e-10,
      u0 = warm ? u : nothing,
      verbose = false,
    )
    u = result[1]
    u_direct = Af \ (b + M * u_direct / tau)
    push!(iters, length(result[5]))
    push!(jvals, J(u))
    push!(relerrs, norm(u - u_direct) / norm(u_direct))
  end
  return iters, jvals, relerrs
end

function run_study6()
  files = ("study6_warmcold.csv", "study6_tau.csv")
  if !needs_run(files...)
    println("study6: cached, skipping")
    return
  end
  println("study6: heat equation")
  Random.seed!(1)

  N = SMALL ? 16 : 32
  m = 4
  T = SMALL ? 0.05 : 0.2

  K, M, b, dofspar, U = laplace_setup(N, m, 2)
  coords = [(i / N, j / N) for j = 1:N-1 for i = 1:N-1]
  u0 = [heat_ic(x, y) for (x, y) in coords]
  uinf = K \ b  # steady state
  Jinf = 0.5 * dot(uinf, K * uinf) - dot(b, uinf)

  # (a) warm vs cold start at tau = 5e-3
  tau_wc = 5e-3
  nsteps_wc = round(Int, T / tau_wc)
  rows_wc = (mode = String[], step = Int[], iters = Int[])
  for (mode, warm) in (("warm", true), ("cold", false))
    iters, _, _ = heat_trajectory(K, M, b, dofspar, u0, tau_wc, nsteps_wc; warm)
    append!(rows_wc.mode, fill(mode, nsteps_wc))
    append!(rows_wc.step, 1:nsteps_wc)
    append!(rows_wc.iters, iters)
    println("  $mode start: $(sum(iters)) total DD iters over $nsteps_wc steps")
  end
  savetable("study6_warmcold.csv", rows_wc)

  # (b) tau sweep, warm-started
  taus = SMALL ? [5e-3, 1e-2] : [1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2]
  rows_t = (
    tau = Float64[],
    step = Int[],
    t = Float64[],
    iters = Int[],
    energy_gap = Float64[],
    relerr = Float64[],
  )
  for tau in taus
    nsteps = round(Int, T / tau)
    iters, jvals, relerrs =
      heat_trajectory(K, M, b, dofspar, u0, tau, nsteps; warm = true)
    append!(rows_t.tau, fill(tau, nsteps))
    append!(rows_t.step, 1:nsteps)
    append!(rows_t.t, tau .* (1:nsteps))
    append!(rows_t.iters, iters)
    append!(rows_t.energy_gap, jvals .- Jinf)
    append!(rows_t.relerr, relerrs)
    @printf(
      "  tau = %.1e: %d steps, %d total DD iters\n",
      tau,
      nsteps,
      sum(iters)
    )
  end
  savetable("study6_tau.csv", rows_t)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study6()
end
