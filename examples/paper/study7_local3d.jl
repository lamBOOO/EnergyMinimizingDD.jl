# Study 7: Schroedinger EVP — evolution of the subdomain (local) solutions.
# Runs var_dd with save_local_updates = true and stores, for selected
# iterations, the current iterate and each subdomain's local update
# (u_next_i - u_cur, supported inside subdomain i). Visualized as a 3D
# surface series in fig11.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function run_study7()
  if !needs_run("study7_local3d.csv")
    println("study7: cached, skipping")
    return
  end
  println("study7: Schroedinger local updates (3D series)")
  Random.seed!(1)

  N = SMALL ? 16 : 32
  m = 4
  overlap = 2

  K, M, b, dofspar, U = schroedinger_setup(N, m, overlap)
  e = Energies.GeneralizedRayleighQuotient(K, M)
  u, lambda, e_hist, sol_hist, resnorm_hist, local_update_hist = Solvers.var_dd(
    e,
    dofspar;
    maxiter = SMALL ? 15 : 60,
    tol = 1e-10,
    verbose = false,
    save_local_updates = true,
  )

  niters = length(local_update_hist)
  # Early, mid and (near-)final iterations
  ks = sort(unique(clamp.([1, 2, 5, niters], 1, niters)))

  ndofs = size(K, 1)
  rows = (
    N = Int[],
    iter = Int[],
    kind = String[],   # "solution" or "update"
    sub = Int[],       # subdomain index (0 for the solution column)
    idx = Int[],
    value = Float64[],
  )
  for k in ks
    # Iterates carry an arbitrary eigenvector sign that may flip between
    # sweeps; normalize each snapshot (and the updates relative to the
    # iterate they were computed from) for plotting.
    s_next = sign(sum(sol_hist[k+1]))  # iterate AFTER sweep k (combine step)
    s_cur = sign(sum(sol_hist[k]))     # iterate the local solves saw
    append!(rows.N, fill(N, ndofs))
    append!(rows.iter, fill(k, ndofs))
    append!(rows.kind, fill("solution", ndofs))
    append!(rows.sub, fill(0, ndofs))
    append!(rows.idx, 1:ndofs)
    append!(rows.value, s_next .* sol_hist[k+1])
    for (i, upd) in enumerate(local_update_hist[k])
      append!(rows.N, fill(N, ndofs))
      append!(rows.iter, fill(k, ndofs))
      append!(rows.kind, fill("update", ndofs))
      append!(rows.sub, fill(i, ndofs))
      append!(rows.idx, 1:ndofs)
      append!(rows.value, s_cur .* upd)
    end
  end
  savetable("study7_local3d.csv", rows)
  println("  stored iterations $(ks) of $(niters) (lambda = $lambda)")
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study7()
end
