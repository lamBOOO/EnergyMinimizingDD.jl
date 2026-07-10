# Study 5: p-Laplacian (NonlinearEnergy).
# Convergence of the energy gap for several exponents p, and accuracy of the
# converged solution against a monolithic Gridap Newton reference.

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function run_study5()
  files = ("study5_conv.csv", "study5_summary.csv")
  if !needs_run(files...)
    println("study5: cached, skipping")
    return
  end
  println("study5: p-Laplacian")
  Random.seed!(1)

  N = SMALL ? 12 : 20
  m = SMALL ? 4 : 9
  overlap = 2
  # NOTE: the degenerate regime p < 2 does not converge within the iteration
  # budget (flux singular at grad u = 0); flagged as a limitation in the text.
  ps = SMALL ? [3.0] : [2.5, 3.0, 4.0]
  maxiter = SMALL ? 10 : 80

  rows_c = (p = Float64[], iter = Int[], energy_gap = Float64[])
  rows_s = (
    p = Float64[],
    iters = Int[],
    relerr = Float64[],
    J_dd = Float64[],
    J_ref = Float64[],
  )

  for p in ps
    ea, ga, dofspar, U, ndofs =
      FEMDiscretizations.FEM_PLaplacian(N, m, p, x -> 1.0, overlap)
    e = Energies.NonlinearEnergy("p-Laplacian p=$p", ea, ga, ndofs)

    uh_ref, Ug = FEMDiscretizations.solve_p_laplacian_gridap(N, p)
    u_ref = get_free_dof_values(uh_ref)
    J_ref = ea(u_ref)

    u_dd, J_dd, e_hist, _, resnorm_hist = Solvers.var_dd(
      e,
      dofspar;
      maxiter = maxiter,
      tol = 1e-6,
      verbose = false,
    )

    for (k, ev) in enumerate(e_hist)
      push!(rows_c.p, p)
      push!(rows_c.iter, k - 1)
      push!(rows_c.energy_gap, ev - J_ref)
    end
    push!(rows_s.p, p)
    push!(rows_s.iters, length(resnorm_hist))
    push!(rows_s.relerr, norm(u_dd - u_ref) / norm(u_ref))
    push!(rows_s.J_dd, J_dd)
    push!(rows_s.J_ref, J_ref)
    @printf(
      "  p = %.1f done (%d iters, rel. dof error %.2e)\n",
      p,
      length(resnorm_hist),
      rows_s.relerr[end]
    )
  end
  savetable("study5_conv.csv", rows_c)
  savetable("study5_summary.csv", rows_s)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study5()
end
