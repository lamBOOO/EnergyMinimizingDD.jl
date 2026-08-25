# Study 1: problem setup illustration data.
# - METIS overlapping partition (owner + overlap multiplicity per dof)
# - Schroedinger ground state computed with var_dd

isdefined(Main, :PAPER_COMMON) || include("common.jl")

function run_study1()
  files = ("study1_partition.csv", "study1_schroedinger.csv")
  if !needs_run(files...)
    println("study1: cached, skipping")
    return
  end
  println("study1: setup illustrations")
  Random.seed!(1)

  N = SMALL ? 16 : 32
  m = 4
  overlap = 2

  # Partition illustration: owner = smallest subdomain index containing the
  # dof, multiplicity = number of subdomains containing it (overlap regions).
  K, M, b, dofspar, U = schroedinger_setup(N, m, overlap)
  ndofs = size(K, 1)
  owner = zeros(Int, ndofs)
  mult = zeros(Int, ndofs)
  for (i, dofs) in enumerate(dofspar)
    for j in dofs
      owner[j] == 0 && (owner[j] = i)
      mult[j] += 1
    end
  end
  savetable(
    "study1_partition.csv",
    (N = fill(N, ndofs), idx = 1:ndofs, owner = owner, mult = mult),
  )

  # Schroedinger ground state (default exponential potential)
  e_evp = Energies.GeneralizedRayleighQuotient(K, M)
  u_gs, lambda_gs, = Solvers.var_dd(
    e_evp,
    dofspar;
    maxiter = 100,
    tol = 1e-8,
    verbose = false,
  )
  u_gs .*= sign(sum(u_gs))  # fix sign for plotting
  savetable(
    "study1_schroedinger.csv",
    (
      N = fill(N, ndofs),
      idx = 1:ndofs,
      value = u_gs,
      lambda = fill(lambda_gs, ndofs),
    ),
  )
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study1()
end
