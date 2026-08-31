# Reproducible comparison of two varDD variants for a convex semilinear
# Poisson energy:
#
#   1. the original nonlinear local minimizations;
#   2. one frozen quadratic Taylor model per outer DD sweep for linear local
#      solves.
#
# Both variants use an exact nonlinear second-level minimization of the
# original energy. The benchmark reports the total L-BFGS iterations spent in
# the local subproblems of the full nonlinear variant. Every case starts from
# the zero initial guess.
#
# Run from the repository root with
#
#   julia --project=. examples/semilinear_quadratic_model_benchmark.jl

using VariationalDD
using LinearAlgebra
using Printf
using Statistics

const FEM = VariationalDD.FEMDiscretizations
const Energies = VariationalDD.Energies
const Solvers = VariationalDD.Solvers

const MODES = (
  (1.50, 1, 1),
  (0.55, 2, 3),
  (0.35, 3, 2),
)
const BETAS = (1.0, 10.0, 100.0, 1000.0)

exact_solution(x) = sum(
  coefficient * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in MODES
)

minus_laplacian(x) = pi^2 * sum(
  coefficient * (kx^2 + ky^2) * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in MODES
)

function semilinear_problem(N, beta)
  forcing(x) = minus_laplacian(x) + beta * exact_solution(x)^3
  energy_assembler,
  gradient_assembler,
  hessian_assembler,
  subdomains,
  _,
  ndofs,
  _,
  initial,
  _ = FEM.FEM_SemilinearPoisson(
    N,
    4;
    potential = s -> beta * s^4 / 4,
    potential_gradient = s -> beta * s^3,
    potential_hessian = s -> 3 * beta * s^2,
    forcing = forcing,
    overlap = 2,
    quadrature_degree = 8,
    initial_guess = x -> 0.0,
  )
  energy = Energies.NonlinearEnergy(
    "semilinear benchmark",
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    ndofs,
  )
  return energy, subdomains, initial
end

function run_vardd(energy, subdomains, initial; quadratic_model)
  initial_residual = Energies.residual_norm(energy, initial)
  local_inner_iterations = Ref(0)
  local_solve_count = Ref(0)
  max_local_inner_iterations = Ref(0)
  function record_local_iterations(_, _, info)
    iterations = max(0, info.iterations)
    local_inner_iterations[] += iterations
    local_solve_count[] += 1
    max_local_inner_iterations[] = max(
      max_local_inner_iterations[], iterations
    )
    return nothing
  end
  result = Solvers.var_dd(
    energy,
    subdomains;
    u0 = initial,
    maxiter = 100,
    tol = 1e-8 * initial_residual,
    quadratic_model = quadratic_model,
    local_solve_callback = record_local_iterations,
    verbose = false,
  )
  average_local_inner_iterations = iszero(local_solve_count[]) ?
                                   0.0 :
                                   local_inner_iterations[] / local_solve_count[]
  return (
    sweeps = length(result[3]) - 1,
    relative_residual = result[5][end] / initial_residual,
    monotone = all(diff(result[3]) .<= 1e-12),
    local_inner_iterations = local_inner_iterations[],
    average_local_inner_iterations = average_local_inner_iterations,
    max_local_inner_iterations = max_local_inner_iterations[],
  )
end

function median_timing(run; repeats = 3)
  result = run() # warm up compilation
  times = Float64[]
  for _ = 1:repeats
    GC.gc()
    elapsed = @elapsed result = run()
    push!(times, elapsed)
  end
  return median(times), result
end

println("Semilinear varDD benchmark (m=4, relative tolerance 1e-8)")
println(
  " beta    N   dofs | nonlinear local solves: sweeps  total    avg  max  relres   median(s)  monotone | ",
  "quadratic model: sweeps  relres   median(s)  monotone",
)
for beta in BETAS, N in (8, 16, 32)
  energy, subdomains, initial = semilinear_problem(N, beta)
  nonlinear_time, nonlinear = median_timing(
    () -> run_vardd(energy, subdomains, initial; quadratic_model = false),
  )
  quadratic_time, quadratic = median_timing(
    () -> run_vardd(energy, subdomains, initial; quadratic_model = true),
  )
  @printf(
    "%5g %4d %6d | %6d %6d %6.2f %4d  %.2e  %9.6f  %8s | %6d  %.2e  %9.6f  %8s\n",
    beta,
    N,
    length(initial),
    nonlinear.sweeps,
    nonlinear.local_inner_iterations,
    nonlinear.average_local_inner_iterations,
    nonlinear.max_local_inner_iterations,
    nonlinear.relative_residual,
    nonlinear_time,
    string(nonlinear.monotone),
    quadratic.sweeps,
    quadratic.relative_residual,
    quadratic_time,
    string(quadratic.monotone),
  )
end
