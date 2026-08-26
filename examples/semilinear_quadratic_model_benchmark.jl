# Reproducible comparison of two varDD variants for a convex semilinear
# Poisson energy:
#
#   1. the original nonlinear local and second-level minimizations;
#   2. one frozen quadratic Taylor model per outer DD sweep, minimized by the
#      same subdomain and second-level framework using linear solves.
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
const BETA = 1.0

exact_solution(x) = sum(
  coefficient * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in MODES
)

minus_laplacian(x) = pi^2 * sum(
  coefficient * (kx^2 + ky^2) * sinpi(kx * x[1]) * sinpi(ky * x[2]) for
  (coefficient, kx, ky) in MODES
)

forcing(x) = minus_laplacian(x) + BETA * exact_solution(x)^3

function semilinear_problem(N)
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
    potential = s -> BETA * s^4 / 4,
    potential_gradient = s -> BETA * s^3,
    potential_hessian = s -> 3 * BETA * s^2,
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
  result = Solvers.var_dd(
    energy,
    subdomains;
    u0 = initial,
    maxiter = 100,
    tol = 1e-8 * initial_residual,
    quadratic_model = quadratic_model,
    verbose = false,
  )
  return (
    sweeps = length(result[3]) - 1,
    relative_residual = result[5][end] / initial_residual,
    monotone = all(diff(result[3]) .<= 1e-12),
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
  "   N   dofs | nonlinear solves: sweeps  relres   median(s)  monotone | ",
  "quadratic model: sweeps  relres   median(s)  monotone",
)
for N in (8, 16, 32)
  energy, subdomains, initial = semilinear_problem(N)
  nonlinear_time, nonlinear = median_timing(
    () -> run_vardd(energy, subdomains, initial; quadratic_model = false),
  )
  quadratic_time, quadratic = median_timing(
    () -> run_vardd(energy, subdomains, initial; quadratic_model = true),
  )
  @printf(
    "%4d %6d | %6d  %.2e  %9.6f  %8s | %6d  %.2e  %9.6f  %8s\n",
    N,
    length(initial),
    nonlinear.sweeps,
    nonlinear.relative_residual,
    nonlinear_time,
    string(nonlinear.monotone),
    quadratic.sweeps,
    quadratic.relative_residual,
    quadratic_time,
    string(quadratic.monotone),
  )
end
