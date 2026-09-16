using Test
using LinearAlgebra
using SparseArrays
using EnergyMinimizingDD

isdefined(Main, :PAPER_COMMON) || include("../examples/paper/common.jl")
isdefined(Main, :NONLINEAR_SOURCE_METHODS) ||
  include("../examples/paper/nonlinear_source_common.jl")

@testset "study-local nonlinear source baselines" begin
  K = sparse(Symmetric([
    4.0 -1.0  0.0  0.0
   -1.0  4.0 -1.0  0.0
    0.0 -1.0  4.0 -1.0
    0.0  0.0 -1.0  3.0
  ]))
  M = sparse(I, 4, 4)
  b = ones(4)
  energy_function(u) = 0.5 * dot(u, K * u) + 0.1 * sum(exp, -u) - dot(b, u)
  gradient_function(u) = K * u .- 0.1 .* exp.(-u) .- b
  hessian_function(u) = K + spdiagm(0 => 0.1 .* exp.(-u))
  energy = EnergyMinimizingDD.Energies.NonlinearEnergy(
    "tiny exponential problem",
    energy_function,
    gradient_function,
    hessian_function,
    4,
  )
  subdomains = [[1, 2, 3], [2, 3, 4]]
  core = [[1, 2], [3, 4]]
  u0 = zeros(4)
  initial = norm(gradient_function(u0))

  anderson = nonlinear_source_anderson_ras(
    energy,
    subdomains,
    core;
    u0,
    maxiter=8,
    tolerance=1e-9,
    history_depth=2,
  )
  @test anderson.residual_history[end] < 1e-4 * initial
  @test anderson.nonlinear_local_batches == length(anderson.energy_history) - 1

  nonlinear_cg = nonlinear_source_optim_ncg_as(
    energy,
    subdomains;
    u0,
    maxiter=100,
    tolerance=1e-9,
  )
  @test nonlinear_cg.residual_history[end] < 1e-8 * initial
  @test nonlinear_cg.linear_as_batches > 0
  @test nonlinear_cg.global_jacobian_products > 0

  newton = nonlinear_source_newton_pcg_as(
    energy,
    subdomains;
    u0,
    maxiter=8,
    tolerance=1e-9,
    inner_iterations=4,
  )
  @test newton.residual_history[end] < 1e-8 * initial
  @test newton.linear_as_batches > 0
  @test all(diff(newton.energy_history) .<= 1e-12)

  imex = nonlinear_source_energy_imex_pcg_as(
    energy,
    K,
    M,
    subdomains;
    u0,
    maxiter=15,
    tolerance=1e-8,
    timestep=1.0,
    inner_maxiter=20,
    inner_relative_tolerance=1e-12,
  )
  @test imex.residual_history[end] < 1e-6 * initial
  @test imex.linear_as_batches > 0

  aspin = nonlinear_source_aspin(
    energy,
    subdomains;
    u0,
    maxiter=8,
    tolerance=1e-9,
    inner_maxiter=8,
    inner_relative_tolerance=1e-9,
  )
  @test aspin.residual_history[end] < 1e-6 * initial
  @test aspin.nonlinear_local_batches > 0
  @test aspin.linear_as_batches > 0

  raspen = nonlinear_source_raspen(
    energy,
    subdomains,
    core;
    u0,
    maxiter=6,
    tolerance=1e-9,
    inner_maxiter=8,
    inner_relative_tolerance=1e-9,
  )
  @test raspen.residual_history[end] < 1e-7 * initial
  @test raspen.nonlinear_local_batches > 0
  @test raspen.linear_as_batches > 0
end
