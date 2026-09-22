using Test
using LinearAlgebra
using SparseArrays
using EnergyMinimizingDD

isdefined(Main, :PAPER_COMMON) || include("../examples/paper/common.jl")
isdefined(Main, :SEMILINEAR_SOURCE_METHODS) ||
  include("../examples/paper/nonlinear_source_common.jl")

@testset "paper semilinear baselines" begin
  K = sparse(Symmetric([
    4.0 -1.0  0.0  0.0
   -1.0  4.0 -1.0  0.0
    0.0 -1.0  4.0 -1.0
    0.0  0.0 -1.0  3.0
  ]))
  b = ones(4)
  objective(u) = 0.5 * dot(u, K * u) + 0.1 * sum(exp, -u) - dot(b, u)
  gradient(u) = K * u .- 0.1 .* exp.(-u) .- b
  hessian(u) = K + spdiagm(0 => 0.1 .* exp.(-u))
  energy = EnergyMinimizingDD.Energies.NonlinearEnergy(
    "tiny exponential problem", objective, gradient, hessian, 4
  )
  subdomains = [[1, 2, 3], [2, 3, 4]]
  core = [[1, 2], [3, 4]]
  initial = zeros(4)
  initial_residual = norm(gradient(initial))

  anderson = nonlinear_source_anderson_ras(
    energy, subdomains, core;
    u0=initial, maxiter=8, tolerance=1e-9, history_depth=2,
  )
  @test anderson.residual_history[end] < 1e-4 * initial_residual

  nonlinear_cg = nonlinear_source_optim_ncg_as(
    energy, subdomains; u0=initial, maxiter=100, tolerance=1e-9
  )
  @test nonlinear_cg.residual_history[end] < 1e-8 * initial_residual
  @test nonlinear_cg.linear_as_batches > 0

  newton = nonlinear_source_newton_pcg_as(
    energy, subdomains;
    u0=initial, maxiter=8, tolerance=1e-9, inner_iterations=4,
  )
  @test newton.residual_history[end] < 1e-8 * initial_residual
  @test all(diff(newton.energy_history) .<= 1e-12)

  emdd = nonlinear_source_vardd(
    energy, subdomains;
    u0=initial, maxiter=20, tolerance=1e-8, history_depth=1,
  )
  @test emdd.residual_history[end] < 1e-6 * initial_residual
end
