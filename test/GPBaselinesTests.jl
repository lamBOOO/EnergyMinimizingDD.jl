using Test
using LinearAlgebra
using SparseArrays
using EnergyMinimizingDD

isdefined(Main, :gp_gfdn_pcg_as_history) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "study10_gp.jl"))

@testset "paper Gross--Pitaevskii baselines" begin
  n = 8
  K = spdiagm(0 => collect(range(1.0, 2.0; length=n)))
  M = spdiagm(0 => ones(n))
  beta = 5.0
  quartic(u) = sum(abs2.(u) .^ 2)
  cubic(u) = u .^ 3
  density_matrix(u) = spdiagm(0 => u .^ 2)
  energy = Energies.GrossPitaevskiiRayleighQuotient(
    K, M, beta, quartic, cubic; density_matrix,
  )
  subdomains = [Int32.(1:5), Int32.(4:8)]
  initial = collect(range(1.0, 2.0; length=n))
  Energies.normalize_M!(initial, M)

  for conjugate in (false, true)
    solution, history = gp_gfdn_pcg_as_history(
      energy, density_matrix, subdomains, initial;
      conjugate,
      inner_iterations=4,
      maxiter=10,
      tol=1e-10,
    )
    energies = getindex.(history, 2)
    @test all(diff(energies) .<= 1e-10)
    @test last(history)[3] < first(history)[3]
    @test abs(dot(solution, M * solution) - 1) < 1e-12
  end
end
