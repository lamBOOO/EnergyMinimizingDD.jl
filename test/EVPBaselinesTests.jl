using Test
using LinearAlgebra
using SparseArrays
using EnergyMinimizingDD

isdefined(Main, :PAPER_COMMON) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "common.jl"))
isdefined(Main, :EVP_COMMON) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "evp_common.jl"))

@testset "paper generalized-eigenproblem baselines" begin
  n = 30
  K = spdiagm(-1 => fill(-1.0, n - 1),
              0 => collect(range(3.0, 4.0; length=n)),
              1 => fill(-1.0, n - 1))
  M = spdiagm(0 => collect(range(0.8, 1.2; length=n)))
  subdomains = [Int32.(1:12), Int32.(9:22), Int32.(19:30)]
  schwarz = schwarz_setup(K, subdomains)
  reference = minimum(eigvals(Symmetric(Matrix(K)), Symmetric(Matrix(M))))

  results = (
    evp_lopsd_as(K, M, schwarz; maxiter=80, relative_tolerance=1e-6),
    evp_lobpcg_as(K, M, schwarz; maxiter=80, relative_tolerance=1e-6),
    evp_jd_gmres_as(K, M, schwarz;
      maxiter=30, relative_tolerance=1e-6, inner_maxiter=4),
    evp_vardd(K, M, subdomains;
      maxiter=80, relative_tolerance=1e-6, history_depth=1),
  )
  for result in results
    @test abs(last(result.history).lambda - reference) < 1e-4
    @test last(result.history).relative_residual < 1e-4
  end
end
