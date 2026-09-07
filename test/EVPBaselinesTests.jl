using Test
using LinearAlgebra
using SparseArrays
using Random
using EnergyMinimizingDD

isdefined(Main, :PAPER_COMMON) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "common.jl"))
isdefined(Main, :EVP_COMMON) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "evp_common.jl"))

@testset "study-local generalized eigenproblem baselines" begin
  Random.seed!(71)
  n = 30
  K = spdiagm(
    -1 => fill(-1.0, n - 1),
    0 => collect(range(3.0, 4.0; length=n)),
    1 => fill(-1.0, n - 1),
  )
  M = spdiagm(0 => collect(range(0.8, 1.2; length=n)))
  subdomains = [
    Int32.(1:12),
    Int32.(9:22),
    Int32.(19:30),
  ]
  schwarz = schwarz_setup(K, subdomains)
  reference = minimum(eigvals(Symmetric(Matrix(K)), Symmetric(Matrix(M))))

  current = normalize_M!(collect(range(0.5, 1.5; length=n)), M)
  indices = subdomains[2]
  local_result = EnergyMinimizingDD.Solvers.generalized_rayleigh_inf_step(
    EnergyMinimizingDD.Energies.GeneralizedRayleighQuotient(K, M),
    current,
    indices;
    collect_info=true,
  )
  complement = copy(current)
  complement[indices] .= 0
  basis = zeros(n, 1 + length(indices))
  basis[:, 1] .= complement
  for (column, degree) in pairs(indices)
    basis[degree, column+1] = 1
  end
  dense_values = eigvals(
    Symmetric(basis' * K * basis),
    Symmetric(basis' * M * basis),
  )
  @test rayleigh(K, M, local_result.u) ≈ first(dense_values) rtol=1e-7
  @test local_result.info.k_nnz < local_result.info.dimension^2

  pcg = evp_pcg_as(
    K,
    ones(n),
    schwarz;
    relative_tolerance=1e-10,
    maxiter=100,
  )
  @test pcg.converged
  @test norm(K * pcg.x - ones(n)) / sqrt(n) < 1e-9
  @test pcg.as_batches == pcg.iterations

  results = Dict(
    :lopsd => evp_lopsd_as(
      K, M, schwarz; maxiter=50, relative_tolerance=1e-6
    ),
    :lobpcg => evp_lobpcg_as(
      K, M, schwarz; maxiter=50, relative_tolerance=1e-6
    ),
    :jd => evp_jd_gmres_as(
      K,
      M,
      schwarz;
      maxiter=20,
      relative_tolerance=1e-6,
      inner_maxiter=12,
    ),
    :si_lanczos => evp_si_lanczos_pcg_as(
      K,
      M,
      schwarz;
      maxiter=50,
      relative_tolerance=1e-6,
      inner_relative_tolerance=0.0,
      inner_maxiter=8,
      restart_dimension=20,
    ),
  )
  for (method, result) in results
    residual_limit = method == :lopsd ? 2e-3 : 2e-5
    @test last(result.history).relative_residual < residual_limit
    @test abs(last(result.history).lambda - reference) < 2e-5
    @test all(isfinite, result.u)
    method in (:jd, :si_lanczos) &&
      @test last(result.history).linear_as_batches > length(result.history) - 1
  end
  @test all(stat -> stat.iterations == 8, results[:si_lanczos].inner_stats)

  for (method, iterations) in EVP_LANCZOS_PCG_ITERATIONS
    fixed_work = evp_method_result(
      method,
      K,
      M,
      subdomains,
      schwarz;
      maxiter=1,
      relative_tolerance=1e-6,
    )
    @test only(fixed_work.inner_stats).iterations == iterations
    @test !only(fixed_work.inner_stats).converged
  end

  local_info = NamedTuple[]
  vardd = evp_vardd(
    K,
    M,
    subdomains;
    maxiter=30,
    relative_tolerance=1e-6,
    history_depth=1,
  )
  append!(local_info, vardd.local_stats)
  @test last(vardd.history).relative_residual < 2e-5
  @test !isempty(local_info)
  @test all(stat -> stat.dimension == length(subdomains[stat.subdomain]) + 1, local_info)
  @test all(stat -> stat.k_nnz < stat.dimension^2, local_info)
  @test last(vardd.history).local_iterations_critical > 0
  @test last(vardd.history).local_iterations_total >=
        last(vardd.history).local_iterations_critical
  @test all(
    stat -> stat.effective_rank <= stat.basis_columns,
    vardd.combination_stats,
  )
end
