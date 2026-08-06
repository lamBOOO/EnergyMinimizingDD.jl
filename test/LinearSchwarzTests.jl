using Test
using LinearAlgebra

if !isdefined(Main, :PAPER_COMMON)
  include(joinpath(@__DIR__, "..", "examples", "paper", "common.jl"))
end
include(joinpath(@__DIR__, "..", "examples", "paper", "study8_linear_cmp.jl"))

@testset "linear Schwarz baselines" begin
  K, _, b, overlapping, _, core = laplace_setup(
    8, 3, 1; return_core_partition=true
  )
  schwarz = schwarz_setup(K, overlapping; core_dofs=core)

  @test length(schwarz.dofs) == 3
  owned_dofs = [
    schwarz.dofs[i][schwarz.owner_mask[i]] for i in eachindex(schwarz.dofs)
  ]
  @test all(i -> all(in(schwarz.dofs[i]), owned_dofs[i]), eachindex(owned_dofs))
  owned = vcat(owned_dofs...)
  @test sort(owned) == collect(1:size(K, 1))
  @test maximum(abs.(length.(owned_dofs) .- length(owned) / 3)) <= 4

  history = gmres_ras(K, b, schwarz; maxiter=40, tol=1e-11)
  @test first(history) == (0, norm(b - K * ones(length(b))))
  @test last(history)[2] < 1e-10
  @test all(diff(last.(history)) .<= 1e-12)
  @test all(first.(history) .== (0:(length(history) - 1)) .* nsub(schwarz))

  high_contrast, _, _, _, _ = laplace_setup(
    8, 3, 1;
    diffusion=x -> x.data[1] < 0.5 ? 100.0 : 1.0,
  )
  @test norm(high_contrast - K) > norm(K)
  @test isposdef(Symmetric(Matrix(high_contrast)))

  _, _, _, cartesian_overlap, _, cartesian_core = laplace_setup(
    8,
    4,
    1;
    partitioning=:cartesian,
    return_core_partition=true,
  )
  @test length(cartesian_core) == 4
  @test all(i -> cartesian_core[i] ⊆ cartesian_overlap[i], 1:4)
  @test_throws ArgumentError laplace_setup(8, 3, 1; partitioning=:cartesian)

  dimensions = Int[]
  Solvers.var_dd(
    Energies.QuadraticEnergy(K, b),
    overlapping;
    maxiter=2,
    tol=-1.0,
    history_depth=1,
    subspace_callback=(_, candidates) -> push!(dimensions, size(candidates, 2)),
    verbose=false,
  )
  @test dimensions == [4, 5]
end
