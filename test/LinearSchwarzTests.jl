using Test
using LinearAlgebra

if !isdefined(Main, :PAPER_COMMON)
  include(joinpath(@__DIR__, "..", "examples", "paper", "common.jl"))
end
include(joinpath(@__DIR__, "..", "examples", "paper", "study8_linear_cmp.jl"))

@testset "linear Schwarz baselines" begin
  K_manufactured, _, b_manufactured, _, U_manufactured =
    study8_sign_changing_problem_setup(16, 1, 1; partitioning=:cartesian)
  discrete_solution = K_manufactured \ b_manufactured
  exact_values = collect(get_free_dof_values(interpolate_everywhere(
    study8_sign_changing_exact_solution, U_manufactured
  )))
  @test norm(discrete_solution - exact_values) / norm(exact_values) < 0.02

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

  x_sol = K \ b
  error_history = gmres_ras(
    K, b, schwarz; maxiter=40, tol=1e-11, x_sol
  )
  @test length(first(error_history)) == 3
  @test first(error_history)[3] ≈ norm(ones(length(b)) - x_sol) / norm(x_sol)
  @test last(error_history)[2] < 1e-10
  @test last(error_history)[3] < 1e-10

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

  reference_owners = metis_cell_owners(4, 2)
  nested_owners = prolong_cell_owners(reference_owners, 8)
  @test length(nested_owners) == 8^2
  for parent_y = 1:4, parent_x = 1:4
    parent = parent_x + 4 * (parent_y - 1)
    children = [
      child_x + 8 * (child_y - 1) for
      child_y = (2 * parent_y-1):(2 * parent_y),
      child_x = (2 * parent_x-1):(2 * parent_x)
    ]
    @test all(nested_owners[children] .== reference_owners[parent])
  end
  _, _, _, nested_overlap, _, nested_core = laplace_setup(
    8,
    2,
    1;
    cell_owners=nested_owners,
    return_core_partition=true,
  )
  @test all(i -> nested_core[i] ⊆ nested_overlap[i], 1:2)
  @test_throws ArgumentError prolong_cell_owners(reference_owners, 6)
  @test_throws DimensionMismatch laplace_setup(
    8, 2, 1; cell_owners=nested_owners[1:end-1]
  )
  @test_throws ArgumentError laplace_setup(
    8, 2, 1; cell_owners=fill(Int32(1), 8^2)
  )

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
