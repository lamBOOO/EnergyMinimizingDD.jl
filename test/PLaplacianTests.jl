using Test
using LinearAlgebra
using VariationalDD

const PLFEM = VariationalDD.FEMDiscretizations
const PLEnergies = VariationalDD.Energies
const PLSolvers = VariationalDD.Solvers

@testset "triangular p-Laplacian energy and one-level nonlinear DD" begin
  function setup(p)
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    subdomains,
    _,
    ndofs,
    stiffness,
    initial = PLFEM.FEM_PLaplacian(6, 4, p, x -> 1.0, 1)
    energy = PLEnergies.NonlinearEnergy(
      "p-Laplacian p=$p",
      energy_assembler,
      gradient_assembler,
      hessian_assembler,
      ndofs,
    )
    return energy, subdomains, stiffness, initial
  end

  @testset "analytic derivatives" begin
    energy, _, _, initial = setup(3.0)
    direction = collect(range(-0.7, 0.9; length=length(initial)))
    direction ./= norm(direction)
    step = 1e-6
    finite_difference = (
      PLEnergies.energy(energy, initial + step * direction) -
      PLEnergies.energy(energy, initial - step * direction)
    ) / (2step)
    @test finite_difference ≈ dot(PLEnergies.gradient(energy, initial), direction) rtol=2e-6

    hessian_action = PLEnergies.hessian(energy, initial) * direction
    gradient_difference = (
      PLEnergies.gradient(energy, initial + step * direction) -
      PLEnergies.gradient(energy, initial - step * direction)
    ) / (2step)
    @test norm(hessian_action - gradient_difference) / norm(hessian_action) < 2e-5
    @test norm(PLEnergies.hessian(energy, initial) - PLEnergies.hessian(energy, initial)') < 1e-11
  end

  @testset "p=2 recovers Poisson" begin
    energy, _, stiffness, initial = setup(2.0)
    @test norm(PLEnergies.hessian(energy, initial) - stiffness) / norm(stiffness) < 1e-12
  end

  @testset "METIS partitions of the triangle graph" begin
    for m in (2, 4, 8)
      _, _, _, dofs, _, ndofs, _, _, core_dofs =
        PLFEM.FEM_PLaplacian(8, m, 2.0)
      restricted_dofs =
        PLFEM.create_balanced_disjoint_dofs_partition(core_dofs, ndofs)
      @test length(dofs) == m
      @test all(!isempty, dofs)
      @test length(restricted_dofs) == m
      @test sort(vcat(restricted_dofs...)) == collect(1:ndofs)
      @test all(i -> restricted_dofs[i] ⊆ dofs[i], eachindex(dofs))
      restricted_sizes = length.(restricted_dofs)
      @test maximum(restricted_sizes) - minimum(restricted_sizes) <= 2
    end
  end

  @testset "shared infimum and linear combination interface" begin
    energy, subdomains, _, initial = setup(3.0)
    active = collect(Int, subdomains[1])
    local_minimizer = PLSolvers.inf_step(energy, initial, active)
    inactive = setdiff(eachindex(initial), active)
    @test local_minimizer[inactive] == initial[inactive]
    @test PLEnergies.energy(energy, local_minimizer) <=
          PLEnergies.energy(energy, initial)

    combined = PLSolvers.combine_step(
      energy, hcat(initial, local_minimizer)
    )
    @test PLEnergies.energy(energy, combined) <=
          PLEnergies.energy(energy, initial)
  end

  @testset "common var_dd outer iteration" begin
    energy, subdomains, _, initial = setup(3.0)
    plain = PLSolvers.var_dd(
      energy,
      subdomains;
      u0=initial,
      maxiter=4,
      tol=1e-14,
      verbose=false,
    )
    history = PLSolvers.var_dd(
      energy,
      subdomains;
      u0=initial,
      maxiter=4,
      tol=1e-14,
      history_depth=1,
      verbose=false,
    )
    @test all(diff(plain[3]) .<= 1e-11)
    @test all(diff(history[3]) .<= 1e-11)
    @test plain[3][2] ≈ history[3][2] atol=1e-12
    @test plain[5][end] < PLEnergies.residual_norm(energy, initial)
    @test history[5][end] < PLEnergies.residual_norm(energy, initial)
  end

  @testset "derivative convergence below the energy roundoff floor" begin
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    subdomains,
    _,
    ndofs,
    _,
    initial = PLFEM.FEM_PLaplacian(8, 4, 2.0, x -> 1.0, 2)
    energy = PLEnergies.NonlinearEnergy(
      "Poisson roundoff regression",
      energy_assembler,
      gradient_assembler,
      hessian_assembler,
      ndofs,
    )
    initial_residual = PLEnergies.residual_norm(energy, initial)
    result = PLSolvers.var_dd(
      energy,
      subdomains;
      u0=initial,
      maxiter=60,
      tol=1e-9 * initial_residual,
      verbose=false,
    )
    relative_residual = result[5][end] / initial_residual
    @test relative_residual < 1e-9
    @test all(diff(result[3]) .<= 100eps(Float64))
  end
end
