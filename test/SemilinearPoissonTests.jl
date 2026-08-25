using Test
using LinearAlgebra
using VariationalDD

const SPFEM = VariationalDD.FEMDiscretizations
const SPEnergies = VariationalDD.Energies
const SPSolvers = VariationalDD.Solvers

@testset "generic semilinear Poisson FEM" begin
  function semilinear_energy(; N=6, m=4, potential, dpotential, ddpotential,
                             forcing=x -> 1.0)
    ea, ga, ha, subdomains, U, ndofs, K, initial, core =
      SPFEM.FEM_SemilinearPoisson(
        N,
        m;
        potential=potential,
        potential_gradient=dpotential,
        potential_hessian=ddpotential,
        forcing=forcing,
        overlap=1,
        initial_guess=x -> 0.1 * x[1] * (1 - x[1]) * x[2] * (1 - x[2]),
      )
    energy = SPEnergies.NonlinearEnergy(
      "semilinear test", ea, ga, ha, ndofs
    )
    return energy, subdomains, core, U, K, initial
  end

  @testset "analytic exponential derivatives and convexity" begin
    energy, subdomains, core, _, _, initial = semilinear_energy(
      potential=s -> exp(-s),
      dpotential=s -> -exp(-s),
      ddpotential=s -> exp(-s),
    )
    direction = collect(range(-0.8, 0.7; length=length(initial)))
    direction ./= norm(direction)
    step = 1e-6
    energy_difference = (
      SPEnergies.energy(energy, initial + step * direction) -
      SPEnergies.energy(energy, initial - step * direction)
    ) / (2step)
    @test energy_difference ≈
      dot(SPEnergies.gradient(energy, initial), direction) rtol=2e-6

    hessian_action = SPEnergies.hessian(energy, initial) * direction
    gradient_difference = (
      SPEnergies.gradient(energy, initial + step * direction) -
      SPEnergies.gradient(energy, initial - step * direction)
    ) / (2step)
    @test norm(hessian_action - gradient_difference) / norm(hessian_action) < 2e-6
    @test minimum(eigvals(Symmetric(Matrix(SPEnergies.hessian(energy, initial))))) > 0

    restricted = SPFEM.create_balanced_disjoint_dofs_partition(
      core, length(initial)
    )
    @test sort(vcat(restricted...)) == collect(eachindex(initial))
    @test all(i -> restricted[i] ⊆ subdomains[i], eachindex(subdomains))
  end

  @testset "quadratic potential recovers a linear reaction" begin
    coefficient = 3.0
    energy, _, _, _, _, initial = semilinear_energy(
      potential=s -> 0.5 * coefficient * s^2,
      dpotential=s -> coefficient * s,
      ddpotential=s -> coefficient,
    )
    perturbation = collect(range(-0.2, 0.3; length=length(initial)))
    H0 = SPEnergies.hessian(energy, initial)
    H1 = SPEnergies.hessian(energy, initial + perturbation)
    @test norm(H1 - H0) / norm(H0) < 1e-13
    @test SPEnergies.gradient(energy, initial + perturbation) -
          SPEnergies.gradient(energy, initial) ≈ H0 * perturbation rtol=1e-12
  end

  @testset "optional consistent mass matrix" begin
    result = SPFEM.FEM_SemilinearPoisson(
      4,
      2;
      potential=s -> exp(-s),
      potential_gradient=s -> -exp(-s),
      potential_hessian=s -> exp(-s),
      overlap=1,
      return_mass_matrix=true,
    )
    @test length(result) == 10
    mass = result[end]
    @test issymmetric(mass)
    @test minimum(eigvals(Symmetric(Matrix(mass)))) > 0
  end

  @testset "L-BFGS EMDD local solve uses the linear enriched space" begin
    A = [
      5.0 -1.0  0.0  0.0  0.0
     -1.0  4.0 -1.0  0.0  0.0
      0.0 -1.0  4.0 -1.0  0.0
      0.0  0.0 -1.0  4.0 -1.0
      0.0  0.0  0.0 -1.0  3.0
    ]
    b = [1.0, -0.5, 2.0, 0.25, 1.5]
    hessian_called = Ref(false)
    energy = SPEnergies.NonlinearEnergy(
      "quadratic nonlinear-energy wrapper",
      u -> 0.5 * dot(u, A * u) - dot(b, u),
      u -> A * u - b,
      _ -> begin
        hessian_called[] = true
        error("the enriched EMDD local L-BFGS solve must not use the Hessian")
      end,
      5,
    )
    current = [0.8, -0.4, 0.6, 1.1, -0.7]
    active = Int32[2, 3]
    basis = hcat(
      current,
      [index == active[1] ? 1.0 : 0.0 for index = 1:5],
      [index == active[2] ? 1.0 : 0.0 for index = 1:5],
    )
    expected = basis * ((basis' * A * basis) \ (basis' * b))
    enriched = SPSolvers.inf_step(energy, current, active)
    @test !hessian_called[]
    @test enriched ≈ expected atol=1e-8 rtol=1e-8

    inactive = setdiff(eachindex(current), active)
    scale = dot(current[inactive], enriched[inactive]) /
            dot(current[inactive], current[inactive])
    @test enriched[inactive] ≈ scale .* current[inactive]
  end

  @testset "shared nonlinear DD interface" begin
    exact(x) = sinpi(x[1]) * sinpi(x[2])
    forcing(x) = 2pi^2 * exact(x) - exp(-exact(x))
    energy, subdomains, _, _, _, initial = semilinear_energy(
      potential=s -> exp(-s),
      dpotential=s -> -exp(-s),
      ddpotential=s -> exp(-s),
      forcing=forcing,
    )
    initial_residual = SPEnergies.residual_norm(energy, initial)
    result = SPSolvers.var_dd(
      energy,
      subdomains;
      u0=initial,
      maxiter=4,
      tol=1e-12,
      history_depth=1,
      verbose=false,
    )
    @test all(diff(result[3]) .<= 1e-11)
    @test result[5][end] < initial_residual
  end
end
