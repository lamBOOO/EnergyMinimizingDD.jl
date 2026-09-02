using Test
using LinearAlgebra
using EnergyMinimizingDD.Energies
using EnergyMinimizingDD.FEMDiscretizations
using EnergyMinimizingDD.Solvers

include(joinpath(@__DIR__, "..", "examples", "paper", "common.jl"))
include(joinpath(@__DIR__, "..", "examples", "paper", "study10_gp.jl"))

@testset "Gross-Pitaevskii quotient" begin
  @testset "algebraic energy, gradient, and scale invariance" begin
    K = Symmetric([3.0 -0.4 0.0; -0.4 2.0 -0.2; 0.0 -0.2 4.0])
    M = Diagonal([1.0, 1.5, 0.8])
    weights = [0.7, 1.1, 0.9]
    quartic(u) = sum(weights .* u .^ 4)
    cubic(u) = weights .* u .^ 3
    e = Energies.GrossPitaevskiiRayleighQuotient(K, M, 2.0, quartic, cubic)
    u = [0.8, -0.3, 1.2]

    @test e(3.7 .* u) ≈ e(u) rtol = 1e-13
    @test Energies.physical_energy(e, u) ≈ e(u) / 2
    @test Energies.chemical_potential(e, 2.1 .* u) ≈
      Energies.chemical_potential(e, u)

    h = 1e-6
    gradient_fd = [
      (e(u .+ h .* (axes(u, 1) .== i)) - e(u .- h .* (axes(u, 1) .== i))) /
      (2h) for i in eachindex(u)
    ]
    @test Energies.gradient(e, u) ≈ gradient_fd rtol = 2e-6 atol = 1e-8
    @test abs(dot(u, Energies.gradient(e, u))) ≤ 1e-11
    @test_throws ArgumentError Energies.GrossPitaevskiiRayleighQuotient(
      K, M, -1.0, quartic, cubic
    )
  end

  @testset "beta zero is the generalized linear EVP" begin
    K = Diagonal([1.0, 2.0, 4.0, 7.0])
    M = Diagonal([1.0, 1.5, 0.8, 2.0])
    quartic(u) = sum(u .^ 4)
    cubic(u) = u .^ 3
    gp = Energies.GrossPitaevskiiRayleighQuotient(K, M, 0.0, quartic, cubic)
    linear = Energies.GeneralizedRayleighQuotient(K, M)
    u = [1.0, 0.5, -0.2, 0.8]
    idx = Int32[2, 3]

    @test gp(u) ≈ linear(u)
    local_gp = Solvers.inf_step(gp, u, idx)
    local_linear = Solvers.inf_step(linear, u, idx)
    @test min(norm(local_gp - local_linear), norm(local_gp + local_linear)) ≤
      1e-12
    candidates = hcat(u, [0.3, 1.0, 0.2, -0.1], [1.0, -0.2, 0.4, 0.7])
    x_gp = Solvers.combine_step(gp, candidates)
    x_linear = Solvers.combine_step(linear, candidates)
    @test abs(dot(x_gp, M * x_linear)) ≈ 1.0 atol = 1e-12
    @test gp(x_gp) ≈ linear(x_linear) atol = 1e-12
  end

  @testset "reduced minimization and varDD convergence" begin
    n = 8
    K = Symmetric(
      Matrix(Tridiagonal(fill(-0.4, n - 1), fill(2.5, n), fill(-0.4, n - 1)))
    )
    M = Diagonal(collect(range(0.8, 1.2; length=n)))
    weights = collect(range(0.5, 1.0; length=n))
    quartic(u) = sum(weights .* u .^ 4)
    cubic(u) = weights .* u .^ 3
    e = Energies.GrossPitaevskiiRayleighQuotient(K, M, 3.0, quartic, cubic)
    u0 = ones(n)
    reference = Solvers.combine_step(
      e,
      hcat(u0, Matrix{Float64}(I, n, n));
      initial=u0,
      maxiter=1000,
      tol=1e-11,
    )
    @test abs(dot(reference, M * reference) - 1) ≤ 1e-12
    @test Energies.residual_norm(e, reference) ≤ 1e-7
    scaled_reference = Solvers.combine_step(
      e,
      hcat(u0, Matrix{Float64}(I, n, n));
      initial=7 .* u0,
      maxiter=1000,
      tol=1e-11,
    )
    @test abs(dot(reference, M * scaled_reference)) ≈ 1.0 atol = 1e-9
    @test Energies.energy(e, scaled_reference) ≈ Energies.energy(e, reference) atol =
      1e-10
    @test_throws ArgumentError Solvers.combine_step(e, zeros(n, 2))

    parts = [Int32[1, 2, 3, 4, 5], Int32[4, 5, 6, 7, 8]]
    solution, _, energies, solutions, residuals = Solvers.var_dd(
      e, parts; u0=u0, maxiter=40, tol=1e-7, history_depth=1, verbose=false
    )
    @test all(diff(energies) .≤ 1e-10)
    @test all(abs(dot(x, M * x) - 1) ≤ 1e-10 for x in solutions[2:end])
    @test last(residuals) ≤ 1e-7
    @test Energies.physical_energy(e, solution) ≈
      Energies.physical_energy(e, reference) atol = 1e-8
  end

  @testset "finite-element nonlinear assembly" begin
    K, M, quartic, cubic, density_matrix, parts, _ =
      FEMDiscretizations.FEM_GrossPitaevskii(5, 2; overlap=1)
    e = Energies.GrossPitaevskiiRayleighQuotient(K, M, 1.0, quartic, cubic)
    u = collect(range(0.2, 1.0; length=size(K, 1)))
    direction = collect(range(-0.4, 0.3; length=length(u)))
    h = 1e-6
    quartic_directional =
      (quartic(u .+ h .* direction) - quartic(u .- h .* direction)) / (2h)
    @test quartic_directional ≈ 4 * dot(cubic(u), direction) rtol = 2e-6
    @test density_matrix(u) * u ≈ cubic(u) rtol = 1e-12 atol = 1e-12

    solution, _, energies, _, residuals = Solvers.var_dd(
      e,
      parts;
      u0=ones(length(u)),
      maxiter=12,
      tol=1e-5,
      history_depth=1,
      verbose=false,
    )
    @test all(diff(energies) .≤ 1e-8)
    @test abs(dot(solution, M * solution) - 1) ≤ 1e-10
    @test last(residuals) ≤ first(residuals)
    @test last(residuals) < 1e-5

    _, gfdn_history = gp_gfdn_au_as_history(
      e,
      density_matrix,
      parts,
      ones(length(u));
      conjugate=false,
      maxiter=12,
      tol=1e-5,
    )
    _, cg_history = gp_gfdn_au_as_history(
      e,
      density_matrix,
      parts,
      ones(length(u));
      conjugate=true,
      maxiter=12,
      tol=1e-5,
    )
    _, exact_cg_history = gp_cg_gfdn_au_exact_history(
      e,
      density_matrix,
      ones(length(u));
      maxiter=12,
      tol=1e-5,
    )
    @test all(diff(getindex.(gfdn_history, 2)) .≤ 1e-10)
    @test all(diff(getindex.(cg_history, 2)) .≤ 1e-10)
    @test all(diff(getindex.(exact_cg_history, 2)) .≤ 1e-10)
    @test last(gfdn_history)[3] < first(gfdn_history)[3]
    @test last(cg_history)[3] < first(cg_history)[3]
    @test last(exact_cg_history)[3] < first(exact_cg_history)[3]
    @test maximum(getindex.(gfdn_history, 4)) ≤ 1e-12
    @test maximum(getindex.(cg_history, 4)) ≤ 1e-12
    @test maximum(getindex.(exact_cg_history, 4)) ≤ 1e-12
  end

  @testset "Henning--Jarlebring exact GFDN benchmark" begin
    N = 32
    K, M, quartic, cubic, density_matrix, _, U =
      FEMDiscretizations.FEM_GrossPitaevskii(
        N,
        1;
        P=hj_gp_potential,
        domain=HJ_GP_DOMAIN,
        quadrature_degree=8,
      )
    e = Energies.GrossPitaevskiiRayleighQuotient(
      K, M, HJ_GP_KAPPA, quartic, cubic
    )
    u0 = hj_gp_initial_vector(U, M)
    solution, history, taus = gp_gfdn_au_exact_history(
      e, density_matrix, u0; maxiter=30
    )

    energies = getindex.(history, 2)
    @test length(history) == 31
    @test length(taus) == 30
    @test all(0 .<= taus .<= 2)
    @test all(diff(energies) .<= 1e-11)
    @test abs(dot(solution, M * solution) - 1) <= 1e-11
    # Discrete 32x32 Q1 reference. Figure 6 of the paper likewise reaches an
    # energy error of order 1e-9 after 30 adaptive exact-GFDN iterations.
    @test abs(last(energies) - 10.929093809728007) <= 3e-9
  end
end
