using Test
using LinearAlgebra
using SparseArrays
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
      K,
      M,
      -1.0,
      quartic,
      cubic,
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
      Matrix(Tridiagonal(fill(-0.4, n - 1), fill(2.5, n), fill(-0.4, n - 1))),
    )
    M = Diagonal(collect(range(0.8, 1.2; length = n)))
    weights = collect(range(0.5, 1.0; length = n))
    quartic(u) = sum(weights .* u .^ 4)
    cubic(u) = weights .* u .^ 3
    density(u) = Diagonal(weights .* u .^ 2)
    e = Energies.GrossPitaevskiiRayleighQuotient(K, M, 3.0, quartic, cubic)
    e_with_density = Energies.GrossPitaevskiiRayleighQuotient(
      K,
      M,
      3.0,
      quartic,
      cubic;
      density_matrix = density,
    )
    u0 = ones(n)
    reference = Solvers.combine_step(
      e,
      hcat(u0, Matrix{Float64}(I, n, n));
      initial = u0,
      maxiter = 1000,
      tol = 1e-11,
    )
    @test abs(dot(reference, M * reference) - 1) ≤ 1e-12
    @test Energies.residual_norm(e, reference) ≤ 1e-7
    explicit_density_reference = Solvers.combine_step(
      e_with_density,
      hcat(u0, Matrix{Float64}(I, n, n));
      initial = u0,
      maxiter = 1000,
      tol = 1e-11,
    )
    @test abs(dot(reference, M * explicit_density_reference)) ≈ 1.0 atol = 1e-9
    @test Energies.energy(e, explicit_density_reference) ≈
          Energies.energy(e, reference) atol = 1e-10
    scaled_reference = Solvers.combine_step(
      e,
      hcat(u0, Matrix{Float64}(I, n, n));
      initial = 7 .* u0,
      maxiter = 1000,
      tol = 1e-11,
    )
    @test abs(dot(reference, M * scaled_reference)) ≈ 1.0 atol = 1e-9
    @test Energies.energy(e, scaled_reference) ≈ Energies.energy(e, reference) atol =
      1e-10
    @test_throws ArgumentError Solvers.combine_step(e, zeros(n, 2))

    parts = [Int32[1, 2, 3, 4, 5], Int32[4, 5, 6, 7, 8]]
    normalized_u0 = copy(u0)
    Energies.normalize_M!(normalized_u0, M)
    frozen = Energies.frozen_density_model(e_with_density, u0)
    @test frozen.A ≈ K + 3.0 .* density(normalized_u0)
    @test frozen.B === M
    @test_throws ArgumentError Energies.frozen_density_model(e, u0)

    previous = collect(range(0.3, 1.1; length = n))
    normalized_previous = copy(previous)
    Energies.normalize_M!(normalized_previous, M)
    mixed_frozen =
      Energies.frozen_density_model(e_with_density, u0; previous, alpha = 0.25)
    @test mixed_frozen.A ≈
          K +
          3.0 .* (
      0.25 .* density(normalized_u0) .+ 0.75 .* density(normalized_previous)
    )
    @test_throws ArgumentError Energies.frozen_density_model(
      e_with_density,
      u0;
      previous,
      alpha = 0.0,
    )
    @test_throws DimensionMismatch Energies.frozen_density_model(
      e_with_density,
      u0;
      previous = ones(n - 1),
      alpha = 0.5,
    )

    tangent_model = Energies.tangent_quadratic_model(e_with_density, u0)
    @test tangent_model.u ≈ normalized_u0
    @test tangent_model.residual ≈
          Energies.projected_residual(e_with_density, normalized_u0)
    @test tangent_model.H ≈ K + 9.0 .* density(normalized_u0)
    @test abs(dot(tangent_model.u, M * tangent_model.u) - 1) ≤ 1e-12
    @test abs(dot(tangent_model.u, tangent_model.residual)) ≤ 1e-11
    @test_throws ArgumentError Energies.tangent_quadratic_model(e, u0)

    projected_model = Energies.projected_newton_model(e_with_density, u0)
    projected_mu = Energies.chemical_potential(e_with_density, normalized_u0)
    @test projected_model.u ≈ normalized_u0
    @test projected_model.residual ≈
          Energies.projected_residual(e_with_density, normalized_u0)
    @test projected_model.H_lagrangian ≈
          K + 9.0 .* density(normalized_u0) - projected_mu .* M
    @test abs(dot(projected_model.u, projected_model.residual)) ≤ 1e-11
    @test_throws ArgumentError Energies.projected_newton_model(e, u0)

    raw_direction = collect(range(-0.7, 0.9; length = n))
    tangent_direction =
      raw_direction .- normalized_u0 .* dot(normalized_u0, M * raw_direction)
    function retracted_energy(t)
      trial = normalized_u0 .+ t .* tangent_direction
      Energies.normalize_M!(trial, M)
      return Energies.physical_energy(e_with_density, trial)
    end
    model_energy(t) =
      Energies.physical_energy(e_with_density, normalized_u0) +
      t * dot(projected_model.residual, tangent_direction) +
      0.5t^2 *
      dot(tangent_direction, projected_model.H_lagrangian * tangent_direction)
    error_large = abs(retracted_energy(2e-3) - model_energy(2e-3))
    error_small = abs(retracted_energy(1e-3) - model_energy(1e-3))
    @test error_large / error_small ≈ 8 rtol=0.08

    nearly_dependent_increments =
      hcat(1e-9 .* tangent_direction, 2e-9 .* tangent_direction)
    anchored_basis = Solvers.anchored_m_orthonormal_basis(
      normalized_u0,
      nearly_dependent_increments,
      M,
    )
    @test size(anchored_basis, 2) == 2
    @test anchored_basis' * M * anchored_basis ≈ I atol=1e-10

    frozen_candidates =
      hcat([Solvers.inf_step(frozen, u0, part) for part in parts]...)
    expected_quadratic_step = Solvers.combine_step(
      e_with_density,
      hcat(u0, frozen_candidates);
      initial = u0,
    )
    quadratic_step, _, quadratic_energies, quadratic_solutions, _ =
      Solvers.var_dd(
        e_with_density,
        parts;
        u0 = u0,
        maxiter = 1,
        tol = eps(Float64),
        frozen_gp_model = true,
        verbose = false,
      )
    @test min(
      norm(quadratic_step - expected_quadratic_step),
      norm(quadratic_step + expected_quadratic_step),
    ) ≤ 1e-7
    @test all(diff(quadratic_energies) .≤ 1e-10)
    @test abs(
      dot(last(quadratic_solutions), M * last(quadratic_solutions)) - 1,
    ) ≤ 1e-10

    mixed_first_step, _, _, _, _ = Solvers.var_dd(
      e_with_density,
      parts;
      u0 = u0,
      maxiter = 1,
      tol = eps(Float64),
      frozen_gp_model = true,
      density_mixing_alpha = 0.25,
      verbose = false,
    )
    @test min(
      norm(mixed_first_step - quadratic_step),
      norm(mixed_first_step + quadratic_step),
    ) ≤ 1e-7
    mixed_solution, _, mixed_energies, mixed_solutions, mixed_residuals =
      Solvers.var_dd(
        e_with_density,
        parts;
        u0 = u0,
        maxiter = 3,
        tol = eps(Float64),
        history_depth = 1,
        frozen_gp_model = true,
        density_mixing_alpha = 0.5,
        verbose = false,
      )
    @test all(isfinite, mixed_energies)
    @test all(diff(mixed_energies) .≤ 1e-10)
    @test all(isfinite, mixed_residuals)
    @test abs(dot(mixed_solution, M * mixed_solution) - 1) ≤ 1e-10
    @test all(abs(dot(x, M * x) - 1) ≤ 1e-10 for x in mixed_solutions[2:end])
    @test_throws ArgumentError Solvers.var_dd(
      e_with_density,
      parts;
      density_mixing_alpha = 0.5,
      verbose = false,
    )

    projected_candidates = [
      Solvers.projected_newton_gp_step(projected_model, part) for part in parts
    ]
    @test all(candidate.info.converged for candidate in projected_candidates)
    @test all(
      dot(projected_model.residual, candidate.u .- projected_model.u) < 0 for
      candidate in projected_candidates
    )
    projected_solution,
    _,
    projected_energies,
    projected_solutions,
    projected_residuals = Solvers.var_dd(
      e_with_density,
      parts;
      u0 = u0,
      maxiter = 20,
      tol = 1e-7,
      history_depth = 1,
      projected_gp_model = true,
      verbose = false,
    )
    @test all(diff(projected_energies) .≤ 1e-10)
    @test all(
      abs(dot(x, M * x) - 1) ≤ 1e-10 for x in projected_solutions[2:end]
    )
    @test last(projected_residuals) ≤ 1e-7
    @test Energies.physical_energy(e_with_density, projected_solution) ≈
          Energies.physical_energy(e_with_density, reference) atol=1e-8
    @test_throws ArgumentError Solvers.var_dd(
      Energies.GeneralizedRayleighQuotient(K, M),
      parts;
      projected_gp_model = true,
      verbose = false,
    )

    tangent_candidates =
      hcat([Solvers.inf_step(tangent_model, u0, part) for part in parts]...)
    @test all(
      abs(dot(candidate, M * candidate) - 1) ≤ 1e-10 for
      candidate in eachcol(tangent_candidates)
    )
    expected_tangent_step = Solvers.combine_step(
      e_with_density,
      hcat(u0, tangent_candidates);
      initial = u0,
    )
    tangent_step, _, tangent_energies, tangent_solutions, _ = Solvers.var_dd(
      e_with_density,
      parts;
      u0 = u0,
      maxiter = 1,
      tol = eps(Float64),
      tangent_gp_model = true,
      verbose = false,
    )
    @test min(
      norm(tangent_step - expected_tangent_step),
      norm(tangent_step + expected_tangent_step),
    ) ≤ 1e-7
    @test all(diff(tangent_energies) .≤ 1e-10)
    @test abs(dot(last(tangent_solutions), M * last(tangent_solutions)) - 1) ≤
          1e-10
    @test_throws ArgumentError Solvers.var_dd(
      Energies.GeneralizedRayleighQuotient(K, M),
      parts;
      tangent_gp_model = true,
      verbose = false,
    )
    @test_throws ArgumentError Solvers.var_dd(
      e_with_density,
      parts;
      frozen_gp_model = true,
      tangent_gp_model = true,
      verbose = false,
    )

    solution, _, energies, solutions, residuals = Solvers.var_dd(
      e,
      parts;
      u0 = u0,
      maxiter = 40,
      tol = 1e-7,
      history_depth = 1,
      verbose = false,
    )
    @test all(diff(energies) .≤ 1e-10)
    @test all(abs(dot(x, M * x) - 1) ≤ 1e-10 for x in solutions[2:end])
    @test last(residuals) ≤ 1e-7
    @test Energies.physical_energy(e, solution) ≈
          Energies.physical_energy(e, reference) atol = 1e-8
    @test_throws ArgumentError Solvers.var_dd(
      Energies.GeneralizedRayleighQuotient(K, M),
      parts;
      frozen_gp_model = true,
      verbose = false,
    )
  end

  @testset "finite-element nonlinear assembly" begin
    K, M, quartic, cubic, density_matrix, parts, _ =
      FEMDiscretizations.FEM_GrossPitaevskii(5, 2; overlap = 1)
    e = Energies.GrossPitaevskiiRayleighQuotient(
      K,
      M,
      1.0,
      quartic,
      cubic;
      density_matrix,
    )
    u = collect(range(0.2, 1.0; length = size(K, 1)))
    direction = collect(range(-0.4, 0.3; length = length(u)))
    h = 1e-6
    quartic_directional =
      (quartic(u .+ h .* direction) - quartic(u .- h .* direction)) / (2h)
    @test quartic_directional ≈ 4 * dot(cubic(u), direction) rtol = 2e-6
    @test density_matrix(u) * u ≈ cubic(u) rtol = 1e-12 atol = 1e-12

    solution, _, energies, _, residuals = Solvers.var_dd(
      e,
      parts;
      u0 = ones(length(u)),
      maxiter = 12,
      tol = 1e-5,
      history_depth = 1,
      verbose = false,
    )
    @test all(diff(energies) .≤ 1e-8)
    @test abs(dot(solution, M * solution) - 1) ≤ 1e-10
    @test last(residuals) ≤ first(residuals)
    @test last(residuals) < 1e-5

    histories = []
    for conjugate in (false, true), inner_iterations in (1, 2, 4)
      _, history = gp_gfdn_au_pcg_as_history(
        e,
        density_matrix,
        parts,
        ones(length(u));
        conjugate,
        inner_iterations,
        maxiter = 12,
        tol = 1e-5,
      )
      push!(histories, history)
      @test all(diff(getindex.(history, 2)) .≤ 1e-10)
      @test last(history)[3] < first(history)[3]
      @test maximum(getindex.(history, 4)) ≤ 1e-12
    end

    normalized = ones(length(u))
    Energies.normalize_M!(normalized, M)
    A_u = sparse(K + density_matrix(normalized))
    rhs =
      A_u * normalized .- dot(normalized, A_u * normalized) .* (M * normalized)
    as_direction = apply_AS(schwarz_setup(A_u, parts), rhs)
    pcg1_direction = gp_fixed_pcg_as(A_u, rhs, parts, 1)
    @test abs(dot(as_direction, pcg1_direction)) ≈
          norm(as_direction) * norm(pcg1_direction) rtol=1e-12
  end

  @testset "Henning--Jarlebring exact GFDN benchmark" begin
    N = 32
    K, M, quartic, cubic, density_matrix, _, U =
      FEMDiscretizations.FEM_GrossPitaevskii(
        N,
        1;
        P = hj_gp_potential,
        domain = HJ_GP_DOMAIN,
        quadrature_degree = 8,
      )
    e = Energies.GrossPitaevskiiRayleighQuotient(
      K,
      M,
      HJ_GP_KAPPA,
      quartic,
      cubic,
    )
    u0 = hj_gp_initial_vector(U, M)
    solution, history, taus =
      gp_gfdn_au_exact_history(e, density_matrix, u0; maxiter = 30)

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
