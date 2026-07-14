using Test
using LinearAlgebra
using Random
using VariationalDD.Energies
using VariationalDD.Solvers

@testset "Subspace solver robustness" begin
  @testset "local minimizer may have zero current-iterate coefficient" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [1.0, 0.0, 0.0]
    u0 = [0.0, 1.0, 0.0]
    energy = Energies.QuadraticEnergy(A, b)

    # In span{e_2, e_1}, the exact minimizer is e_1, whose coefficient in
    # front of the current iterate e_2 is zero.
    local_minimizer = Solvers.inf_step(energy, copy(u0), Int32[1])
    @test all(isfinite, local_minimizer)
    @test local_minimizer ≈ b

    solution, _, _, _, residuals = Solvers.var_dd(
      energy, [Int32[1]]; u0=u0, maxiter=1, tol=1e-14, verbose=false
    )
    @test solution ≈ b
    @test only(residuals) ≤ 1e-14
  end

  @testset "rank-revealing combination basis" begin
    x = [1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 0.0, 1.0]
    candidates = hcat(x, 2 .* x, y, x .+ y)
    B = Solvers.orthonormal_basis(candidates)

    @test size(B) == (4, 2)
    @test B' * B ≈ Matrix{Float64}(I, 2, 2)
    @test norm(candidates - B * (B' * candidates)) ≤ 1e-12

    # Repeated candidates must not cause QR to add arbitrary directions.
    A = Diagonal([1.0, 2.0, 3.0, 4.0])
    b = [2.0, -1.0, 3.0, 0.5]
    energy = Energies.QuadraticEnergy(A, b)
    repeated = hcat(x, x, 2 .* x)
    combined = Solvers.combine_step(energy, repeated)
    expected = (dot(x, b) / dot(x, A * x)) .* x
    @test combined ≈ expected
  end

  @testset "candidate scaling leaves the combination unchanged" begin
    Random.seed!(41)
    n = 8
    G = randn(n, n)
    A = G' * G + I
    b = randn(n)
    u = randn(n)
    idx = Int32[2, 4, 6]
    energy = Energies.QuadraticEnergy(A, b)

    C = zeros(n, 1 + length(idx))
    C[:, 1] .= u
    for (k, j) in pairs(idx)
      C[j, k + 1] = 1.0
    end
    α = (C' * A * C) \ (C' * b)
    @test abs(α[1]) > 1e-8

    local_minimizer = Solvers.inf_step(energy, copy(u), idx)
    rescaled_candidate = local_minimizer ./ α[1]
    combined_raw = Solvers.combine_step(energy, hcat(u, local_minimizer))
    combined_rescaled = Solvers.combine_step(
      energy, hcat(u, rescaled_candidate)
    )
    @test combined_raw ≈ combined_rescaled atol = 1e-12 rtol = 1e-12
  end

  @testset "quadratic energy is monotone under global recombination" begin
    Random.seed!(17)
    n = 12
    G = randn(n, n)
    A = G' * G + I
    b = randn(n)
    energy = Energies.QuadraticEnergy(A, b)
    parts = [Int32[1, 2, 3, 4], Int32[4, 5, 6, 7, 8], Int32[8, 9, 10, 11, 12]]
    u = randn(n)

    for _ in 1:8
      candidates = [Solvers.inf_step(energy, copy(u), idx) for idx in parts]
      u_new = Solvers.combine_step(energy, hcat(u, candidates...))
      @test Energies.energy(energy, u_new) ≤
        Energies.energy(energy, u) +
            1e-12 * max(1.0, abs(Energies.energy(energy, u)))
      u = u_new
    end
  end

  @testset "previous-iterate enrichment and post-mixing" begin
    Random.seed!(29)
    n = 10
    G = randn(n, n)
    A = G' * G + I
    b = randn(n)
    energy = Energies.QuadraticEnergy(A, b)
    parts = [Int32[1, 2, 3, 4], Int32[4, 5, 6, 7], Int32[7, 8, 9, 10]]
    u0 = randn(n)

    baseline = @test_logs (:warn, r"Reached maxiter=2") Solvers.var_dd(
      energy,
      parts;
      u0=u0,
      maxiter=2,
      tol=0.0,
      verbose=false,
    )
    with_previous = @test_logs (:warn, r"Reached maxiter=2") Solvers.var_dd(
      energy,
      parts;
      u0=u0,
      maxiter=2,
      tol=0.0,
      history_depth=1,
      verbose=false,
    )
    @test with_previous[3][1:2] ≈ baseline[3][1:2]
    @test with_previous[3][3] ≤ baseline[3][3] + 1e-12

    one_step = @test_logs (:warn, r"Reached maxiter=1") Solvers.var_dd(
      energy,
      parts;
      u0=u0,
      maxiter=1,
      tol=0.0,
      verbose=false,
    )
    omega = 0.4
    mixed = @test_logs (:warn, r"Reached maxiter=1") Solvers.var_dd(
      energy,
      parts;
      u0=u0,
      maxiter=1,
      tol=0.0,
      mixing_omega=omega,
      verbose=false,
    )
    @test mixed[1] ≈ omega .* u0 .+ (1 - omega) .* one_step[1]
    @test Energies.energy(energy, one_step[1]) ≤
          Energies.energy(energy, mixed[1]) + 1e-12

    generalized_energy = Energies.GeneralizedRayleighQuotient(
      Diagonal([1.0, 2.0]),
      Matrix{Float64}(I, 2, 2),
    )
    aligned_mix = Solvers.mix_iterates(
      generalized_energy,
      [1.0, 0.0],
      [-1.0, 0.0],
      0.5,
    )
    @test aligned_mix ≈ [1.0, 0.0]
    @test dot(aligned_mix, generalized_energy.B * aligned_mix) ≈ 1.0

    @test_throws ArgumentError Solvers.var_dd(energy, parts; history_depth=-1)
    @test_throws ArgumentError Solvers.var_dd(energy, parts; mixing_omega=1.0)
    @test_throws ArgumentError Solvers.var_dd(energy, parts; sweep=:invalid)
  end

  @testset "explicit multiplicative quadratic sweep" begin
    A = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 3.0]
    b = [1.0, 2.0, 3.0]
    energy = Energies.QuadraticEnergy(A, b)
    parts = [Int32[1, 2], Int32[2, 3]]
    u0 = ones(3)

    # Reconstruct one historical serial sweep explicitly: each projectively
    # rescaled local result becomes the input of the next local solve, and the
    # second level combines precisely those sequential candidates.
    sequential = Vector{Vector{Float64}}()
    u_seq = copy(u0)
    for idx in parts
      α = Solvers.quadratic_local_coefficients(energy, u_seq, idx)
      u_seq = copy(u_seq)
      for (k, j) in pairs(idx)
        u_seq[j] += α[k+1] / α[1]
      end
      push!(sequential, copy(u_seq))
    end
    expected = Solvers.combine_step(energy, hcat(u_seq, sequential...))

    result = @test_logs (:warn, r"Reached maxiter=1") Solvers.var_dd(
      energy,
      parts;
      u0=u0,
      maxiter=1,
      tol=0.0,
      sweep=:multiplicative,
      verbose=false,
    )
    @test result[1] ≈ expected
  end
end
