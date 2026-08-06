# Study 8b: controlled sensitivity experiments for the linear source problem.
#
# The full Cartesian product would obscure the interpretation. We instead
# change one feature at a time:
#   mesh:       h, h/2, h/4 with fixed overlap layers or fixed delta/H;
#   overlap:    a direct layer sweep at fixed h and m;
#   contrast:   one off-centre high-diffusivity inclusion;
#   history:    depths 0, 1, 2, 4, 8 on easy and hard coefficients.
#
# Every convergence row records parallel local-solve batches, raw local solves,
# and ideal global matrix-vector products where that count is unambiguous.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
isdefined(Main, :gmres_ras) || include("study8_linear_cmp.jl")

const STUDY8_SENSITIVITY_METHODS = (
  "var_dd_additive", "var_dd_additive_history", "pcg_as", "gmres_ras"
)

function inclusion_diffusion(contrast)
  return x -> begin
    x1, x2 = x.data
    (x1 - 0.57)^2 + (x2 - 0.43)^2 <= 0.18^2 ? contrast : 1.0
  end
end

function study8_method_history(
  method,
  K,
  b,
  dofspar,
  schwarz;
  maxiter,
  relative_tolerance,
  history_depth=1,
  subspace_callback=nothing,
)
  initial_residual = norm(b - K * ones(length(b)))
  absolute_tolerance = relative_tolerance * initial_residual
  history = if method == "var_dd_additive"
    var_dd_linear_history(
      K, b, dofspar; maxiter, tol=absolute_tolerance, history_depth=0
    )
  elseif method == "var_dd_additive_history"
    var_dd_linear_history(
      K,
      b,
      dofspar;
      maxiter,
      tol=absolute_tolerance,
      history_depth,
      subspace_callback,
    )
  elseif method == "pcg_as"
    pcg_as(K, b, schwarz; maxiter, tol=absolute_tolerance)
  elseif method == "gmres_ras"
    gmres_ras(K, b, schwarz; maxiter, tol=absolute_tolerance)
  else
    throw(ArgumentError("unknown Study 8 sensitivity method $method"))
  end
  return [
    (solves, residual / initial_residual) for (solves, residual) in history
  ]
end

"CG iteration count used only as a right-hand-side-dependent work diagnostic."
function local_cg_work(A, rhs; relative_tolerance=1e-8, maxiter=2000)
  norm(rhs) == 0 && return 0
  x = zeros(length(rhs))
  r = copy(rhs)
  p = copy(r)
  rr = dot(r, r)
  target = relative_tolerance * sqrt(rr)
  for iteration in 1:maxiter
    Ap = A * p
    curvature = dot(p, Ap)
    curvature > 0 || return maxiter
    alpha = rr / curvature
    x .+= alpha .* p
    r .-= alpha .* Ap
    sqrt(dot(r, r)) <= target && return iteration
    rr_new = dot(r, r)
    p .= r .+ (rr_new / rr) .* p
    rr = rr_new
  end
  return maxiter
end

"Estimate the spectral condition number of an SPD matrix."
function spd_condition_estimate(A)
  n = size(A, 1)
  n == 0 && return NaN
  values = if n <= 350
    eigvals(Symmetric(Matrix(A)))
  else
    largest = real(only(Arpack.eigs(A; nev=1, which=:LM, ritzvec=false)[1]))
    # Shift-invert is substantially more reliable than `which=:SM` for the
    # high-contrast SPD blocks in this study.
    smallest = real(only(Arpack.eigs(
      A; nev=1, sigma=0.0, which=:LM, ritzvec=false
    )[1]))
    [smallest, largest]
  end
  minimum(values) > 0 || return Inf
  return maximum(values) / minimum(values)
end

function record_study8_diagnostics!(
  rows, experiment, regime, N, m, overlap, contrast, K, b, dofspar
)
  u0 = ones(length(b))
  residual0 = b - K * u0

  for (subdomain, raw_dofs) in enumerate(dofspar)
    dofs = collect(Int, raw_dofs)
    local_matrix = sparse(K[dofs, dofs])
    push!(rows.experiment, experiment)
    push!(rows.regime, regime)
    push!(rows.system, "schwarz_local")
    push!(rows.subdomain, subdomain)
    push!(rows.N, N)
    push!(rows.m, m)
    push!(rows.overlap, overlap)
    push!(rows.contrast, contrast)
    push!(rows.ndofs, size(K, 1))
    push!(rows.system_size, length(dofs))
    push!(rows.condition_estimate, spd_condition_estimate(local_matrix))
    push!(rows.cg_iterations, local_cg_work(local_matrix, residual0[dofs]))
    push!(rows.effective_rank, length(dofs))

    # Orthonormal form of span{u0, e_j: j in I_i}: the coordinate vectors on
    # I_i plus the normalized part of u0 outside I_i. This avoids reporting a
    # condition number caused solely by scaling of the raw coefficient basis.
    complement = copy(u0)
    complement[dofs] .= 0
    complement_norm = norm(complement)
    complement_norm == 0 && continue
    complement ./= complement_norm
    Kcomplement = K * complement
    cross = Kcomplement[dofs]
    augmented = [
      sparse(reshape([dot(complement, Kcomplement)], 1, 1)) sparse(reshape(cross, 1, :));
      sparse(reshape(cross, :, 1)) local_matrix
    ]
    augmented_rhs = vcat(dot(complement, b), b[dofs])
    push!(rows.experiment, experiment)
    push!(rows.regime, regime)
    push!(rows.system, "vardd_local_orthonormal")
    push!(rows.subdomain, subdomain)
    push!(rows.N, N)
    push!(rows.m, m)
    push!(rows.overlap, overlap)
    push!(rows.contrast, contrast)
    push!(rows.ndofs, size(K, 1))
    push!(rows.system_size, size(augmented, 1))
    push!(rows.condition_estimate, spd_condition_estimate(augmented))
    push!(rows.cg_iterations, local_cg_work(augmented, augmented_rhs))
    push!(rows.effective_rank, size(augmented, 1))
  end

  energy = Energies.QuadraticEnergy(K, b)
  candidates = hcat(
    u0, [Solvers.inf_step(energy, u0, dofs) for dofs in dofspar]...
  )
  basis = Solvers.orthonormal_basis(candidates)
  reduced = Symmetric(basis' * K * basis)
  push!(rows.experiment, experiment)
  push!(rows.regime, regime)
  push!(rows.system, "combine_first_sweep")
  push!(rows.subdomain, 0)
  push!(rows.N, N)
  push!(rows.m, m)
  push!(rows.overlap, overlap)
  push!(rows.contrast, contrast)
  push!(rows.ndofs, size(K, 1))
  push!(rows.system_size, size(candidates, 2))
  push!(rows.condition_estimate, spd_condition_estimate(reduced))
  push!(rows.cg_iterations, local_cg_work(reduced, basis' * b))
  push!(rows.effective_rank, size(basis, 2))
end

function run_study8_sensitivity()
  files = (
    "study8_sensitivity.csv",
    "study8_inner_systems.csv",
    "study8_history_subspaces.csv",
  )
  if !needs_run(files...)
    println("study8 sensitivity: cached, skipping")
    return nothing
  end
  println("study8 sensitivity: mesh, overlap, contrast, history, inner work")
  Random.seed!(1)

  convergence = (
    experiment=String[],
    regime=String[],
    method=String[],
    N=Int[],
    m=Int[],
    overlap=Int[],
    contrast=Float64[],
    history_depth=Int[],
    delta_over_H=Float64[],
    iteration=Int[],
    local_batches=Int[],
    local_solves=Int[],
    global_matvecs=Int[],
    relative_residual=Float64[],
  )
  diagnostics = (
    experiment=String[],
    regime=String[],
    system=String[],
    subdomain=Int[],
    N=Int[],
    m=Int[],
    overlap=Int[],
    contrast=Float64[],
    ndofs=Int[],
    system_size=Int[],
    condition_estimate=Float64[],
    cg_iterations=Int[],
    effective_rank=Int[],
  )
  history_subspaces = (
    regime=String[],
    contrast=Float64[],
    history_depth=Int[],
    iteration=Int[],
    basis_columns=Int[],
    effective_rank=Int[],
    condition_estimate=Float64[],
  )
  diagnosed = Set{Tuple{String,String,Int,Int,Int,Float64}}()
  relative_tolerance = SMALL ? 1e-7 : 1e-10
  maxiter = SMALL ? 35 : 300

  function run_configuration(
    experiment,
    regime,
    N,
    m,
    overlap,
    contrast;
    methods=STUDY8_SENSITIVITY_METHODS,
    history_depth=1,
    partitioning=:metis,
  )
    K, _, b, dofspar, _, core_dofs = laplace_setup(
      N,
      m,
      overlap;
      diffusion=inclusion_diffusion(contrast),
      partitioning,
      return_core_partition=true,
    )
    schwarz = schwarz_setup(K, dofspar; core_dofs)
    delta_over_H = overlap * sqrt(m) / N
    for method in methods
      callback = if experiment == "history"
        (iteration, candidates) -> begin
          basis = Solvers.orthonormal_basis(candidates)
          reduced = Symmetric(basis' * K * basis)
          push!(history_subspaces.regime, regime)
          push!(history_subspaces.contrast, contrast)
          push!(history_subspaces.history_depth, history_depth)
          push!(history_subspaces.iteration, iteration)
          push!(history_subspaces.basis_columns, size(candidates, 2))
          push!(history_subspaces.effective_rank, size(basis, 2))
          push!(
            history_subspaces.condition_estimate,
            spd_condition_estimate(reduced),
          )
        end
      else
        nothing
      end
      history = study8_method_history(
        method,
        K,
        b,
        dofspar,
        schwarz;
        maxiter,
        relative_tolerance,
        history_depth,
        subspace_callback=callback,
      )
      for (iteration, (solves, residual)) in enumerate(history)
        batches = solves ÷ m
        push!(convergence.experiment, experiment)
        push!(convergence.regime, regime)
        push!(convergence.method, method)
        push!(convergence.N, N)
        push!(convergence.m, m)
        push!(convergence.overlap, overlap)
        push!(convergence.contrast, contrast)
        push!(
          convergence.history_depth,
          method == "var_dd_additive_history" ? history_depth : 0,
        )
        push!(convergence.delta_over_H, delta_over_H)
        push!(convergence.iteration, iteration - 1)
        push!(convergence.local_batches, batches)
        push!(convergence.local_solves, solves)
        push!(
          convergence.global_matvecs,
          method in ("pcg_as", "gmres_ras") ? 1 + batches : -1,
        )
        push!(convergence.relative_residual, residual)
      end
    end

    key = (String(experiment), String(regime), N, m, overlap, Float64(contrast))
    if key ∉ diagnosed
      record_study8_diagnostics!(
        diagnostics, experiment, regime, N, m, overlap, contrast, K, b, dofspar
      )
      push!(diagnosed, key)
    end
    println(
      "  $experiment/$regime: N=$N, m=$m, overlap=$overlap, contrast=$contrast"
    )
  end

  # In the fixed-relative-overlap sequence, N=20j and overlap=j give
  # delta/H ≈ overlap*sqrt(m)/N = 0.1 exactly for m=4.
  mesh_sizes = SMALL ? [10, 20, 30] : collect(20:20:120)
  m = 4
  for N in mesh_sizes
    run_configuration(
      "mesh", "fixed_layers", N, m, 2, 1.0; partitioning=:cartesian
    )
    scaled_overlap = SMALL ? max(1, N ÷ 10) : N ÷ 20
    run_configuration(
      "mesh",
      "fixed_delta_over_H",
      N,
      m,
      scaled_overlap,
      1.0;
      partitioning=:cartesian,
    )
  end

  overlap_N = SMALL ? 16 : 64
  for overlap in (SMALL ? [1, 2] : [1, 2, 4, 8])
    run_configuration("overlap", "layer_sweep", overlap_N, m, overlap, 1.0)
  end

  contrast_N = SMALL ? 16 : 64
  contrasts = SMALL ? [1.0, 1e1, 1e2] : 10.0 .^ (0:6)
  for contrast in contrasts
    run_configuration("contrast", "inclusion", contrast_N, m, 2, contrast)
  end

  history_depths = SMALL ? [0, 1, 2] : [0, 1, 2, 4, 8]
  history_contrasts = [1.0, 1e4]
  for contrast in history_contrasts, depth in history_depths
    run_configuration(
      "history",
      contrast == 1 ? "homogeneous" : "contrast_1e4",
      contrast_N,
      m,
      2,
      contrast;
      methods=("var_dd_additive_history",),
      history_depth=depth,
    )
  end

  savetable("study8_sensitivity.csv", convergence)
  savetable("study8_inner_systems.csv", diagnostics)
  savetable("study8_history_subspaces.csv", history_subspaces)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_study8_sensitivity()
end
