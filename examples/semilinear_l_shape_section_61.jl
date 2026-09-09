# Reproducible EMDD comparison for Experiment 1 in Spicher--Wihler (2026),
# Section 6.1. The problem is
#
#   -Delta u = 12*exp(-u^2) + g  in Omega,
#          u = 0                 on boundary(Omega),
#
# where Omega = (-1,1)^2 \ ([-1,0] x [0,1]). The manufactured solution
#
#   u(x,y) = 2*r^(-4/3)*x*y*(1-x^2)*(1-y^2)
#
# has the paper's re-entrant-corner singularity. The meshes use its grading
# exponent beta=0.4. As in semilinear_quadratic_model_benchmark.jl, this
# compares nonlinear local minimizations with one frozen quadratic model per
# outer EMDD sweep. (The paper's IMEX method instead uses Delta t=3.)
#
# Run from the repository root with
#
#   julia --project=. examples/semilinear_l_shape_section_61.jl
#
# Use SMALL=1 for a single coarse smoke-test case.

using EnergyMinimizingDD
using CairoMakie
using Gridap
using LinearAlgebra
using Printf
using SpecialFunctions
using Statistics

const FEM = EnergyMinimizingDD.FEMDiscretizations
const Energies = EnergyMinimizingDD.Energies
const Solvers = EnergyMinimizingDD.Solvers

const REACTION_AMPLITUDE = 12.0
const MESH_GRADING = 0.4
const MIDPOINT_QUADRATURE = Gridap.ReferenceFEs.GenericQuadrature(
  [Gridap.Point(0.5, 0.0), Gridap.Point(0.0, 0.5), Gridap.Point(0.5, 0.5)],
  fill(1 / 6, 3),
  "triangle edge-midpoint rule",
)

function exact_solution(x)
  x1, x2 = x[1], x[2]
  radius_squared = x1^2 + x2^2
  iszero(radius_squared) && return 0.0
  return 2 * x1 * x2 * (1 - x1^2) * (1 - x2^2) * radius_squared^(-2 / 3)
end

function minus_laplacian_exact(x)
  x1, x2 = x[1], x[2]
  radius_squared = x1^2 + x2^2
  iszero(radius_squared) && return 0.0

  polynomial = 2 * x1 * x2 * (1 - x1^2) * (1 - x2^2)
  derivative_x1 = 2 * (1 - 3x1^2) * x2 * (1 - x2^2)
  derivative_x2 = 2 * x1 * (1 - x1^2) * (1 - 3x2^2)
  laplacian_polynomial = -12x1 * x2 * (1 - x2^2) - 12x2 * x1 * (1 - x1^2)

  return -radius_squared^(-2 / 3) * laplacian_polynomial +
         (8 / 3) *
         radius_squared^(-5 / 3) *
         (x1 * derivative_x1 + x2 * derivative_x2) -
         (16 / 9) * polynomial * radius_squared^(-5 / 3)
end

function forcing(x)
  return minus_laplacian_exact(x) -
         REACTION_AMPLITUDE * exp(-exact_solution(x)^2)
end

potential(s) = -REACTION_AMPLITUDE * sqrt(pi) * erf(s) / 2
potential_gradient(s) = -REACTION_AMPLITUDE * exp(-s^2)
potential_hessian(s) = 2 * REACTION_AMPLITUDE * s * exp(-s^2)

function semilinear_l_shape_problem(N)
  model = FEM.FEM_LShapeModel(N; grading=MESH_GRADING)
  energy_assembler, gradient_assembler, hessian_assembler, subdomains, trial_space, ndofs, _, initial, _ = FEM.FEM_SemilinearPoisson(
    N,
    4;
    potential=potential,
    potential_gradient=potential_gradient,
    potential_hessian=potential_hessian,
    forcing=forcing,
    overlap=2,
    quadrature_rule=MIDPOINT_QUADRATURE,
    initial_guess=x -> 0.0,
    model=model,
  )
  energy = Energies.NonlinearEnergy(
    "Section 6.1 semilinear L-shaped problem",
    energy_assembler,
    gradient_assembler,
    hessian_assembler,
    ndofs,
  )
  exact_fe = interpolate_everywhere(exact_solution, trial_space)
  exact_values = collect(get_free_dof_values(exact_fe))
  return energy, subdomains, trial_space, initial, exact_values
end

function run_vardd(energy, subdomains, initial, exact_values; quadratic_model)
  initial_residual = Energies.residual_norm(energy, initial)
  local_inner_iterations = Ref(0)
  local_solve_count = Ref(0)
  max_local_inner_iterations = Ref(0)
  function record_local_iterations(_, _, info)
    iterations = max(0, info.iterations)
    local_inner_iterations[] += iterations
    local_solve_count[] += 1
    max_local_inner_iterations[] = max(max_local_inner_iterations[], iterations)
    return nothing
  end
  result = Solvers.var_dd(
    energy,
    subdomains;
    u0=initial,
    maxiter=100,
    tol=1e-8 * initial_residual,
    quadratic_model=quadratic_model,
    local_solve_callback=record_local_iterations,
    verbose=false,
  )
  average_local_inner_iterations = if iszero(local_solve_count[])
    0.0
  else
    local_inner_iterations[] / local_solve_count[]
  end
  return (
    solution=result[1],
    sweeps=length(result[3]) - 1,
    relative_residual=result[5][end] / initial_residual,
    relative_nodal_error=norm(result[1] - exact_values) / norm(exact_values),
    monotone=all(diff(result[3]) .<= 1e-11),
    local_inner_iterations=local_inner_iterations[],
    average_local_inner_iterations=average_local_inner_iterations,
    max_local_inner_iterations=max_local_inner_iterations[],
    residual_history=result[5] ./ initial_residual,
  )
end

function save_paraview_export(result)
  vtk_dir = mkpath(joinpath(@__DIR__, "..", "output", "vtk"))
  filename = joinpath(vtk_dir, "semilinear_l_shape_section_61_N$(result.N)")
  writevtk(
    result.trial_space.space.fe_basis.trian,
    filename;
    cellfields=[
      "u_exact" => FEFunction(result.trial_space, result.exact_values),
      "u_nonlinear" =>
        FEFunction(result.trial_space, result.nonlinear.solution),
      "u_quadratic" =>
        FEFunction(result.trial_space, result.quadratic.solution),
    ],
  )
  return println("saved ParaView export to $(filename).vtu")
end

function median_timing(run; repeats=3)
  result = run() # warm up compilation
  times = Float64[]
  for _ in 1:repeats
    GC.gc()
    elapsed = @elapsed result = run()
    push!(times, elapsed)
  end
  return median(times), result
end

function save_benchmark_plots(results)
  figure_dir = mkpath(joinpath(@__DIR__, "figures"))
  pdf_dir = mkpath(joinpath(@__DIR__, "..", "output", "pdf"))
  colors = (:steelblue, :darkorange)

  finest = results[end]
  convergence = Figure(; size=(760, 500), fontsize=17)
  axis = Axis(
    convergence[1, 1];
    xlabel="EMDD sweep",
    ylabel="relative Euler residual",
    yscale=log10,
    title="Section 6.1 L-shaped problem, N=$(finest.N)",
  )
  for (result, label, color, marker) in (
    (finest.nonlinear, "nonlinear local solves", colors[1], :circle),
    (finest.quadratic, "quadratic local model", colors[2], :diamond),
  )
    sweeps = 0:(length(result.residual_history) - 1)
    lines!(
      axis,
      sweeps,
      result.residual_history;
      label=label,
      color=color,
      linewidth=2.5,
    )
    scatter!(
      axis,
      sweeps,
      result.residual_history;
      color=color,
      marker=marker,
      markersize=8,
    )
  end
  axislegend(axis; position=:rt)
  convergence_name = joinpath(
    figure_dir, "semilinear_l_shape_section_61_convergence"
  )
  save(
    joinpath(pdf_dir, "semilinear_l_shape_section_61_convergence.pdf"),
    convergence,
  )
  save(convergence_name * ".png", convergence; px_per_unit=2)

  dofs = [result.dofs for result in results]
  scaling = Figure(; size=(1060, 440), fontsize=17)
  error_axis = Axis(
    scaling[1, 1];
    xlabel="degrees of freedom",
    ylabel="relative nodal error",
    xscale=log10,
    yscale=log10,
    title="discretization error (curves coincide)",
  )
  timing_axis = Axis(
    scaling[1, 2];
    xlabel="degrees of freedom",
    ylabel="median runtime [s]",
    xscale=log10,
    yscale=log10,
    title="solver runtime",
  )
  for (field, label, color, marker) in (
    (:nonlinear, "nonlinear local solves", colors[1], :circle),
    (:quadratic, "quadratic local model", colors[2], :diamond),
  )
    values = [getproperty(result, field) for result in results]
    errors = [value.relative_nodal_error for value in values]
    times = [getproperty(result, Symbol(field, "_time")) for result in results]
    lines!(error_axis, dofs, errors; label=label, color=color, linewidth=2.5)
    scatter!(error_axis, dofs, errors; color=color, marker=marker, markersize=9)
    lines!(timing_axis, dofs, times; label=label, color=color, linewidth=2.5)
    scatter!(timing_axis, dofs, times; color=color, marker=marker, markersize=9)
  end
  axislegend(timing_axis; position=:lt)
  scaling_name = joinpath(figure_dir, "semilinear_l_shape_section_61_scaling")
  save(joinpath(pdf_dir, "semilinear_l_shape_section_61_scaling.pdf"), scaling)
  save(scaling_name * ".png", scaling; px_per_unit=2)
  return println("saved PNG plots to $figure_dir and PDF plots to $pdf_dir")
end

function run_semilinear_l_shape_benchmark()
  small = get(ENV, "SMALL", "0") == "1"
  mesh_sizes = small ? (4,) : (4, 8, 16)
  repeats = small ? 1 : 3

  println("Section 6.1 semilinear L-shaped benchmark (m=4, beta=0.4)")
  println(
    "   N   dofs | nonlinear local solves: sweeps  total    avg  max  relres   nodalerr  median(s) | ",
    "quadratic model: sweeps  relres   nodalerr  median(s)",
  )
  benchmark_results = NamedTuple[]
  for N in mesh_sizes
    energy, subdomains, trial_space, initial, exact_values = semilinear_l_shape_problem(
      N
    )
    nonlinear_time, nonlinear = median_timing(
      () -> run_vardd(
        energy, subdomains, initial, exact_values; quadratic_model=false
      );
      repeats=repeats,
    )
    quadratic_time, quadratic = median_timing(
      () -> run_vardd(
        energy, subdomains, initial, exact_values; quadratic_model=true
      );
      repeats=repeats,
    )
    @printf(
      "%4d %6d | %6d %6d %6.2f %4d  %.2e  %.2e  %9.6f | %6d  %.2e  %.2e  %9.6f\n",
      N,
      length(initial),
      nonlinear.sweeps,
      nonlinear.local_inner_iterations,
      nonlinear.average_local_inner_iterations,
      nonlinear.max_local_inner_iterations,
      nonlinear.relative_residual,
      nonlinear.relative_nodal_error,
      nonlinear_time,
      quadratic.sweeps,
      quadratic.relative_residual,
      quadratic.relative_nodal_error,
      quadratic_time,
    )
    push!(
      benchmark_results,
      (
        N=N,
        dofs=length(initial),
        trial_space=trial_space,
        exact_values=exact_values,
        nonlinear=nonlinear,
        nonlinear_time=nonlinear_time,
        quadratic=quadratic,
        quadratic_time=quadratic_time,
      ),
    )
  end

  save_benchmark_plots(benchmark_results)
  return save_paraview_export(benchmark_results[end])
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_semilinear_l_shape_benchmark()
end
