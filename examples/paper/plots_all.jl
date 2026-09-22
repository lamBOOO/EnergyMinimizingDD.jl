# Regenerate the five figures included in the final paper from cached CSV data.

isdefined(Main, :PAPER_COMMON) || include("common.jl")
using CairoMakie

set_theme!(Theme(
  fontsize=18,
  Axis=(xticklabelsize=16, yticklabelsize=16, xlabelsize=20,
        ylabelsize=20, titlesize=20, xgridcolor=(:black, 0.12),
        ygridcolor=(:black, 0.12), spinewidth=1.2),
  Lines=(linewidth=2.5,),
  Legend=(labelsize=16, framevisible=true),
))

const PAPER_WIDTH = 1000
const MARKER_SIZE = 10
tab10_colors(n) = [Makie.to_colormap(:tab10)[mod1(i, 10)] for i = 1:n]
pick(table, column, mask) = collect(getproperty(table, column)[mask])

function savefigs(figure, name)
  save(joinpath(FIG_DIR, name * ".pdf"), figure)
  save(joinpath(FIG_DIR, name * ".png"), figure; px_per_unit=2)
  println("  saved $(name).{pdf,png}")
  return figure
end

function add_series!(axis, x, y; color, marker, linestyle=:solid,
                     linewidth=2.5, marker_stride=8, markersize=MARKER_SIZE)
  lines!(axis, x, y; color, linestyle, linewidth)
  indices = collect(1:marker_stride:length(x))
  !isempty(x) && last(indices) != length(x) && push!(indices, length(x))
  scatter!(axis, x[indices], y[indices]; color, marker, markersize,
           strokecolor=:black, strokewidth=0.7)
end

function legend_elements(methods, markers, colors;
                         linestyles=fill(:solid, length(methods)),
                         linewidths=fill(2.5, length(methods)),
                         markersizes=fill(MARKER_SIZE, length(methods)))
  return [[
    LineElement(color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i]),
    MarkerElement(color=colors[i], marker=markers[method],
                  markersize=markersizes[i], strokecolor=:black,
                  strokewidth=0.7),
  ] for (i, method) in enumerate(methods)]
end

function add_partition_inset!(position, parts, m)
  mask = parts.m .== m
  N = parts.N[findfirst(mask)]
  owner = Matrix(reshape(Int.(pick(parts, :owner, mask)), N, N)')
  multiplicity = Matrix(reshape(Int.(pick(parts, :mult, mask)), N, N)')
  axis = Axis(position; width=Relative(0.28), height=Relative(0.28),
              halign=:right, valign=:top, tellwidth=false, tellheight=false,
              aspect=DataAspect())
  hidedecorations!(axis)
  hidespines!(axis)
  centers = range(1 / (2N), 1 - 1 / (2N); length=N)
  heatmap!(axis, centers, centers, float.(owner); colormap=:Spectral_9)
  heatmap!(axis, centers, centers,
    [RGBAf(0, 0, 0, multiplicity[i, j] > 1 ? 0.12f0 * multiplicity[i, j] : 0)
     for i in axes(multiplicity, 1), j in axes(multiplicity, 2)])
  return axis
end

function add_square_solution_inset!(position, solutions; values=:value, selector=nothing)
  mask = isnothing(selector) ? trues(length(solutions.N)) : selector
  N = solutions.N[findfirst(mask)]
  field = zeros(N + 1, N + 1)
  field[2:N, 2:N] .= Matrix(reshape(pick(solutions, values, mask), N - 1, N - 1)')
  axis = Axis(position; width=Relative(0.28), height=Relative(0.28),
              halign=:left, valign=:top, tellwidth=false, tellheight=false,
              aspect=DataAspect())
  hidedecorations!(axis)
  hidespines!(axis)
  heatmap!(axis, range(0, 1; length=N + 1), range(0, 1; length=N + 1),
           field; colormap=:viridis)
  return axis
end

function fig12b_poisson_cmp_paper()
  problems = (
    ("Poisson", loadtable("study8_linear_cmp_poisson.csv")),
    ("variable diffusion", loadtable("study8_linear_cmp_sign_changing.csv")),
  )
  parts = loadtable("study8_partitions.csv")
  ms = sort(unique(problems[1][2].m))
  methods = ("var_dd_additive", "var_dd_additive_history", "ras", "pcg_as", "gmres_ras")
  labels = ("EMDD (q = 1)", "EMDD (q = 2)", "RAS", "CG+AS", "GMRES+RAS")
  markers = Dict(zip(methods, (:circle, :hexagon, :utriangle, :diamond, :pentagon)))
  colors = tab10_colors(length(methods))
  figure = Figure(size=(PAPER_WIDTH, 500))
  for (row, (problem, table)) in enumerate(problems), (column, m) in enumerate(ms)
    axis = Axis(figure[row, column];
      xlabel=row == 2 ? "iteration" : "",
      ylabel=column == 1 ? "$problem\nrelative residual" : "",
      title=row == 1 ? "m = $m" : "", yscale=log10,
      yticks=LogTicks(collect(1:-2:-11)), limits=((0, 100), (1e-11, 5)))
    for (index, method) in enumerate(methods)
      mask = (table.m .== m) .& (table.method .== method) .&
             (table.relative_residual .> 0)
      add_series!(axis, pick(table, :solves, mask) ./ m,
                  pick(table, :relative_residual, mask);
                  color=colors[index], marker=markers[method])
    end
    add_partition_inset!(figure[row, column], parts, m)
  end
  Legend(figure[3, 1:length(ms)], legend_elements(methods, markers, colors), collect(labels);
         orientation=:horizontal, nbanks=1)
  rowgap!(figure.layout, 8)
  return savefigs(figure, "fig12b_poisson_cmp_paper")
end

function fig13_evp_cmp()
  table = loadtable("study9_evp_cmp.csv")
  solutions = loadtable("study9_evp_solution.csv")
  parts = loadtable("study9_partitions.csv")
  ms = sort(unique(table.m))
  methods = ("var_dd", "var_dd_history", "lopsd_as", "lobpcg_as",
             "jd_gmres_as_1", "jd_gmres_as_2", "jd_gmres_as_4")
  labels = ("EMDD (q = 1)", "EMDD (q = 2)", "LOPSD+AS", "LOBPCG+AS",
            "JD-GMRES(AS, 1)", "JD-GMRES(AS, 2)", "JD-GMRES(AS, 4)")
  markers = Dict(zip(methods, (:circle, :hexagon, :rect, :utriangle,
                               :diamond, :pentagon, :dtriangle)))
  colors = [tab10_colors(4)..., RGBf(0.55, 0.55, 0.55),
            RGBf(0.38, 0.38, 0.38), RGBf(0.20, 0.20, 0.20)]
  styles = [fill(:solid, 4)..., (:dot, :dense), (:dash, :dense), (:dashdot, :dense)]
  widths = [fill(2.8, 4)..., fill(1.8, 3)...]
  figure = Figure(size=(PAPER_WIDTH, 350))
  for (column, m) in enumerate(ms)
    axis = Axis(figure[1, column]; xlabel="outer iteration",
      ylabel=column == 1 ? "relative residual" : "", title="m = $m",
      yscale=log10, limits=((0, 50), (5e-7, 2)))
    for (index, method) in enumerate(methods)
      mask = (table.m .== m) .& (table.method .== method) .&
             (table.relative_residual .> 0)
      add_series!(axis, pick(table, :iteration, mask),
                  pick(table, :relative_residual, mask);
                  color=colors[index], marker=markers[method],
                  linestyle=styles[index], linewidth=widths[index])
    end
    add_square_solution_inset!(figure[1, column], solutions)
    add_partition_inset!(figure[1, column], parts, m)
  end
  Legend(figure[2, 1:length(ms)],
    legend_elements(methods, markers, colors; linestyles=styles, linewidths=widths),
    collect(labels); orientation=:horizontal, nbanks=2)
  return savefigs(figure, "fig13_evp_cmp")
end

function fig14b_gp_convergence_paper()
  table = loadtable("study10_gp_conv.csv")
  solutions = loadtable("study10_gp_solutions.csv")
  parts = loadtable("study10_partitions.csv")
  ms = sort(unique(table.m))
  methods = ("gp_additive", "gp_additive_history", "gp_projected_qemdd",
    "gp_projected_qemdd_history", "gfdn_pcg_as_1", "gfdn_pcg_as_2",
    "gfdn_pcg_as_4", "cg_gfdn_pcg_as_1", "cg_gfdn_pcg_as_2",
    "cg_gfdn_pcg_as_4")
  labels = ("EMDD (q = 1)", "EMDD (q = 2)", "projected qEMDD (q = 1)",
    "projected qEMDD (q = 2)", "GFDN-PCG(AS, 1)", "GFDN-PCG(AS, 2)",
    "GFDN-PCG(AS, 4)", "CG-GFDN-PCG(AS, 1)", "CG-GFDN-PCG(AS, 2)",
    "CG-GFDN-PCG(AS, 4)")
  markers = Dict(zip(methods, (:circle, :hexagon, :utriangle, :cross,
    :circle, :rect, :dtriangle, :circle, :rect, :dtriangle)))
  colors = [tab10_colors(4)..., fill(RGBf(0.42, 0.42, 0.42), 6)...]
  styles = [fill(:solid, 4)..., fill((:dash, :dense), 3)...,
            fill((:dot, :dense), 3)...]
  widths = [fill(2.8, 4)..., fill(1.8, 6)...]
  figure = Figure(size=(PAPER_WIDTH, 390))
  for (column, m) in enumerate(ms)
    axis = Axis(figure[1, column]; xlabel="outer iteration",
      ylabel=column == 1 ? "relative residual" : "", title="m = $m",
      yscale=log10, limits=((0, 30), (1e-7, 2)))
    for (index, method) in enumerate(methods)
      mask = (table.m .== m) .& (table.method .== method) .& (table.resnorm .> 0)
      add_series!(axis, pick(table, :iteration, mask), pick(table, :resnorm, mask);
                  color=colors[index], marker=markers[method],
                  linestyle=styles[index], linewidth=widths[index], markersize=8)
    end
    add_square_solution_inset!(figure[1, column], solutions; values=:density)
    add_partition_inset!(figure[1, column], parts, m)
  end
  Legend(figure[2, 1:length(ms)],
    legend_elements(methods, markers, colors; linestyles=styles,
                    linewidths=widths, markersizes=fill(8, length(methods))),
    collect(labels); orientation=:horizontal, nbanks=2, labelsize=13)
  return savefigs(figure, "fig14b_gp_convergence_paper")
end

function semilinear_figure(data_file, output_name; case_column=nothing)
  table = loadtable(data_file)
  ms = sort(unique(table.m))
  row_values = isnothing(case_column) ? sort(unique(table.beta)) : unique(table.case)
  methods = ("var_dd", "var_dd_history", "var_dd_quadratic",
    "var_dd_quadratic_history", "nonlinear_cg_optim_as", "anderson_ras",
    "newton_pcg_as_1", "newton_pcg_as_2", "newton_pcg_as_4")
  labels = ("EMDD (q = 1)", "EMDD (q = 2)", "qEMDD (q = 1)",
    "qEMDD (q = 2)", "AS-NCG", "Anderson-RAS (m = 4)",
    "Newton-PCG(AS, 1)", "Newton-PCG(AS, 2)", "Newton-PCG(AS, 4)")
  markers = Dict(zip(methods, (:circle, :hexagon, :rect, :utriangle,
    :diamond, :cross, :circle, :rect, :dtriangle)))
  colors = [tab10_colors(5)..., RGBf(0.78, 0.57, 0.02),
            RGBf(0.55, 0.55, 0.55), RGBf(0.45, 0.45, 0.45),
            RGBf(0.35, 0.35, 0.35)]
  styles = [fill(:solid, 5)..., fill((:dash, :dense), 4)...]
  widths = [fill(2.8, 5)..., fill(1.8, 4)...]
  figure = Figure(size=(PAPER_WIDTH, isnothing(case_column) ? 600 : 390))
  for (row, value) in enumerate(row_values), (column, m) in enumerate(ms)
    rowmask = isnothing(case_column) ? table.beta .== value : table.case .== value
    axis = Axis(figure[row, column];
      xlabel=row == length(row_values) ? "outer iteration" : "",
      ylabel=column == 1 ? (isnothing(case_column) ?
        "β = $(Int(value))\nrelative residual" : "relative residual") : "",
      title=row == 1 ? "m = $m" : "", yscale=log10,
      limits=(nothing, (1e-8, 2)))
    for (index, method) in enumerate(methods)
      mask = rowmask .& (table.m .== m) .& (table.method .== method) .&
             (table.relative_residual .> 0)
      add_series!(axis, pick(table, :outer, mask),
                  pick(table, :relative_residual, mask);
                  color=colors[index], marker=markers[method],
                  linestyle=styles[index], linewidth=widths[index])
    end
    hlines!(axis, [1e-7]; color=:black, linestyle=:dot, linewidth=1.2)
  end
  legend_row = length(row_values) + 1
  Legend(figure[legend_row, 1:length(ms)],
    legend_elements(methods, markers, colors; linestyles=styles, linewidths=widths),
    collect(labels); orientation=:horizontal, nbanks=2, labelsize=13)
  return savefigs(figure, output_name)
end

fig17b_semilinear_poisson_paper() = semilinear_figure(
  "study11_semilinear_conv.csv", "fig17b_semilinear_poisson_paper"
)
fig17b_semilinear_l_shape_section61() = semilinear_figure(
  "study11b_semilinear_l_shape_conv.csv",
  "fig17b_semilinear_l_shape_section61"; case_column=:case,
)

function make_all_figures()
  println("plots: generating the final-paper figures")
  fig12b_poisson_cmp_paper()
  fig13_evp_cmp()
  fig14b_gp_convergence_paper()
  fig17b_semilinear_poisson_paper()
  fig17b_semilinear_l_shape_section61()
  println("plots: done -> $FIG_DIR")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && make_all_figures()
