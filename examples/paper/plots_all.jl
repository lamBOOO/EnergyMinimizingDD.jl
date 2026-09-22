# Regenerates ALL Section 4 figures from data/*.csv (no solves).
# Run: julia --project=. examples/paper/plots_all.jl

isdefined(Main, :PAPER_COMMON) || include("common.jl")

using CairoMakie

set_theme!(
  Theme(
    fontsize = 18,
    # font = "Latin Modern Roman",
    Axis = (
      xticklabelsize = 16,
      yticklabelsize = 16,
      xlabelsize = 20,
      ylabelsize = 20,
      titlesize = 20,
      # titlefont = "Latin Modern Roman",
      xgridvisible = true,
      ygridvisible = true,
      xgridcolor = (:black, 0.12),
      ygridcolor = (:black, 0.12),
      spinewidth = 1.2,
    ),
    Axis3 = (
      titlesize = 14,
      # titlefont = "Latin Modern Roman",
      xticklabelsize = 10,
      yticklabelsize = 10,
      zticklabelsize = 10,
    ),
    Lines = (linewidth = 2.5,),
    Scatter = (markersize = 8, strokewidth = 0),
    Legend = (labelsize = 16, framevisible = true),
  ),
)

const PALETTE = Makie.wong_colors()
const MARKERSIZE = 10

"""
    tab10_colors(n)

First `n` entries of the categorical tab10 palette, cycling if `n > 10`. Taking
a fixed prefix keeps a curve's color independent of how many curves a figure
draws; `resample_cmap` would instead interpolate across the whole colormap and
produce both muddy blends and a palette that shifts with `n`.
"""
tab10_colors(n) = [Makie.to_colormap(:tab10)[mod1(i, 10)] for i = 1:n]
const PAPER_FULL_WIDTH = 1000
const PAPER_HALF_WIDTH = PAPER_FULL_WIDTH ÷ 2

"Save a figure as PDF (vector, for LaTeX) and PNG (preview) into figures/."
function savefigs(fig, name)
  save(joinpath(FIG_DIR, name * ".pdf"), fig)
  save(joinpath(FIG_DIR, name * ".png"), fig; px_per_unit = 2)
  println("  saved $(name).{pdf,png}")
  return fig
end

# Select entries of column `col` where `mask` holds (columntable helper)
pick(tbl, col, mask) = collect(getproperty(tbl, col)[mask])

field_matrix(u, N) = Matrix(reshape(u, N - 1, N - 1)')
all_nodes(N) = range(0, 1; length = N + 1)

function field_matrix_with_bc(u, N)
  Z = zeros(N + 1, N + 1)
  Z[2:N, 2:N] .= field_matrix(u, N)
  return Z
end

logfloor(v; floor = 1e-16) = max.(v, floor)

function add_series!(
  ax,
  x,
  y;
  label = nothing,
  color = nothing,
  linestyle = :solid,
  marker = :circle,
  marker_stride = 1,
  markersize = MARKERSIZE,
  markerstrokecolor = :transparent,
  markerstrokewidth = 0,
  linewidth = 2.5,
)
  c = isnothing(color) ? PALETTE[1] : color
  lines!(
    ax,
    x,
    y;
    label = label,
    color = c,
    linestyle = linestyle,
    linewidth = linewidth,
  )
  marker_stride >= 1 || throw(ArgumentError("marker_stride must be positive"))
  marker_indices = collect(1:marker_stride:length(x))
  !isempty(x) &&
    last(marker_indices) != length(x) &&
    push!(marker_indices, length(x))
  scatter!(
    ax,
    x[marker_indices],
    y[marker_indices];
    color = c,
    marker = marker,
    markersize = markersize,
    strokecolor = markerstrokecolor,
    strokewidth = markerstrokewidth,
  )
end

function add_legend!(ax; position = :rt)
  axislegend(ax; position = position, framevisible = true)
end

function color_for(i)
  PALETTE[mod1(i, length(PALETTE))]
end

function add_partition_boundaries!(
  ax,
  owner_grid;
  color = (:black, 0.55),
  linewidth = 0.45,
)
  n = size(owner_grid, 1)
  xs = Float64[]
  ys = Float64[]
  function add_segment!(x1, y1, x2, y2)
    append!(xs, (x1, x2, NaN))
    append!(ys, (y1, y2, NaN))
  end
  for i = 1:(n-1), j = 1:n
    owner_grid[i, j] == owner_grid[i+1, j] && continue
    x = i / n
    add_segment!(x, (j - 1) / n, x, j / n)
  end
  for i = 1:n, j = 1:(n-1)
    owner_grid[i, j] == owner_grid[i, j+1] && continue
    y = j / n
    add_segment!((i - 1) / n, y, i / n, y)
  end
  add_segment!(0, 0, 1, 0)
  add_segment!(1, 0, 1, 1)
  add_segment!(1, 1, 0, 1)
  add_segment!(0, 1, 0, 0)
  lines!(ax, xs, ys; color = color, linewidth = linewidth)
end

function legend_line_marker_elements(
  methods,
  markers;
  colors = [color_for(i) for i in eachindex(methods)],
  linestyles = fill(:solid, length(methods)),
  linewidths = fill(2.5, length(methods)),
  markersizes = fill(MARKERSIZE, length(methods)),
  markerstrokecolors = fill(:transparent, length(methods)),
  markerstrokewidths = fill(0, length(methods)),
)
  return [
    [
      LineElement(
        color = colors[i],
        linestyle = linestyles[i],
        linewidth = linewidths[i],
      ),
      MarkerElement(
        color = colors[i],
        marker = markers[method],
        markersize = markersizes[i],
        strokecolor = markerstrokecolors[i],
        strokewidth = markerstrokewidths[i],
      ),
    ] for (i, method) in enumerate(methods)
  ]
end

function add_partition_inset!(
  figpos,
  parts,
  m;
  halign = 1.03,
  valign = 0.97,
  inset_size = 0.3,
  inset_title = "",
)
  pmask = parts.m .== m
  N = parts.N[findfirst(pmask)]
  pax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign = halign,
    valign = valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
    xticksvisible = false,
    yticksvisible = false,
    xticklabelsvisible = false,
    yticklabelsvisible = false,
    xlabelvisible = false,
    ylabelvisible = false,
  )
  translate!(pax.blockscene, 0, 0, 150)
  hidedecorations!(pax)
  hidespines!(pax)

  owner_grid = Matrix(reshape(Int.(pick(parts, :owner, pmask)), N, N)')
  mult_grid = Matrix(reshape(Int.(pick(parts, :mult, pmask)), N, N)')
  cell_centers = collect(range(1 / (2N), 1 - 1 / (2N); length = N))
  heatmap!(
    pax,
    cell_centers,
    cell_centers,
    float.(owner_grid);
    colormap = :Spectral_9,
  )
  heatmap!(
    pax,
    cell_centers,
    cell_centers,
    [
      RGBAf(0, 0, 0, mult_grid[i, j] > 1 ? 0.1f0 * mult_grid[i, j] : 0.0f0) for
      i in axes(mult_grid, 1), j in axes(mult_grid, 2)
    ];
  )
  add_partition_boundaries!(pax, owner_grid)
  if !isempty(inset_title)
    text!(
      pax,
      0.5,
      0.96;
      text = inset_title,
      align = (:center, :top),
      fontsize = 9,
      color = :white,
      font = :bold,
    )
  end
  return pax
end

function add_gp_solution_inset!(
  figpos,
  solutions,
  beta;
  halign = 0.68,
  valign = 0.97,
  inset_size = 0.23,
  inset_title = "density",
)
  smask = solutions.beta .== beta
  N = solutions.N[findfirst(smask)]
  sax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign = halign,
    valign = valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  translate!(sax.blockscene, 0, 0, 150)
  hidedecorations!(sax)
  hidespines!(sax)
  xs = collect(all_nodes(N))
  heatmap!(
    sax,
    xs,
    xs,
    field_matrix_with_bc(pick(solutions, :density, smask), N);
    colormap = :viridis,
    colorrange = (0, maximum(pick(solutions, :density, smask))),
  )
  text!(
    sax,
    0.5,
    0.96;
    text = inset_title,
    align = (:center, :top),
    fontsize = 9,
    color = :white,
    font = :bold,
  )
  return sax
end

const SEMILINEAR_INSET_SIZE = 40
const SEMILINEAR_INSET_GAP = 2
const SEMILINEAR_INSET_MARGIN = 2

function add_semilinear_solution_inset!(
  figpos,
  solutions;
  halign = 0.68,
  valign = 0.97,
  inset_size = 0.23,
  inset_title = "exact u★",
)
  N = solutions.N[1]
  values = solutions.value
  sax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign = halign,
    valign = valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  translate!(sax.blockscene, 0, 0, 150)
  hidedecorations!(sax)
  hidespines!(sax)
  nodal_values = field_matrix_with_bc(values, N)
  vertices = Point2f[]
  vertex_values = Float64[]
  for iy = 0:N, ix = 0:N
    push!(vertices, Point2f(ix / N, iy / N))
    push!(vertex_values, nodal_values[iy+1, ix+1])
  end
  node(ix, iy) = iy * (N + 1) + ix + 1
  TriangleFace = CairoMakie.GeometryBasics.TriangleFace
  faces = TriangleFace{Int}[]
  face_values = Float64[]
  for iy = 0:(N-1), ix = 0:(N-1)
    lower_left = node(ix, iy)
    lower_right = node(ix + 1, iy)
    upper_left = node(ix, iy + 1)
    upper_right = node(ix + 1, iy + 1)
    push!(faces, TriangleFace(lower_left, lower_right, upper_left))
    push!(
      face_values,
      (vertex_values[lower_left] + vertex_values[lower_right] + vertex_values[upper_left]) /
      3,
    )
    push!(faces, TriangleFace(lower_right, upper_right, upper_left))
    push!(
      face_values,
      (vertex_values[lower_right] + vertex_values[upper_right] + vertex_values[upper_left]) /
      3,
    )
  end
  color_limit = maximum(abs, values)
  polygons = [
    Point2f[vertices[face[1]], vertices[face[2]], vertices[face[3]]] for
    face in faces
  ]
  poly!(
    sax,
    polygons;
    # Per-face colors avoid Cairo's PDF Gouraud-shading primitive, which can be
    # dropped by TeX/PDF post-processing while leaving the wireframe visible.
    color = face_values,
    colormap = :balance,
    colorrange = (-color_limit, color_limit),
    strokewidth = 0,
  )
  wireframe!(
    sax,
    CairoMakie.GeometryBasics.Mesh(vertices, faces);
    color = (:black, 0.14),
    linewidth = 0.18,
  )
  if !isnothing(inset_title)
    text!(
      sax,
      0.5,
      0.96;
      text = inset_title,
      align = (:center, :top),
      fontsize = 9,
      color = :white,
      font = :bold,
    )
  end
  return sax
end

"Fixed-size, top-right layout shared by the two semilinear paper insets."
function semilinear_inset_layout(
  figpos;
  inset_size = SEMILINEAR_INSET_SIZE,
  gap = SEMILINEAR_INSET_GAP,
  margin = SEMILINEAR_INSET_MARGIN,
  columns = 2,
)
  layout = GridLayout(
    figpos;
    width = columns * inset_size + (columns - 1) * gap,
    height = inset_size,
    halign = :right,
    valign = :top,
    tellwidth = false,
    tellheight = false,
    alignmode = Outside(margin),
    default_colgap = gap,
  )
  return layout
end

function fix_semilinear_inset_sizes!(
  layout;
  inset_size = SEMILINEAR_INSET_SIZE,
  gap = SEMILINEAR_INSET_GAP,
  columns = 2,
)
  for column = 1:columns
    colsize!(layout, column, Fixed(inset_size))
  end
  rowsize!(layout, 1, Fixed(inset_size))
  for column = 1:(columns-1)
    colgap!(layout, column, Fixed(gap))
  end
  return layout
end

function add_evp_solution_inset!(
  figpos,
  solutions;
  halign = 0.68,
  valign = 0.97,
  inset_size = 0.23,
  inset_title = "ground state",
)
  N = solutions.N[1]
  values = solutions.value
  sax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign,
    valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  translate!(sax.blockscene, 0, 0, 150)
  hidedecorations!(sax)
  hidespines!(sax)
  heatmap!(
    sax,
    collect(all_nodes(N)),
    collect(all_nodes(N)),
    field_matrix_with_bc(values, N);
    colormap = :viridis,
    colorrange = (0, maximum(values)),
  )
  text!(
    sax,
    0.5,
    0.96;
    text = inset_title,
    align = (:center, :top),
    fontsize = 9,
    color = :white,
    font = :bold,
  )
  return sax
end

function add_lshape_solution_inset!(
  figpos,
  solutions;
  halign = 0.68,
  valign = 0.97,
  inset_size = 0.23,
  inset_title = "exact u★",
)
  sax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign = halign,
    valign = valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = (-1, 1, -1, 1),
  )
  translate!(sax.blockscene, 0, 0, 150)
  hidedecorations!(sax)
  hidespines!(sax)
  vertices = Point2f[]
  vertex_values = Float64[]
  TriangleFace = CairoMakie.GeometryBasics.TriangleFace
  faces = TriangleFace{Int}[]
  face_values = Float64[]
  for index in eachindex(solutions.idx)
    first_vertex = length(vertices) + 1
    append!(
      vertices,
      Point2f[
        (solutions.x1[index], solutions.y1[index]),
        (solutions.x2[index], solutions.y2[index]),
        (solutions.x3[index], solutions.y3[index]),
      ],
    )
    append!(
      vertex_values,
      [
        solutions.value1[index],
        solutions.value2[index],
        solutions.value3[index],
      ],
    )
    push!(faces, TriangleFace(first_vertex, first_vertex + 1, first_vertex + 2))
    push!(
      face_values,
      (solutions.value1[index] + solutions.value2[index] + solutions.value3[index]) /
      3,
    )
  end
  color_limit = maximum(abs, vertex_values)
  polygons = [
    Point2f[vertices[face[1]], vertices[face[2]], vertices[face[3]]] for
    face in faces
  ]
  poly!(
    sax,
    polygons;
    # Per-face colors remain ordinary vector fills in exported PDFs. In
    # contrast, per-vertex colors become a Gouraud mesh that some TeX/PDF
    # pipelines discard while retaining the wireframe drawn below.
    color = face_values,
    colormap = :balance,
    colorrange = (-color_limit, color_limit),
    strokewidth = 0,
  )
  wireframe!(
    sax,
    CairoMakie.GeometryBasics.Mesh(vertices, faces);
    color = (:black, 0.14),
    linewidth = 0.18,
  )
  if !isnothing(inset_title)
    text!(
      sax,
      0.0,
      0.92;
      text = inset_title,
      align = (:center, :top),
      fontsize = 9,
      color = :black,
      font = :bold,
    )
  end
  return sax
end

function add_triangle_partition_inset!(
  figpos,
  parts,
  m;
  halign = 0.99,
  valign = 0.97,
  inset_size = 0.23,
  inset_title = "partition",
  limits = (0, 1, 0, 1),
  title_color = :white,
  title_position = (0.5, 0.96),
)
  pmask = parts.m .== m
  pax = Axis(
    figpos;
    width = Relative(inset_size),
    height = Relative(inset_size),
    halign = halign,
    valign = valign,
    tellwidth = false,
    tellheight = false,
    aspect = DataAspect(),
    limits = limits,
  )
  translate!(pax.blockscene, 0, 0, 150)
  hidedecorations!(pax)
  hidespines!(pax)

  polygons = [
    Point2f[
      (parts.x1[i], parts.y1[i]),
      (parts.x2[i], parts.y2[i]),
      (parts.x3[i], parts.y3[i]),
    ] for i in eachindex(parts.m) if pmask[i]
  ]
  owners = Float64.(pick(parts, :owner, pmask))
  multiplicities = Int.(pick(parts, :mult, pmask))
  poly!(
    pax,
    polygons;
    color = owners,
    colormap = :Spectral_9,
    colorrange = (1, m),
    strokecolor = (:black, 0.18),
    strokewidth = 0.2,
  )
  poly!(
    pax,
    polygons;
    color = [
      RGBAf(0, 0, 0, value > 1 ? 0.1f0 * value : 0.0f0) for
      value in multiplicities
    ],
    strokewidth = 0,
  )
  if !isnothing(inset_title)
    text!(
      pax,
      title_position[1],
      title_position[2];
      text = inset_title,
      align = (:center, :top),
      fontsize = 9,
      color = title_color,
      font = :bold,
    )
  end
  return pax
end

# ---------------------------------------------------------------------------
# Fig 1: problem setup illustration (partition, overlap, solutions)
# ---------------------------------------------------------------------------
function fig12b_poisson_cmp_paper()
  problem_rows = (
    ("Poisson", loadtable("study8_linear_cmp_poisson.csv")),
    ("variable diffusion", loadtable("study8_linear_cmp_sign_changing.csv")),
  )
  parts = loadtable("study8_partitions.csv")
  ms = sort(unique(problem_rows[end][2].m))
  methods =
    ("var_dd_additive", "var_dd_additive_history", "ras", "pcg_as", "gmres_ras")
  labels = Dict(
    "var_dd_additive" => "EMDD (q = 1)",
    "var_dd_additive_history" => "EMDD (q = 2)",
    "ras" => "RAS",
    "pcg_as" => "CG+AS",
    "gmres_ras" => "GMRES+RAS",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "var_dd_additive_history" => :hexagon,
    "ras" => :utriangle,
    "pcg_as" => :diamond,
    "gmres_ras" => :pentagon,
  )
  colors = tab10_colors(length(methods))
  finite_residuals = vcat(
    [
      tbl.relative_residual[tbl.relative_residual .> 0] for
      (_, tbl) in problem_rows
    ]...,
  )
  ylims = (1e-11, maximum(finite_residuals) * 3)
  yticks = LogTicks(collect(1:-2:-11))
  fig = Figure(size = (PAPER_FULL_WIDTH, 500))
  for (row, (problem_label, tbl)) in enumerate(problem_rows)
    for (column, m) in enumerate(ms)
      ax = Axis(
        fig[row, column];
        xlabel = row == length(problem_rows) ? "iteration" : "",
        ylabel = column == 1 ?
                 "$problem_label\nrelative residual" : "",
        ylabelsize = 13,
        yscale = log10,
        yticks,
        title = row == 1 ? "m = $m" : "",
        limits = ((0, 100), ylims),
      )
      for (index, method) in enumerate(methods)
        mask =
          (tbl.m .== m) .& (tbl.method .== method) .&
          (tbl.relative_residual .> 0)
        add_series!(
          ax,
          pick(tbl, :solves, mask) ./ m,
          pick(tbl, :relative_residual, mask);
          label = labels[method],
          color = colors[index],
          marker = markers[method],
          linewidth = 2.8,
          marker_stride = 8,
          markerstrokecolor = :black,
          markerstrokewidth = 0.7,
        )
      end
      inset_layout = semilinear_inset_layout(fig[row, column]; columns = 1)
      add_partition_inset!(
        inset_layout[1, 1],
        parts,
        m;
        halign = :center,
        valign = :center,
        inset_size = 1.0,
        inset_title = "",
      )
      fix_semilinear_inset_sizes!(inset_layout; columns = 1)
    end
  end
  Legend(
    fig[length(problem_rows)+1, 1:length(ms)],
    legend_line_marker_elements(
      methods,
      markers;
      colors,
      linewidths = fill(2.8, length(methods)),
      markerstrokecolors = fill(:black, length(methods)),
      markerstrokewidths = fill(0.7, length(methods)),
    ),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig12b_poisson_cmp_paper")
end
function fig13_evp_cmp()
  tbl = loadtable("study9_evp_cmp.csv")
  solutions = loadtable("study9_evp_solution.csv")
  parts = loadtable("study9_partitions.csv")
  ms = sort(unique(tbl.m))
  primary_methods = ("var_dd", "var_dd_history", "lopsd_as", "lobpcg_as")
  jd_methods = ("jd_gmres_as_1", "jd_gmres_as_2", "jd_gmres_as_4")
  methods = (primary_methods..., jd_methods...)
  labels = Dict(
    "var_dd" => "EMDD (q = 1)",
    "var_dd_history" => "EMDD (q = 2)",
    "lopsd_as" => "LOPSD+AS",
    "lobpcg_as" => "LOBPCG+AS",
    "jd_gmres_as_1" => "JD-GMRES(AS, 1)",
    "jd_gmres_as_2" => "JD-GMRES(AS, 2)",
    "jd_gmres_as_4" => "JD-GMRES(AS, 4)",
  )
  markers = Dict(
    "var_dd" => :circle,
    "var_dd_history" => :hexagon,
    "lopsd_as" => :rect,
    "lobpcg_as" => :utriangle,
    "jd_gmres_as_1" => :diamond,
    "jd_gmres_as_2" => :pentagon,
    "jd_gmres_as_4" => :diamond,
  )
  primary_colors = tab10_colors(length(primary_methods))
  jd_colors = [RGBf(level, level, level) for level in (0.55, 0.38, 0.20)]
  jd_linestyles = [(:dot, :dense), (:dash, :dense), (:dashdot, :dense)]
  colors = [primary_colors..., jd_colors...]
  linestyles = [fill(:solid, length(primary_methods))..., jd_linestyles...]
  linewidths =
    [fill(2.8, length(primary_methods))..., fill(1.8, length(jd_methods))...]
  markersizes = [
    fill(MARKERSIZE, length(primary_methods))...,
    fill(MARKERSIZE, length(jd_methods))...,
  ]
  finite_res = tbl.relative_residual[tbl.relative_residual .> 0]
  ylims = (1e-6 * 0.5, maximum(finite_res) * 1.5)
  ytick_exps =
    sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size=(PAPER_FULL_WIDTH, 350))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "outer iteration",
      ylabel = j == 1 ? "relative residual" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = ((0, 50), ylims),
    )
    for (i, method) in enumerate(methods)
      mask =
        (tbl.m .== m) .& (tbl.method .== method) .& (tbl.relative_residual .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        logfloor(pick(tbl, :relative_residual, mask));
        label = labels[method],
        color = colors[i],
        linestyle = linestyles[i],
        marker = markers[method],
        linewidth = linewidths[i],
        markersize = markersizes[i],
        marker_stride = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 0.7,
      )
    end
    inset_layout = semilinear_inset_layout(fig[1, j])
    add_evp_solution_inset!(
      inset_layout[1, 1],
      solutions;
      halign=:center,
      valign=:center,
      inset_size=1.0,
      inset_title="",
    )
    add_partition_inset!(
      inset_layout[1, 2],
      parts,
      m;
      halign=:center,
      valign=:center,
      inset_size=1.0,
      inset_title="",
    )
    fix_semilinear_inset_sizes!(inset_layout)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(
      primary_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(primary_methods)),
      colors = primary_colors,
      linewidths = fill(2.8, length(primary_methods)),
      markerstrokecolors = fill(:black, length(primary_methods)),
      markerstrokewidths = fill(0.7, length(primary_methods)),
    ),
    [labels[method] for method in primary_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  Legend(
    fig[3, 1:length(ms)],
    legend_line_marker_elements(
      jd_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(jd_methods)),
      colors = jd_colors,
      linestyles = jd_linestyles,
      linewidths = fill(1.8, length(jd_methods)),
      markerstrokecolors = fill(:black, length(jd_methods)),
      markerstrokewidths = fill(0.7, length(jd_methods)),
    ),
    [labels[method] for method in jd_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig13_evp_cmp")
end
function fig14b_gp_convergence_paper()
  tbl = loadtable("study10_gp_conv.csv")
  parts = loadtable("study10_partitions.csv")
  solutions = loadtable("study10_gp_solutions.csv")
  beta = maximum(tbl.beta)
  ms = sort(unique(tbl.m))
  primary_methods = (
    "gp_additive",
    "gp_additive_history",
    # "gp_quadratic",
    # "gp_quadratic_history",
    # "gp_tangent_quadratic",
    # "gp_tangent_quadratic_history",
    "gp_projected_qemdd",
    "gp_projected_qemdd_history",
  )
  gfdn_methods = ("gfdn_pcg_as_1", "gfdn_pcg_as_2", "gfdn_pcg_as_4")
  cg_gfdn_methods = ("cg_gfdn_pcg_as_1", "cg_gfdn_pcg_as_2", "cg_gfdn_pcg_as_4")
  metric_methods = (gfdn_methods..., cg_gfdn_methods...)
  metric_legend_methods = (
    gfdn_methods[1],
    cg_gfdn_methods[1],
    gfdn_methods[2],
    cg_gfdn_methods[2],
    gfdn_methods[3],
    cg_gfdn_methods[3],
  )
  methods = (primary_methods..., metric_methods...)
  labels = Dict(
    "gp_additive" => "EMDD (q = 1)",
    "gp_additive_history" => "EMDD (q = 2)",
    # "gp_quadratic" => "quadratic EMDD (q = 1)",
    # "gp_quadratic_history" => "quadratic EMDD (q = 2)",
    # "gp_tangent_quadratic" => "tangent KKT (q = 1)",
    # "gp_tangent_quadratic_history" => "tangent KKT (q = 2)",
    "gp_projected_qemdd" => "projected qEMDD (q = 1)",
    "gp_projected_qemdd_history" => "projected qEMDD (q = 2)",
    "gfdn_pcg_as_1" => "GFDN-PCG(AS, 1)",
    "gfdn_pcg_as_2" => "GFDN-PCG(AS, 2)",
    "gfdn_pcg_as_4" => "GFDN-PCG(AS, 4)",
    "cg_gfdn_pcg_as_1" => "CG-GFDN-PCG(AS, 1)",
    "cg_gfdn_pcg_as_2" => "CG-GFDN-PCG(AS, 2)",
    "cg_gfdn_pcg_as_4" => "CG-GFDN-PCG(AS, 4)",
  )
  markers = Dict(
    "gp_additive" => :circle,
    "gp_additive_history" => :hexagon,
    # "gp_quadratic" => :star4,
    # "gp_quadratic_history" => :star6,
    # "gp_tangent_quadratic" => :pentagon,
    # "gp_tangent_quadratic_history" => :diamond,
    "gp_projected_qemdd" => :utriangle,
    "gp_projected_qemdd_history" => :cross,
    "gfdn_pcg_as_1" => :circle,
    "gfdn_pcg_as_2" => :rect,
    "gfdn_pcg_as_4" => :dtriangle,
    "cg_gfdn_pcg_as_1" => :circle,
    "cg_gfdn_pcg_as_2" => :rect,
    "cg_gfdn_pcg_as_4" => :dtriangle,
  )
  primary_colors = tab10_colors(length(primary_methods))
  gfdn_colors =
    [RGBf(0.25, 0.25, 0.25), RGBf(0.45, 0.45, 0.45), RGBf(0.62, 0.62, 0.62)]
  cg_gfdn_colors = copy(gfdn_colors)
  metric_linestyles = [
    fill((:dash, :dense), length(gfdn_methods))...,
    fill((:dot, :dense), length(cg_gfdn_methods))...,
  ]
  metric_colors = [gfdn_colors..., cg_gfdn_colors...]
  metric_legend_colors = [
    gfdn_colors[1],
    cg_gfdn_colors[1],
    gfdn_colors[2],
    cg_gfdn_colors[2],
    gfdn_colors[3],
    cg_gfdn_colors[3],
  ]
  metric_legend_linestyles = [
    (:dash, :dense),
    (:dot, :dense),
    (:dash, :dense),
    (:dot, :dense),
    (:dash, :dense),
    (:dot, :dense),
  ]
  colors = [primary_colors..., metric_colors...]
  linestyles = [fill(:solid, length(primary_methods))..., metric_linestyles...]
  linewidths = [
    fill(2.8, length(primary_methods))...,
    fill(1.8, length(metric_methods))...,
  ]
  markersizes = [
    fill(MARKERSIZE, length(primary_methods))...,
    fill(7, length(metric_methods))...,
  ]
  mask_beta = tbl.beta .== beta
  positive_residuals = tbl.resnorm[mask_beta .& (tbl.resnorm .> 0)]
  ylimits = (1e-7, maximum(positive_residuals) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))

  fig = Figure(size = (PAPER_FULL_WIDTH, 390))
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel = "outer iteration",
      ylabel = column == 1 ? "relative residual" : "",
      title = "m = $m",
      yscale = log10,
      yticks = LogTicks(ytick_exps),
      limits = ((0, 30), ylimits),
    )
    for (index, method) in enumerate(methods)
      mask =
        mask_beta .& (tbl.m .== m) .& (tbl.method .== method) .&
        (tbl.resnorm .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        logfloor(pick(tbl, :resnorm, mask));
        label = labels[method],
        color = colors[index],
        linestyle = linestyles[index],
        marker = markers[method],
        linewidth = linewidths[index],
        markersize = markersizes[index],
        marker_stride = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 0.7,
      )
    end
    inset_layout = semilinear_inset_layout(fig[1, column])
    add_partition_inset!(
      inset_layout[1, 2],
      parts,
      m;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = "",
    )
    add_gp_solution_inset!(
      inset_layout[1, 1],
      solutions,
      beta;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = "",
    )
    fix_semilinear_inset_sizes!(inset_layout)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(
      primary_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(primary_methods)),
      colors = primary_colors,
      linewidths = fill(2.8, length(primary_methods)),
      markerstrokecolors = fill(:black, length(primary_methods)),
      markerstrokewidths = fill(0.7, length(primary_methods)),
    ),
    [labels[method] for method in primary_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  Legend(
    fig[3, 1:length(ms)],
    legend_line_marker_elements(
      metric_legend_methods,
      markers;
      markersizes = fill(7, length(metric_methods)),
      colors = metric_legend_colors,
      linestyles = metric_legend_linestyles,
      linewidths = fill(1.8, length(metric_methods)),
      markerstrokecolors = fill(:black, length(metric_methods)),
      markerstrokewidths = fill(0.7, length(metric_methods)),
    ),
    [labels[method] for method in metric_legend_methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig14b_gp_convergence_paper")
end
function fig17b_semilinear_poisson_paper()
  conv = loadtable("study11_semilinear_conv.csv")
  solutions = loadtable("study11_semilinear_solution.csv")
  parts = loadtable("study11_semilinear_partitions.csv")
  ms = sort(unique(conv.m))
  betas = filter(!=(10.0), sort(unique(conv.beta)))
  primary_methods = (
    "var_dd",
    "var_dd_history",
    "var_dd_quadratic",
    "var_dd_quadratic_history",
    "nonlinear_cg_optim_as",
  )
  reference_methods =
    ("anderson_ras", "newton_pcg_as_1", "newton_pcg_as_2", "newton_pcg_as_4")
  methods = (primary_methods..., reference_methods...)
  labels = Dict(
    "var_dd" => "EMDD (q = 1)",
    "var_dd_history" => "EMDD (q = 2)",
    "var_dd_quadratic" => "qEMDD (q = 1)",
    "var_dd_quadratic_history" => "qEMDD (q = 2)",
    "anderson_ras" => "Anderson-RAS (m = 4)",
    "nonlinear_cg_optim_as" => "AS-NCG (Hager-Zhang)",
    "newton_pcg_as_1" => "Newton–PCG(AS, 1)",
    "newton_pcg_as_2" => "Newton–PCG(AS, 2)",
    "newton_pcg_as_4" => "Newton–PCG(AS, 4)",
    # "newton_pcg_as_8" => "Newton–PCG(AS, 8)",
  )
  markers = Dict(
    "var_dd" => :circle,
    "var_dd_history" => :hexagon,
    "var_dd_quadratic" => :rect,
    "var_dd_quadratic_history" => :utriangle,
    "anderson_ras" => :cross,
    "nonlinear_cg_optim_as" => :diamond,
    "newton_pcg_as_1" => :circle,
    "newton_pcg_as_2" => :rect,
    "newton_pcg_as_4" => :dtriangle,
    # "newton_pcg_as_8" => :hexagon,
  )
  primary_colors = tab10_colors(length(primary_methods))
  reference_colors = [
    RGBf(0.78, 0.57, 0.02),
    RGBf(0.55, 0.55, 0.55),
    RGBf(0.45, 0.45, 0.45),
    RGBf(0.35, 0.35, 0.35),
  ]
  reference_linestyles = fill((:dash, :dense), length(reference_methods))
  colors = [primary_colors..., reference_colors...]
  linestyles =
    [fill(:solid, length(primary_methods))..., reference_linestyles...]
  linewidths = [
    fill(2.8, length(primary_methods))...,
    fill(1.8, length(reference_methods))...,
  ]
  markersizes = [
    fill(MARKERSIZE, length(primary_methods))...,
    fill(MARKERSIZE, length(reference_methods))...,
  ]
  initial_residuals = conv.relative_residual[(conv.outer .== 0) .& isfinite.(
    conv.relative_residual,
  )]
  ylimits = (1e-8, max(maximum(initial_residuals), 1.0) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))
  fig = Figure(size = (PAPER_FULL_WIDTH, 600))
  for (row, beta) in enumerate(betas), (column, m) in enumerate(ms)
    ax = Axis(
      fig[row, column];
      xlabel = row == length(betas) ? "outer iteration" : "",
      ylabel = column == 1 ? "β = $(Int(beta))\nrelative residual" : "",
      title = row == 1 ? "m = $m" : "",
      yscale = log10,
      yticks = LogTicks(ytick_exps),
      limits = (nothing, ylimits),
    )
    for (index, method) in enumerate(methods)
      mask =
        (conv.beta .== beta) .& (conv.m .== m) .& (conv.method .== method) .&
        (conv.relative_residual .> 0)
      add_series!(
        ax,
        pick(conv, :outer, mask),
        pick(conv, :relative_residual, mask);
        label = labels[method],
        color = colors[index],
        linestyle = linestyles[index],
        marker = markers[method],
        linewidth = linewidths[index],
        markersize = markersizes[index],
        marker_stride = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 0.7,
      )
    end
    hlines!(ax, [1e-7]; color = :black, linestyle = :dot, linewidth = 1.2)
    inset_layout = semilinear_inset_layout(fig[row, column])
    add_triangle_partition_inset!(
      inset_layout[1, 2],
      parts,
      m;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = nothing,
    )
    add_semilinear_solution_inset!(
      inset_layout[1, 1],
      solutions;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = nothing,
    )
    fix_semilinear_inset_sizes!(inset_layout)
  end
  legend_row = length(betas) + 1
  Legend(
    fig[legend_row, 1:length(ms)],
    legend_line_marker_elements(
      primary_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(primary_methods)),
      colors = primary_colors,
      linewidths = fill(2.8, length(primary_methods)),
      markerstrokecolors = fill(:black, length(primary_methods)),
      markerstrokewidths = fill(0.7, length(primary_methods)),
    ),
    [labels[method] for method in primary_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  Legend(
    fig[legend_row+1, 1:length(ms)],
    legend_line_marker_elements(
      reference_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(reference_methods)),
      colors = reference_colors,
      linestyles = reference_linestyles,
      linewidths = fill(1.8, length(reference_methods)),
      markerstrokecolors = fill(:black, length(reference_methods)),
      markerstrokewidths = fill(0.7, length(reference_methods)),
    ),
    [labels[method] for method in reference_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig17b_semilinear_poisson_paper")
end

# Exact Fig. 17b design duplicated for the non-monotone exponential problem
# on the graded L-shaped domain from Spicher--Wihler (2026), Section 6.1.
function fig17b_semilinear_l_shape_section61()
  conv = loadtable("study11b_semilinear_l_shape_conv.csv")
  solutions = loadtable("study11b_semilinear_l_shape_solution.csv")
  parts = loadtable("study11b_semilinear_l_shape_partitions.csv")
  ms = sort(unique(conv.m))
  cases = unique(conv.case)
  primary_methods = (
    "var_dd",
    "var_dd_history",
    "var_dd_quadratic",
    "var_dd_quadratic_history",
    "nonlinear_cg_optim_as",
  )
  reference_methods =
    ("anderson_ras", "newton_pcg_as_1", "newton_pcg_as_2", "newton_pcg_as_4")
  methods = (primary_methods..., reference_methods...)
  labels = Dict(
    "var_dd" => "EMDD (q = 1)",
    "var_dd_history" => "EMDD (q = 2)",
    "var_dd_quadratic" => "qEMDD (q = 1)",
    "var_dd_quadratic_history" => "qEMDD (q = 2)",
    "anderson_ras" => "Anderson-RAS (m = 4)",
    "nonlinear_cg_optim_as" => "AS-NCG (Hager-Zhang)",
    "newton_pcg_as_1" => "Newton-PCG(AS, 1)",
    "newton_pcg_as_2" => "Newton-PCG(AS, 2)",
    "newton_pcg_as_4" => "Newton-PCG(AS, 4)",
  )
  markers = Dict(
    "var_dd" => :circle,
    "var_dd_history" => :hexagon,
    "var_dd_quadratic" => :rect,
    "var_dd_quadratic_history" => :utriangle,
    "anderson_ras" => :cross,
    "nonlinear_cg_optim_as" => :ltriangle,
    "newton_pcg_as_1" => :circle,
    "newton_pcg_as_2" => :rect,
    "newton_pcg_as_4" => :dtriangle,
  )
  primary_colors = tab10_colors(length(primary_methods))
  reference_colors = [
    RGBf(0.78, 0.57, 0.02),
    RGBf(0.55, 0.55, 0.55),
    RGBf(0.45, 0.45, 0.45),
    RGBf(0.35, 0.35, 0.35),
  ]
  reference_linestyles = fill((:dash, :dense), length(reference_methods))
  colors = [primary_colors..., reference_colors...]
  linestyles =
    [fill(:solid, length(primary_methods))..., reference_linestyles...]
  linewidths = [
    fill(2.8, length(primary_methods))...,
    fill(1.8, length(reference_methods))...,
  ]
  markersizes = [
    fill(MARKERSIZE, length(primary_methods))...,
    fill(MARKERSIZE, length(reference_methods))...,
  ]
  initial_residuals = conv.relative_residual[(conv.outer .== 0) .& isfinite.(
    conv.relative_residual,
  )]
  ylimits = (1e-8, max(maximum(initial_residuals), 1.0) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))
  fig = Figure(size = (PAPER_FULL_WIDTH, 390))
  for (row, case) in enumerate(cases), (column, m) in enumerate(ms)
    ax = Axis(
      fig[row, column];
      xlabel = row == length(cases) ? "outer iteration" : "",
      ylabel = column == 1 ? "relative residual" : "",
      title = "m = $m",
      yscale = log10,
      yticks = LogTicks(ytick_exps),
      limits = (nothing, ylimits),
    )
    for (index, method) in enumerate(methods)
      mask =
        (conv.case .== case) .& (conv.m .== m) .& (conv.method .== method) .&
        (conv.relative_residual .> 0)
      add_series!(
        ax,
        pick(conv, :outer, mask),
        pick(conv, :relative_residual, mask);
        label = labels[method],
        color = colors[index],
        linestyle = linestyles[index],
        marker = markers[method],
        linewidth = linewidths[index],
        markersize = markersizes[index],
        marker_stride = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 0.7,
      )
    end
    hlines!(ax, [1e-7]; color = :black, linestyle = :dot, linewidth = 1.2)
    inset_layout = semilinear_inset_layout(fig[row, column])
    add_triangle_partition_inset!(
      inset_layout[1, 2],
      parts,
      m;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = nothing,
      limits = (-1, 1, -1, 1),
      title_color = :black,
      title_position = (0.0, 0.92),
    )
    add_lshape_solution_inset!(
      inset_layout[1, 1],
      solutions;
      halign = :center,
      valign = :center,
      inset_size = 1.0,
      inset_title = nothing,
    )
    fix_semilinear_inset_sizes!(inset_layout)
  end
  legend_row = length(cases) + 1
  Legend(
    fig[legend_row, 1:length(ms)],
    legend_line_marker_elements(
      primary_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(primary_methods)),
      colors = primary_colors,
      linewidths = fill(2.8, length(primary_methods)),
      markerstrokecolors = fill(:black, length(primary_methods)),
      markerstrokewidths = fill(0.7, length(primary_methods)),
    ),
    [labels[method] for method in primary_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  Legend(
    fig[legend_row+1, 1:length(ms)],
    legend_line_marker_elements(
      reference_methods,
      markers;
      markersizes = fill(MARKERSIZE, length(reference_methods)),
      colors = reference_colors,
      linestyles = reference_linestyles,
      linewidths = fill(1.8, length(reference_methods)),
      markerstrokecolors = fill(:black, length(reference_methods)),
      markerstrokewidths = fill(0.7, length(reference_methods)),
    ),
    [labels[method] for method in reference_methods];
    orientation = :horizontal,
    nbanks = 1,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig17b_semilinear_l_shape_section61")
end

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
