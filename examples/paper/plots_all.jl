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
function fig01_setup()
  part = loadtable("study1_partition.csv")
  gs = loadtable("study1_schroedinger.csv")

  fig = Figure(size = (880, 780))

  Npart = part.N[1]
  xs = collect(interior_nodes(Npart))
  nsub = maximum(part.owner)

  ax1 = Axis(
    fig[1, 1];
    title = "METIS partition (owner)",
    xlabel = "x1",
    ylabel = "x2",
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  heatmap!(
    ax1,
    xs,
    xs,
    field_matrix(float.(part.owner), Npart);
    colormap = :tab10,
    colorrange = (0.5, nsub + 0.5),
  )

  ax2 = Axis(
    fig[1, 2];
    title = "overlap multiplicity",
    xlabel = "x1",
    ylabel = "x2",
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  hm2 = heatmap!(
    ax2,
    xs,
    xs,
    field_matrix(float.(part.mult), Npart);
    colormap = :viridis,
    colorrange = (0.5, maximum(part.mult) + 0.5),
  )
  Colorbar(fig[1, 3], hm2; ticks = collect(1:maximum(part.mult)))

  Ngs = gs.N[1]
  ax3 = Axis(
    fig[2, 1:2];
    title = "Schroedinger ground state",
    xlabel = "x1",
    ylabel = "x2",
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  heatmap!(
    ax3,
    collect(all_nodes(Ngs)),
    collect(all_nodes(Ngs)),
    field_matrix_with_bc(gs.value, Ngs);
    colormap = :viridis,
  )


  colgap!(fig.layout, 12)
  rowgap!(fig.layout, 12)
  savefigs(fig, "fig01_setup")
end

# ---------------------------------------------------------------------------
# Fig 2: EVP convergence histories (sweep m, sweep overlap)
# ---------------------------------------------------------------------------
function conv_panel!(figpos, tbl, values, labelfn; title = "")
  ax = Axis(
    figpos;
    xlabel = "iteration",
    ylabel = "eigenvalue error",
    yscale = log10,
    title = title,
  )
  for (i, v) in enumerate(values)
    mask = (tbl.param .== v) .& (tbl.err .> 1e-13)
    add_series!(
      ax,
      pick(tbl, :iter, mask),
      logfloor(pick(tbl, :err, mask));
      label = labelfn(v),
      color = color_for(i),
    )
  end
  add_legend!(ax; position = :rt)
  return ax
end

function fig02_evp_convergence()
  tm = loadtable("study2_sweep_m.csv")
  to = loadtable("study2_sweep_olap.csv")
  fig = Figure(size = (900, 350))
  conv_panel!(
    fig[1, 1],
    tm,
    sort(unique(tm.param)),
    v -> "m = $v";
    title = "overlap = 2",
  )
  conv_panel!(
    fig[1, 2],
    to,
    sort(unique(to.param)),
    v -> "overlap = $v";
    title = "m = 6",
  )
  savefigs(fig, "fig02_evp_convergence")
end

# ---------------------------------------------------------------------------
# Fig 3: EVP vs inverse-iteration baseline
# ---------------------------------------------------------------------------
function fig03_evp_baseline()
  tbl = loadtable("study2_baseline.csv")
  fig = Figure(size = (600, 400))
  ax = Axis(
    fig[1, 1];
    xlabel = "iteration",
    ylabel = "eigenvalue error",
    yscale = log10,
  )
  for (i, (method, label)) in enumerate((
    ("var_dd", "energy-minimizing DD"),
    ("inverse_iteration", "inverse iteration"),
  ))
    mask = (tbl.method .== method) .& (tbl.err .> 1e-13)
    add_series!(
      ax,
      pick(tbl, :iter, mask),
      logfloor(pick(tbl, :err, mask));
      label = label,
      color = color_for(i),
    )
  end
  add_legend!(ax; position = :rt)
  savefigs(fig, "fig03_evp_baseline")
end

# ---------------------------------------------------------------------------
# Fig 4: FEM accuracy of the converged eigenvalue vs h
# ---------------------------------------------------------------------------
function fig04_evp_haccuracy()
  tbl = loadtable("study2_haccuracy.csv")
  hs = collect(tbl.h)
  errs = collect(tbl.err)
  fig = Figure(size = (600, 400))
  ax = Axis(
    fig[1, 1];
    xlabel = "mesh size h",
    ylabel = "eigenvalue error vs exact",
    xscale = log10,
    yscale = log10,
  )
  add_series!(ax, hs, errs; label = "var_dd (converged)", color = color_for(1))
  href = [minimum(hs), maximum(hs)]
  eref = errs[argmin(hs)] .* (href ./ minimum(hs)) .^ 2
  lines!(
    ax,
    href,
    eref;
    label = "quadratic reference",
    color = :black,
    linestyle = :dash,
    linewidth = 1.5,
  )
  add_legend!(ax; position = :lt)
  savefigs(fig, "fig04_evp_haccuracy")
end

# ---------------------------------------------------------------------------
# Fig 5: robustness — iterations vs m and vs overlap
# ---------------------------------------------------------------------------
const PROBLEM_STYLE =
  ("evp" => (:solid, "EVP"), "poisson" => (:dash, "Poisson"))

function fig05_robustness()
  tbl = loadtable("study3_robustness.csv")
  Ns = sort(unique(tbl.N))
  ms = sort(unique(tbl.m))
  N_fix = maximum(Ns) > 40 ? 40 : maximum(Ns)
  fig = Figure(size = (900, 350))

  ax1 = Axis(
    fig[1, 1];
    xlabel = "number of subdomains m",
    ylabel = "iterations",
    title = "overlap = 2",
    xticks = ms,
  )
  for (ci, N) in enumerate(Ns), (problem, (ls, plabel)) in PROBLEM_STYLE
    mask = (tbl.problem .== problem) .& (tbl.N .== N) .& (tbl.overlap .== 2)
    add_series!(
      ax1,
      pick(tbl, :m, mask),
      pick(tbl, :iters, mask);
      label = "$plabel, N = $N",
      color = color_for(ci),
      linestyle = ls,
    )
  end
  add_legend!(ax1; position = :lt)

  olaps = sort(unique(tbl.overlap))
  ax2 = Axis(
    fig[1, 2];
    xlabel = "overlap",
    ylabel = "iterations",
    title = "N = $N_fix",
    xticks = olaps,
  )
  for (ci, m) in enumerate(ms), (problem, (ls, plabel)) in PROBLEM_STYLE
    mask = (tbl.problem .== problem) .& (tbl.N .== N_fix) .& (tbl.m .== m)
    add_series!(
      ax2,
      pick(tbl, :overlap, mask),
      pick(tbl, :iters, mask);
      label = "$plabel, m = $m",
      color = color_for(ci),
      linestyle = ls,
    )
  end
  add_legend!(ax2; position = :rt)
  savefigs(fig, "fig05_robustness")
end

# ---------------------------------------------------------------------------
# Fig 6: timing (overlap = 2 slice)
# ---------------------------------------------------------------------------
function fig06_timing()
  tbl = loadtable("study3_robustness.csv")
  ms = sort(unique(tbl.m))
  Nticks = sort(unique(tbl.N))
  fig = Figure(size = (900, 350))

  ax1 = Axis(
    fig[1, 1];
    xlabel = "mesh parameter N",
    ylabel = "wall-clock time [s]",
    xscale = log10,
    yscale = log10,
    xticks = (Nticks, string.(Nticks)),
    title = "time per solve",
  )
  ax2 = Axis(
    fig[1, 2];
    xlabel = "mesh parameter N",
    ylabel = "time per iteration [s]",
    xscale = log10,
    yscale = log10,
    xticks = (Nticks, string.(Nticks)),
    title = "time per iteration",
  )
  for (ci, m) in enumerate(ms), (problem, (ls, plabel)) in PROBLEM_STYLE
    mask =
      (tbl.problem .== problem) .& (tbl.m .== m) .& (tbl.overlap .== 2) .&
      .!isnan.(tbl.time_s)
    any(mask) || continue
    Ns = pick(tbl, :N, mask)
    ts = pick(tbl, :time_s, mask)
    its = pick(tbl, :iters, mask)
    add_series!(
      ax1,
      Ns,
      ts;
      label = "$plabel, m = $m",
      color = color_for(ci),
      linestyle = ls,
    )
    add_series!(
      ax2,
      Ns,
      ts ./ its;
      label = "$plabel, m = $m",
      color = color_for(ci),
      linestyle = ls,
    )
  end
  add_legend!(ax1; position = :lt)
  add_legend!(ax2; position = :lt)
  savefigs(fig, "fig06_timing")
end

# ---------------------------------------------------------------------------
# Fig 7: Poisson convergence histories
# ---------------------------------------------------------------------------
function fig07_poisson()
  tbl = loadtable("study4_poisson.csv")
  ms = sort(unique(tbl.m))
  fig = Figure(size = (900, 350))
  ax1 =
    Axis(fig[1, 1]; xlabel = "iteration", ylabel = "energy gap", yscale = log10)
  ax2 = Axis(
    fig[1, 2];
    xlabel = "iteration",
    ylabel = "residual norm",
    yscale = log10,
  )
  for (i, m) in enumerate(ms)
    mask_e =
      (tbl.m .== m) .& (tbl.quantity .== "energy_gap") .& (tbl.value .> 1e-14)
    mask_r =
      (tbl.m .== m) .& (tbl.quantity .== "resnorm") .& (tbl.value .> 1e-14)
    add_series!(
      ax1,
      pick(tbl, :iter, mask_e),
      logfloor(pick(tbl, :value, mask_e));
      label = "m = $m",
      color = color_for(i),
    )
    add_series!(
      ax2,
      pick(tbl, :iter, mask_r),
      logfloor(pick(tbl, :value, mask_r));
      label = "m = $m",
      color = color_for(i),
    )
  end
  hlines!(
    ax2,
    [1e-12];
    label = "tolerance",
    color = :black,
    linestyle = :dash,
    linewidth = 1.5,
  )
  add_legend!(ax1; position = :rt)
  add_legend!(ax2; position = :rt)
  savefigs(fig, "fig07_poisson")
end
# ---------------------------------------------------------------------------
# Fig 9: heat equation — warm vs cold start and tau sweep
# ---------------------------------------------------------------------------
function fig09_heat_warmstart()
  wc = loadtable("study6_warmcold.csv")
  ts = loadtable("study6_tau.csv")

  fig = Figure(size = (900, 350))
  ax1 =
    Axis(fig[1, 1]; xlabel = "time step n", ylabel = "DD iterations per step")
  for (i, (mode, label)) in
      enumerate((("cold", "cold start"), ("warm", "warm start")))
    mask = wc.mode .== mode
    add_series!(
      ax1,
      pick(wc, :step, mask),
      pick(wc, :iters, mask);
      label = label,
      color = color_for(i),
    )
  end
  ylims!(ax1, 0, maximum(wc.iters) + 2)
  add_legend!(ax1; position = :rc)

  taus = sort(unique(ts.tau))
  totals = [sum(pick(ts, :iters, ts.tau .== tau)) for tau in taus]
  ax2 = Axis(
    fig[1, 2];
    xlabel = "time step size tau",
    ylabel = "total DD iterations (T = 0.2)",
    xscale = log10,
  )
  add_series!(ax2, taus, totals; color = color_for(1))
  savefigs(fig, "fig09_heat_warmstart")
end

# ---------------------------------------------------------------------------
# Fig 10: heat equation — dissipation and accuracy
# ---------------------------------------------------------------------------
function fig10_heat_dissipation()
  ts = loadtable("study6_tau.csv")
  taus = sort(unique(ts.tau))
  fig = Figure(size = (900, 350))
  ax1 = Axis(
    fig[1, 1];
    xlabel = "t",
    ylabel = "energy gap to steady state",
    yscale = log10,
  )
  ax2 = Axis(
    fig[1, 2];
    xlabel = "t",
    ylabel = "rel. error vs direct solve",
    yscale = log10,
  )
  for (i, tau) in enumerate(taus)
    mask = ts.tau .== tau
    label = @sprintf("tau = %.1e", tau)
    add_series!(
      ax1,
      pick(ts, :t, mask),
      logfloor(pick(ts, :energy_gap, mask));
      label = label,
      color = color_for(i),
      markersize = 6,
    )
    add_series!(
      ax2,
      pick(ts, :t, mask),
      logfloor(pick(ts, :relerr, mask));
      label = label,
      color = color_for(i),
      markersize = 6,
    )
  end
  add_legend!(ax1; position = :rt)
  add_legend!(ax2; position = :rb)
  savefigs(fig, "fig10_heat_dissipation")
end

# ---------------------------------------------------------------------------
# Fig 11: Schroedinger EVP — 3D series of iterate and subdomain updates
# ---------------------------------------------------------------------------
function fig11_local3d()
  tbl = loadtable("study7_local3d.csv")
  N = tbl.N[1]
  xs = collect(all_nodes(N))
  ks = sort(unique(tbl.iter))
  subs = sort(unique(tbl.sub[tbl.kind .== "update"]))
  ncols = 1 + length(subs)

  fig = Figure(size = (330 * ncols, 300 * length(ks)))
  for (ri, k) in enumerate(ks)
    mask_u = (tbl.iter .== k) .& (tbl.kind .== "solution")
    ax = Axis3(
      fig[ri, 1];
      title = "iterate after sweep $k",
      azimuth = 0.7pi,
      elevation = 0.22pi,
      xticksvisible = false,
      yticksvisible = false,
    )
    surface!(
      ax,
      xs,
      xs,
      field_matrix_with_bc(pick(tbl, :value, mask_u), N);
      colormap = :viridis,
    )

    mask_all = (tbl.iter .== k) .& (tbl.kind .== "update")
    zmax = maximum(abs.(tbl.value[mask_all])) + 1e-12
    for (cj, s) in enumerate(subs)
      mask = mask_all .& (tbl.sub .== s)
      axu = Axis3(
        fig[ri, cj+1];
        title = ri == 1 ? "update, subdomain $s" : "",
        azimuth = 0.7pi,
        elevation = 0.22pi,
        xticksvisible = false,
        yticksvisible = false,
      )
      surface!(
        axu,
        xs,
        xs,
        field_matrix_with_bc(pick(tbl, :value, mask), N);
        colormap = :viridis,
        colorrange = (-zmax, zmax),
      )
      zlims!(axu, -zmax, zmax)
    end
  end
  savefigs(fig, "fig11_local3d")
end

# ---------------------------------------------------------------------------
# Fig 12: Poisson -- comparison against one-level Schwarz baselines
# ---------------------------------------------------------------------------
function fig12_poisson_cmp()
  tbl = loadtable("study8_linear_cmp.csv")
  parts = loadtable("study8_partitions.csv")
  ms = sort(unique(tbl.m))
  labels = Dict(
    "var_dd_additive" => "additive varDD",
    "remdd_q1" => "REMDD (q = 1)",
    "remdd_q2" => "REMDD (q = 2)",
    "var_dd_additive_history" => "additive varDD + previous",
    "var_dd_additive_mix_025" => "additive varDD, ω = 0.25",
    "var_dd_additive_mix_05" => "additive varDD, ω = 0.5",
    "var_dd_additive_mix_075" => "additive varDD, ω = 0.75",
    "var_dd_multiplicative" => "multiplicative varDD",
    "as" => "damped AS",
    "ras" => "RAS",
    "pcg_as" => "CG+AS",
    "gmres_ras" => "GMRES+RAS",
  )
  methods = (
    "var_dd_additive",
    "remdd_q1",
    "var_dd_additive_history",
    "remdd_q2",
    "var_dd_additive_mix_025",
    "var_dd_additive_mix_05",
    "var_dd_additive_mix_075",
    "var_dd_multiplicative",
    "as",
    "ras",
    "pcg_as",
    "gmres_ras",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "remdd_q1" => :rtriangle,
    "remdd_q2" => :ltriangle,
    "var_dd_additive_history" => :hexagon,
    "var_dd_additive_mix_025" => :cross,
    "var_dd_additive_mix_05" => :xcross,
    "var_dd_additive_mix_075" => :dtriangle,
    "var_dd_multiplicative" => :star5,
    "as" => :rect,
    "ras" => :utriangle,
    "pcg_as" => :diamond,
    "gmres_ras" => :pentagon,
  )
  colors = tab10_colors(length(methods))
  finite_res = tbl.resnorm[tbl.resnorm .> 0]
  ylims = (1e-10 * 0.5, maximum(finite_res) * 3)
  ytick_exps =
    sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size = (max(430 * length(ms), 1200), 350))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "iteration",
      ylabel = j == 1 ? "res. norm ‖Axₖ-b‖₂" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = ((0, 100), ylims),
    )
    for (i, method) in enumerate(methods)
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.resnorm .> 1e-14)
      add_series!(
        ax,
        pick(tbl, :solves, mask) ./ m,
        logfloor(pick(tbl, :resnorm, mask));
        label = labels[method],
        color = colors[i],
        marker = markers[method],
      )
    end
    add_partition_inset!(fig[1, j], parts, m)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig12_poisson_cmp")
end

# Paper-facing subset of Figure 12 for the standard Poisson problem and the
# sign-changing variable-coefficient manufactured problem.
# Here q counts the current iterate as the first global vector, so q=1 is plain
# additive varDD and q=2 retains the preceding global iterate.
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

# Two-level comparison isolating the effect of the Nicolaides coarse space.
function fig12c_poisson_nicolaides()
  tbl = loadtable("study8_linear_cmp.csv")
  parts = loadtable("study8_partitions.csv")
  ms = sort(unique(tbl.m))
  methods = (
    "var_dd_additive",
    "emdd_q1_nicolaides",
    "remdd_q1",
    "remdd_q1_nicolaides",
    "var_dd_additive_history",
    "emdd_q2_nicolaides",
    "remdd_q2",
    "remdd_q2_nicolaides",
  )
  labels = Dict(
    "var_dd_additive" => "EMDD (q = 1)",
    "emdd_q1_nicolaides" => "EMDD (q = 1) + Nicolaides",
    "remdd_q1" => "REMDD (q = 1)",
    "remdd_q1_nicolaides" => "REMDD (q = 1) + Nicolaides",
    "var_dd_additive_history" => "EMDD (q = 2)",
    "emdd_q2_nicolaides" => "EMDD (q = 2) + Nicolaides",
    "remdd_q2" => "REMDD (q = 2)",
    "remdd_q2_nicolaides" => "REMDD (q = 2) + Nicolaides",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "emdd_q1_nicolaides" => :rect,
    "remdd_q1" => :rtriangle,
    "remdd_q1_nicolaides" => :diamond,
    "var_dd_additive_history" => :hexagon,
    "emdd_q2_nicolaides" => :pentagon,
    "remdd_q2" => :ltriangle,
    "remdd_q2_nicolaides" => :star5,
  )
  colors = tab10_colors(length(methods))
  selected = map(method -> method in methods, tbl.method)
  finite_res = tbl.resnorm[selected .& (tbl.resnorm .> 0)]
  ylims = (1e-10 * 0.5, maximum(finite_res) * 3)
  yticks = LogTicks(collect(1:-2:-9))
  fig = Figure(; size = (max(430 * length(ms), 1200), 350))
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel = "parallel local-solve batches",
      ylabel = column == 1 ? "residual norm ‖Axₖ-b‖₂" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = ((0, 60), ylims),
    )
    for (index, method) in enumerate(methods)
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.resnorm .> 1e-14)
      add_series!(
        ax,
        pick(tbl, :solves, mask) ./ m,
        logfloor(pick(tbl, :resnorm, mask));
        label = labels[method],
        color = colors[index],
        marker = markers[method],
      )
    end
    add_partition_inset!(fig[1, column], parts, m)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  return savefigs(fig, "fig12c_poisson_nicolaides")
end

# Weak scaling with fixed H/h and H/delta, including q=1 and q=2.
function fig12d_poisson_weak_scaling()
  tbl = loadtable("study8_weak_scaling.csv")
  ms = sort(unique(tbl.m))
  styles = (
    ("EMDD", "none", "EMDD", PALETTE[1], :circle, :dash),
    (
      "EMDD",
      "multiplicity",
      "EMDD + multiplicity PoU",
      PALETTE[1],
      :cross,
      :dot,
    ),
    (
      "EMDD",
      "harmonic",
      "EMDD + harmonic Nicolaides",
      PALETTE[1],
      :rect,
      :solid,
    ),
    ("REMDD", "none", "REMDD", PALETTE[2], :utriangle, :dash),
    (
      "REMDD",
      "multiplicity",
      "REMDD + multiplicity PoU",
      PALETTE[2],
      :xcross,
      :dot,
    ),
    (
      "REMDD",
      "harmonic",
      "REMDD + harmonic Nicolaides",
      PALETTE[2],
      :diamond,
      :solid,
    ),
  )
  fig = Figure(; size = (PAPER_FULL_WIDTH, 430))
  for (column, q) in enumerate((1, 2))
    ax = Axis(
      fig[1, column];
      xlabel = "number of subdomains, m",
      ylabel = column == 1 ? "parallel local-solve batches" : "",
      xscale = log2,
      xticks = (ms, string.(ms)),
      title = "q = $q",
    )
    for (family, coarse_kind, label, color, marker, linestyle) in styles
      batches = Int[]
      for m in ms
        mask =
          (tbl.m .== m) .& (tbl.q .== q) .& (tbl.family .== family) .&
          (tbl.coarse_kind .== coarse_kind)
        push!(batches, maximum(tbl.local_batches[mask]))
      end
      add_series!(ax, ms, batches; label, color, marker, linestyle)
    end
  end
  Legend(
    fig[2, 1:2],
    [
      [
        LineElement(; color = style[4], linestyle = style[6], linewidth = 2.5),
        MarkerElement(;
          color = style[4],
          marker = style[5],
          markersize = MARKERSIZE,
        ),
      ] for style in styles
    ],
    [style[3] for style in styles];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
    labelsize = 14,
  )
  rowgap!(fig.layout, 8)
  return savefigs(fig, "fig12d_poisson_weak_scaling")
end

# ---------------------------------------------------------------------------
# Fig 13: EVP residual -- comparison against one-level preconditioned baselines
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Figs 14--16: Gross--Pitaevskii convergence and ground-state densities
# ---------------------------------------------------------------------------
function fig14_gp_convergence()
  tbl = loadtable("study10_gp_conv.csv")
  parts = loadtable("study10_partitions.csv")
  solutions = loadtable("study10_gp_solutions.csv")
  betas = sort(unique(tbl.beta))
  ms = sort(unique(tbl.m))
  methods = (
    "gp_additive",
    "gp_additive_history",
    "gp_quadratic",
    "gp_quadratic_history",
    "gp_quadratic_history_2",
    "gp_quadratic_history_3",
    "gp_tangent_quadratic",
    "gp_tangent_quadratic_history",
    "gp_projected_qemdd",
    "gp_projected_qemdd_history",
    "gfdn_pcg_as_1",
    "gfdn_pcg_as_2",
    "gfdn_pcg_as_4",
    "cg_gfdn_pcg_as_1",
    "cg_gfdn_pcg_as_2",
    "cg_gfdn_pcg_as_4",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gp_quadratic" => "quadratic GP-EMDD",
    "gp_quadratic_history" => "quadratic GP-EMDD + history",
    "gp_quadratic_history_2" => "quadratic GP-EMDD + history (2)",
    "gp_quadratic_history_3" => "quadratic GP-EMDD + history (3)",
    "gp_tangent_quadratic" => "tangent KKT (q = 1)",
    "gp_tangent_quadratic_history" => "tangent KKT (q = 2)",
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
    "gp_quadratic" => :star4,
    "gp_quadratic_history" => :star6,
    "gp_quadratic_history_2" => :rect,
    "gp_quadratic_history_3" => :dtriangle,
    "gp_tangent_quadratic" => :pentagon,
    "gp_tangent_quadratic_history" => :diamond,
    "gp_projected_qemdd" => :utriangle,
    "gp_projected_qemdd_history" => :cross,
    "gfdn_pcg_as_1" => :circle,
    "gfdn_pcg_as_2" => :rect,
    "gfdn_pcg_as_4" => :dtriangle,
    "cg_gfdn_pcg_as_1" => :circle,
    "cg_gfdn_pcg_as_2" => :rect,
    "cg_gfdn_pcg_as_4" => :dtriangle,
  )
  colors = tab10_colors(length(methods))
  positive_residuals = tbl.resnorm[tbl.resnorm .> 0]
  ylimits = (1e-7, maximum(positive_residuals) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))
  yticks = LogTicks(ytick_exps)

  fig = Figure(size = (330 * length(ms), 275 * length(betas) + 90))
  for (row, beta) in enumerate(betas), (column, m) in enumerate(ms)
    ax = Axis(
      fig[row, column];
      xlabel = row == length(betas) ? "iteration" : "",
      ylabel = column == 1 ? "κ = $(Int(beta))\nresidual norm ‖rₖ‖₂" : "",
      title = row == 1 ? "m = $m" : "",
      yscale = log10,
      yticks = yticks,
      limits = (nothing, ylimits),
    )
    for (i, method) in enumerate(methods)
      mask =
        (tbl.method .== method) .& (tbl.beta .== beta) .& (tbl.m .== m) .&
        (tbl.resnorm .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        logfloor(pick(tbl, :resnorm, mask));
        label = labels[method],
        color = colors[i],
        marker = markers[method],
      )
    end
    inset_layout = semilinear_inset_layout(fig[row, column])
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
    fig[length(betas)+1, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig14_gp_convergence")
end

# Paper-facing subset of Figure 14, mirroring Figures 12b and 17b. Only the
# Henning--Jarlebring section 2.3 benchmark (kappa = 500) is shown, and q
# counts the current iterate as the first global vector. Nonlinear EMDD is
# shown for q=1,2 for both nonlinear and frozen-nonlinearity EMDD.
# The second legend compares fixed-step PCG metric inversions.
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

function fig16_gp_energy_gap()
  tbl = loadtable("study10_gp_conv.csv")
  parts = loadtable("study10_partitions.csv")
  solutions = loadtable("study10_gp_solutions.csv")
  betas = sort(unique(tbl.beta))
  ms = sort(unique(tbl.m))
  methods = (
    "gp_additive",
    "gp_additive_history",
    "gp_quadratic",
    "gp_quadratic_history",
    "gp_quadratic_history_2",
    "gp_quadratic_history_3",
    "gp_tangent_quadratic",
    "gp_tangent_quadratic_history",
    "gp_projected_qemdd",
    "gp_projected_qemdd_history",
    "gfdn_pcg_as_1",
    "gfdn_pcg_as_2",
    "gfdn_pcg_as_4",
    "cg_gfdn_pcg_as_1",
    "cg_gfdn_pcg_as_2",
    "cg_gfdn_pcg_as_4",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gp_quadratic" => "quadratic GP-EMDD",
    "gp_quadratic_history" => "quadratic GP-EMDD + history",
    "gp_quadratic_history_2" => "quadratic GP-EMDD + history (2)",
    "gp_quadratic_history_3" => "quadratic GP-EMDD + history (3)",
    "gp_tangent_quadratic" => "tangent KKT (q = 1)",
    "gp_tangent_quadratic_history" => "tangent KKT (q = 2)",
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
    "gp_quadratic" => :star4,
    "gp_quadratic_history" => :star6,
    "gp_quadratic_history_2" => :rect,
    "gp_quadratic_history_3" => :dtriangle,
    "gp_tangent_quadratic" => :pentagon,
    "gp_tangent_quadratic_history" => :diamond,
    "gp_projected_qemdd" => :utriangle,
    "gp_projected_qemdd_history" => :cross,
    "gfdn_pcg_as_1" => :circle,
    "gfdn_pcg_as_2" => :rect,
    "gfdn_pcg_as_4" => :dtriangle,
    "cg_gfdn_pcg_as_1" => :circle,
    "cg_gfdn_pcg_as_2" => :rect,
    "cg_gfdn_pcg_as_4" => :dtriangle,
  )
  colors = tab10_colors(length(methods))
  positive_gaps = tbl.energy_gap[tbl.energy_gap .> 0]
  ylimits = (max(minimum(positive_gaps) / 2, 1e-14), maximum(positive_gaps) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))
  yticks = LogTicks(ytick_exps)

  fig = Figure(size = (330 * length(ms), 275 * length(betas) + 90))
  for (row, beta) in enumerate(betas), (column, m) in enumerate(ms)
    ax = Axis(
      fig[row, column];
      xlabel = row == length(betas) ? "iteration" : "",
      ylabel = column == 1 ? "κ = $(Int(beta))\nenergy gap E(uₖ)−E(u★)" : "",
      title = row == 1 ? "m = $m" : "",
      yscale = log10,
      yticks = yticks,
      limits = (nothing, ylimits),
    )
    for (i, method) in enumerate(methods)
      mask =
        (tbl.method .== method) .& (tbl.beta .== beta) .& (tbl.m .== m) .&
        (tbl.energy_gap .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        logfloor(pick(tbl, :energy_gap, mask));
        label = labels[method],
        color = colors[i],
        marker = markers[method],
      )
    end
    inset_layout = semilinear_inset_layout(fig[row, column])
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
    fig[length(betas)+1, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig16_gp_energy_gap")
end

function fig15_gp_ground_states()
  tbl = loadtable("study10_gp_solutions.csv")
  betas = sort(unique(tbl.beta))
  N = tbl.N[1]
  xs = collect(range(-8 + 16 / N, 8 - 16 / N; length = N - 1))
  fig = Figure(size = (330 * length(betas), 410))
  for (column, beta) in enumerate(betas)
    mask = tbl.beta .== beta
    ax = Axis(
      fig[1, column];
      title = "κ = $(Int(beta))",
      xlabel = "x₁",
      ylabel = column == 1 ? "x₂" : "",
      aspect = DataAspect(),
    )
    density_max = maximum(pick(tbl, :density, mask))
    hm = heatmap!(
      ax,
      xs,
      xs,
      field_matrix(pick(tbl, :density, mask), N);
      colormap = :viridis,
      colorrange = (0, density_max),
    )
    Colorbar(
      fig[2, column],
      hm;
      vertical = false,
      label = "density |u|²",
      width = Relative(0.85),
    )
  end
  rowgap!(fig.layout, 5)
  savefigs(fig, "fig15_gp_ground_states")
end

# ---------------------------------------------------------------------------
# Fig 17: manufactured cubic semilinear Poisson comparison
# ---------------------------------------------------------------------------
function fig17_semilinear_poisson()
  conv = loadtable("study11_semilinear_conv.csv")
  solutions = loadtable("study11_semilinear_solution.csv")
  parts = loadtable("study11_semilinear_partitions.csv")
  ms = sort(unique(conv.m))
  methods = (
    "nonlinear_as",
    "anderson_ras",
    "nonlinear_cg_optim_as",
    "newton_pcg_as_4",
    "newton_pcg_as_8",
    "energy_imex_pcg_as",
    "aspin",
    "raspen",
    "var_dd",
    "var_dd_history",
    "var_dd_quadratic",
    "newton_pcg_as_1",
    "var_dd_quadratic_history",
    "newton_pcg_as_2",
  )
  labels = Dict(
    "nonlinear_as" => "nAS + optimal damping",
    "anderson_ras" => "Anderson-RAS (m = 4, NonlinearSolve.jl)",
    "nonlinear_cg_optim_as" => "AS-NCG (Optim.jl, Hager-Zhang)",
    "newton_pcg_as_4" => "Newton-PCG(AS, 4)",
    "newton_pcg_as_8" => "Newton-PCG(AS, 8)",
    "energy_imex_pcg_as" => "energy-IMEX-PCG(AS), Δt = 1",
    "aspin" => "ASPIN",
    "raspen" => "RASPEN",
    "var_dd" => "varDD",
    "var_dd_history" => "varDD + history (1)",
    "var_dd_quadratic" => "quadratic-model varDD",
    "newton_pcg_as_1" => "Newton-PCG(AS, 1)",
    "var_dd_quadratic_history" => "quadratic-model varDD + history (1)",
    "newton_pcg_as_2" => "Newton-PCG(AS, 2)",
  )
  markers = Dict(
    "nonlinear_as" => :circle,
    "anderson_ras" => :cross,
    "nonlinear_cg_optim_as" => :ltriangle,
    "newton_pcg_as_4" => :dtriangle,
    "newton_pcg_as_8" => :hexagon,
    "energy_imex_pcg_as" => :star5,
    "aspin" => :xcross,
    "raspen" => :pentagon,
    "var_dd" => :diamond,
    "var_dd_history" => :utriangle,
    "var_dd_quadratic" => :star4,
    "newton_pcg_as_1" => :circle,
    "var_dd_quadratic_history" => :star6,
    "newton_pcg_as_2" => :rect,
  )
  colors = tab10_colors(length(methods))
  initial_residuals = conv.relative_residual[(conv.outer .== 0) .& isfinite.(
    conv.relative_residual,
  )]
  ylimits = (1e-8, max(maximum(initial_residuals), 1.0) * 2)
  ytick_exps =
    sort(collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1]))))

  fig = Figure(size = (max(990, 330 * length(ms)), 550))
  Label(
    fig[0, 1:length(ms)],
    "−Δu + βu³ = f  in Ω,    β = 1,    u = 0  on ∂Ω";
    fontsize = 22,
    font = :bold,
  )
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel = "outer iteration",
      ylabel = column == 1 ? "relative residual" : "",
      title = "m = $m",
      yscale = log10,
      yticks = LogTicks(ytick_exps),
      limits = (nothing, ylimits),
    )
    for (index, method) in enumerate(methods)
      mask =
        (conv.beta .== 1.0) .& (conv.m .== m) .& (conv.method .== method) .&
        (conv.relative_residual .> 0)
      add_series!(
        ax,
        pick(conv, :outer, mask),
        logfloor(pick(conv, :relative_residual, mask); floor = 1e-16);
        label = labels[method],
        color = colors[index],
        marker = markers[method],
      )
    end
    hlines!(ax, [1e-7]; color = :black, linestyle = :dot, linewidth = 1.2)
    add_semilinear_solution_inset!(
      fig[1, column],
      solutions;
      halign = 0.68,
      inset_size = 0.23,
    )
    add_triangle_partition_inset!(
      fig[1, column],
      parts,
      m;
      halign = 0.99,
      inset_size = 0.23,
      inset_title = "partition",
    )
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 6,
    framevisible = true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig17_semilinear_poisson")
end

# Paper-facing subset of Figure 17, mirroring the focused linear-source
# comparison in Figure 12b. Here q counts the current iterate as the first
# global vector, so q=1 is plain EMDD and q=2 adds u_{k-1}.
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

# ---------------------------------------------------------------------------
# Figs 18--20: linear-source sensitivity studies
# ---------------------------------------------------------------------------

function study8_terminal_series(tbl, mask, parameter)
  xs = sort(unique(getproperty(tbl, parameter)[mask]))
  ys = Float64[]
  for x in xs
    indices = findall(mask .& (getproperty(tbl, parameter) .== x))
    terminal = indices[argmax(tbl.local_batches[indices])]
    push!(ys, tbl.local_batches[terminal])
  end
  return xs, ys
end

function fig18_poisson_scaling()
  tbl = loadtable("study8_sensitivity.csv")
  methods =
    ["var_dd_additive", "var_dd_additive_history", "pcg_as", "gmres_ras"]
  labels = Dict(
    "var_dd_additive" => "additive varDD",
    "var_dd_additive_history" => "additive varDD + history",
    "pcg_as" => "CG+AS",
    "gmres_ras" => "GMRES+RAS",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "var_dd_additive_history" => :hexagon,
    "pcg_as" => :diamond,
    "gmres_ras" => :pentagon,
  )
  colors = tab10_colors(length(methods))
  panels = (
    ("mesh", "fixed_layers", :N, "nested METIS, fixed layers ℓ = 2"),
    (
      "mesh",
      "fixed_delta_over_H",
      :N,
      "nested METIS, fixed relative overlap δ/H ≈ 0.1",
    ),
    ("overlap", "layer_sweep", :overlap, "overlap sweep, N = 64"),
  )
  fig = Figure(size = (430 * length(panels), 390))
  for (column, (experiment, regime, parameter, title)) in enumerate(panels)
    ax = Axis(
      fig[1, column];
      xlabel = parameter == :N ? "elements per direction, 1/h" :
               "overlap layers ℓ",
      ylabel = column == 1 ? "parallel local-solve batches" : "",
      title,
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== experiment) .& (tbl.regime .== regime) .&
        (tbl.method .== method)
      xs, ys = study8_terminal_series(tbl, mask, parameter)
      add_series!(
        ax,
        xs,
        ys;
        label = labels[method],
        color = colors[index],
        marker = markers[method],
      )
    end
  end
  Legend(
    fig[2, 1:length(panels)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig18_poisson_scaling")
end

function fig19_poisson_contrast()
  tbl = loadtable("study8_sensitivity.csv")
  inner = loadtable("study8_inner_systems.csv")
  methods =
    ["var_dd_additive", "var_dd_additive_history", "pcg_as", "gmres_ras"]
  labels = Dict(
    "var_dd_additive" => "additive varDD",
    "var_dd_additive_history" => "additive varDD + history",
    "pcg_as" => "CG+AS",
    "gmres_ras" => "GMRES+RAS",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "var_dd_additive_history" => :hexagon,
    "pcg_as" => :diamond,
    "gmres_ras" => :pentagon,
  )
  colors = tab10_colors(length(methods))
  fig = Figure(size = (1420, 410))
  ax_iterations = Axis(
    fig[1, 1];
    xlabel = "diffusion contrast κ",
    ylabel = "parallel local-solve batches to tolerance",
    xscale = log10,
    title = "outer convergence",
  )
  for (index, method) in enumerate(methods)
    mask = (tbl.experiment .== "contrast") .& (tbl.method .== method)
    xs, ys = study8_terminal_series(tbl, mask, :contrast)
    add_series!(
      ax_iterations,
      xs,
      ys;
      label = labels[method],
      color = colors[index],
      marker = markers[method],
    )
  end

  ax_condition = Axis(
    fig[1, 2];
    xlabel = "diffusion contrast κ",
    ylabel = "estimated κ₂",
    xscale = log10,
    yscale = log10,
    title = "first-sweep condition estimates",
  )
  system_specs = (
    ("schwarz_local", "Kᵢ = Rᵢ K Rᵢᵀ", PALETTE[7], :rect, :solid),
    (
      "vardd_local_orthonormal",
      "Qᵢᵀ K Qᵢ (local varDD)",
      PALETTE[5],
      :utriangle,
      :dash,
    ),
    (
      "combine_first_sweep",
      "Qᴄᵀ K Qᴄ (combination)",
      PALETTE[6],
      :diamond,
      :solid,
    ),
  )
  for (system, label, color, marker, linestyle) in system_specs
    system_mask = (inner.experiment .== "contrast") .& (inner.system .== system)
    contrasts = sort(unique(inner.contrast[system_mask]))
    maxima = [
      maximum(
        inner.condition_estimate[system_mask .& (inner.contrast .== contrast)],
      ) for contrast in contrasts
    ]
    add_series!(
      ax_condition,
      contrasts,
      maxima;
      label,
      color,
      marker,
      linestyle,
    )
  end
  axislegend(ax_condition; position = :lt)

  ax_inner = Axis(
    fig[1, 3];
    xlabel = "diffusion contrast κ",
    ylabel = "local CG iterations",
    xscale = log10,
    title = "iterative local-solve work (censored)",
    limits = (nothing, (0, 2200)),
  )
  mask = (inner.experiment .== "contrast") .& (inner.system .== "schwarz_local")
  contrasts = sort(unique(inner.contrast[mask]))
  medians = Float64[]
  maxima = Float64[]
  for contrast in contrasts
    values = inner.cg_iterations[mask .& (inner.contrast .== contrast)]
    push!(medians, median(values))
    push!(maxima, maximum(values))
  end
  add_series!(
    ax_inner,
    contrasts,
    medians;
    label = "median subdomain",
    color = PALETTE[5],
    marker = :rect,
  )
  add_series!(
    ax_inner,
    contrasts,
    maxima;
    label = "most difficult subdomain",
    color = PALETTE[6],
    marker = :diamond,
  )
  iteration_cap = 2000
  censored = unique(
    vcat(
      contrasts[medians .>= iteration_cap],
      contrasts[maxima .>= iteration_cap],
    ),
  )
  hlines!(
    ax_inner,
    [iteration_cap];
    color = (:black, 0.55),
    linestyle = :dot,
    linewidth = 1.4,
  )
  scatter!(
    ax_inner,
    censored,
    fill(iteration_cap, length(censored));
    label = "censored (≥ 2000)",
    color = :black,
    marker = :utriangle,
    markersize = 14,
  )
  axislegend(ax_inner; position = :lt)
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig19_poisson_contrast")
end

function fig20_poisson_history()
  tbl = loadtable("study8_sensitivity.csv")
  regimes = ("homogeneous", "contrast_1e4")
  titles = Dict(
    "homogeneous" => "homogeneous diffusion",
    "contrast_1e4" => "inclusion contrast κ = 10⁴",
  )
  fig = Figure(size = (850, 370))
  Label(
    fig[0, 1:2],
    "One previous iterate provides nearly all of the history benefit";
    fontsize = 22,
    font = :bold,
  )
  for (column, regime) in enumerate(regimes)
    mask = (tbl.experiment .== "history") .& (tbl.regime .== regime)
    depths, batches = study8_terminal_series(tbl, mask, :history_depth)
    ax_batches = Axis(
      fig[1, column];
      xlabel = "history depth q",
      ylabel = column == 1 ? "parallel local-solve batches to tolerance" : "",
      title = titles[regime],
    )
    add_series!(
      ax_batches,
      depths,
      batches;
      color = PALETTE[2],
      marker = :hexagon,
    )
    vlines!(ax_batches, [1]; color = (:black, 0.45), linestyle = :dash)
  end
  savefigs(fig, "fig20_poisson_history")
end

# ---------------------------------------------------------------------------
# Figs 21--23: semilinear sensitivity studies
# ---------------------------------------------------------------------------

const SEMILINEAR_BENCHMARK_LABELS = Dict(
  "anderson_ras" => "Anderson-RAS (m = 4, NonlinearSolve.jl)",
  "newton_pcg_as" => "Newton-PCG(AS, 4)",
  "energy_imex_pcg_as" => "energy-IMEX-PCG(AS)",
  "aspin" => "ASPIN",
  "raspen" => "RASPEN",
  "var_dd" => "varDD",
  "var_dd_history" => "varDD + history",
)

const SEMILINEAR_BENCHMARK_METHODS = (
  "anderson_ras",
  "newton_pcg_as",
  "energy_imex_pcg_as",
  "aspin",
  "raspen",
  "var_dd",
  "var_dd_history",
)

const SEMILINEAR_BENCHMARK_MARKERS = Dict(
  "anderson_ras" => :cross,
  "newton_pcg_as" => :dtriangle,
  "energy_imex_pcg_as" => :star5,
  "aspin" => :xcross,
  "raspen" => :pentagon,
  "var_dd" => :diamond,
  "var_dd_history" => :utriangle,
)

function fig21_semilinear_mesh_scaling()
  tbl = loadtable("study11_semilinear_sensitivity.csv")
  methods = collect(SEMILINEAR_BENCHMARK_METHODS)
  mask_mesh = tbl.experiment .== "mesh"
  Ns = sort(unique(tbl.N[mask_mesh]))
  colors = tab10_colors(length(methods))
  fig = Figure(size = (1120, 500))
  Label(
    fig[0, 1:3],
    "Mesh refinement h, h/2, h/4 with fixed physical overlap";
    fontsize = 22,
    font = :bold,
  )
  specifications = (
    (:outer_iterations, "outer iterations"),
    (:nonlinear_local_batches, "nonlinear local batches"),
    (:linear_as_batches, "linear AS batches"),
  )
  for (column, (quantity, ylabel)) in enumerate(specifications)
    ax =
      Axis(fig[1, column]; xlabel = "cells per coordinate direction N", ylabel)
    for (index, method) in enumerate(methods)
      mask = mask_mesh .& (tbl.method .== method)
      any(mask) || continue
      values = pick(tbl, quantity, mask)
      all(iszero, values) && quantity != :outer_iterations && continue
      add_series!(
        ax,
        pick(tbl, :N, mask),
        values;
        color = colors[index],
        marker = SEMILINEAR_BENCHMARK_MARKERS[method],
      )
    end
    ax.xticks = Ns
  end
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, SEMILINEAR_BENCHMARK_MARKERS; colors),
    [SEMILINEAR_BENCHMARK_LABELS[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig21_semilinear_mesh_scaling")
end

function fig22_semilinear_newton_inner()
  tbl = loadtable("study11_semilinear_sensitivity.csv")
  mask = tbl.experiment .== "newton_inner"
  parameters = [1, 2, 4, 8, 0]
  labels = ["1", "2", "4", "8", "accurate"]
  positions = collect(eachindex(parameters))
  outer = [
    only(tbl.outer_iterations[mask .& (tbl.parameter .== p)]) for
    p in parameters
  ]
  batches = [
    only(tbl.linear_as_batches[mask .& (tbl.parameter .== p)]) for
    p in parameters
  ]
  converged = [
    only(tbl.converged[mask .& (tbl.parameter .== p)]) == 1 for p in parameters
  ]
  fig = Figure(size = (780, 350))
  Label(
    fig[0, 1:2],
    "Inexact Newton trades outer steps for inner PCG(AS) work";
    fontsize = 22,
    font = :bold,
  )
  for (column, (values, ylabel)) in enumerate((
    (outer, "outer Newton iterations"),
    (batches, "total linear AS batches"),
  ))
    ax = Axis(
      fig[1, column];
      xlabel = "PCG(AS) steps per Newton update",
      ylabel,
      xticks = (positions, labels),
    )
    add_series!(ax, positions, values; color = PALETTE[3], marker = :dtriangle)
    failed = positions[.!converged]
    if !isempty(failed)
      scatter!(
        ax,
        failed,
        values[.!converged];
        color = :black,
        marker = :utriangle,
        markersize = 14,
        label = "iteration budget reached",
      )
      column == 1 && axislegend(ax; position = :rt)
    end
  end
  savefigs(fig, "fig22_semilinear_newton_inner")
end

function fig23_semilinear_history()
  tbl = loadtable("study11_semilinear_sensitivity.csv")
  algorithms = ("anderson_ras", "var_dd_history")
  titles = ("Anderson-RAS", "varDD")
  fig = Figure(size = (780, 350))
  Label(
    fig[0, 1:2],
    "Effect of multisecant and iterate history depth";
    fontsize = 22,
    font = :bold,
  )
  for (column, (method, title)) in enumerate(zip(algorithms, titles))
    mask = (tbl.experiment .== "history") .& (tbl.method .== method)
    order = sortperm(tbl.parameter[mask])
    depths = tbl.parameter[mask][order]
    iterations = tbl.outer_iterations[mask][order]
    ax = Axis(
      fig[1, column];
      xlabel = method == "anderson_ras" ? "Anderson memory m" :
               "history depth q",
      ylabel = column == 1 ? "outer iterations to tolerance" : "",
      title,
      xticks = depths,
    )
    add_series!(
      ax,
      depths,
      iterations;
      color = PALETTE[column+1],
      marker = :hexagon,
    )
  end
  savefigs(fig, "fig23_semilinear_history")
end

# ---------------------------------------------------------------------------
# Figs 24--27: generalized-eigenproblem sensitivity and work diagnostics
# ---------------------------------------------------------------------------

const EVP_SENSITIVITY_LABELS = Dict(
  "var_dd" => "varDD",
  "var_dd_history" => "varDD + history (1)",
  "lobpcg_as" => "LOBPCG+AS",
  "jd_gmres_as" => "JD-GMRES(AS)",
  "si_lanczos_pcg_as" => "SI-Lanczos-PCG(AS)",
)

const EVP_SENSITIVITY_METHODS =
  ("var_dd", "var_dd_history", "lobpcg_as", "jd_gmres_as", "si_lanczos_pcg_as")

const EVP_SENSITIVITY_MARKERS = Dict(
  "var_dd" => :circle,
  "var_dd_history" => :hexagon,
  "lobpcg_as" => :utriangle,
  "jd_gmres_as" => :diamond,
  "si_lanczos_pcg_as" => :pentagon,
)

function evp_terminal_series(tbl, mask, parameter, quantity)
  parameters = sort(unique(getproperty(tbl, parameter)[mask]))
  values = Float64[]
  for value in parameters
    indices = findall(mask .& (getproperty(tbl, parameter) .== value))
    terminal = indices[argmax(tbl.iteration[indices])]
    push!(values, getproperty(tbl, quantity)[terminal])
  end
  return parameters, values
end

function fig24_evp_scaling()
  tbl = loadtable("study9_evp_sensitivity.csv")
  methods = collect(EVP_SENSITIVITY_METHODS)
  colors = tab10_colors(length(methods))
  panels = (
    ("mesh", "fixed_layers", :N, "fixed overlap layers, ℓ = 2"),
    ("mesh", "fixed_delta_over_H", :N, "fixed relative overlap, δ/H = 0.1"),
    (
      "oscillation",
      "frequency_sweep",
      :frequency,
      "oscillatory diffusion, contrast κ = 10³",
    ),
  )
  fig = Figure(size = (1320, 420))
  for (column, (experiment, regime, parameter, title)) in enumerate(panels)
    ax = Axis(
      fig[1, column];
      xlabel = parameter == :N ? "elements per direction, 1/h" : "frequency ν",
      ylabel = column == 1 ? "outer iterations to tolerance" : "",
      title,
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== experiment) .& (tbl.regime .== regime) .&
        (tbl.method .== method)
      xs, ys = evp_terminal_series(tbl, mask, parameter, :iteration)
      add_series!(
        ax,
        xs,
        ys;
        color = colors[index],
        marker = EVP_SENSITIVITY_MARKERS[method],
      )
    end
  end
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, EVP_SENSITIVITY_MARKERS; colors),
    [EVP_SENSITIVITY_LABELS[method] for method in methods];
    orientation = :horizontal,
    nbanks = 1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig24_evp_scaling")
end

function fig25_evp_linear_work()
  tbl = loadtable("study9_evp_sensitivity.csv")
  methods = ["lobpcg_as", "jd_gmres_as", "si_lanczos_pcg_as"]
  colors = tab10_colors(length(methods))
  fig = Figure(size = (900, 390))
  Label(
    fig[0, 1:2],
    "Linear-preconditioner work is separate from varDD local eigenproblems";
    fontsize = 22,
    font = :bold,
  )
  for (column, (quantity, ylabel)) in enumerate((
    (:linear_as_batches, "parallel linear AS batches"),
    (:global_operator_products, "global K/M operator applications"),
  ))
    ax = Axis(
      fig[1, column];
      xlabel = "elements per direction, 1/h",
      ylabel,
      title = column == 1 ? "local linear solves" : "global operator work",
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== "mesh") .& (tbl.regime .== "fixed_delta_over_H") .&
        (tbl.method .== method)
      if quantity == :global_operator_products
        xs = sort(unique(tbl.N[mask]))
        ys = Float64[]
        for N in xs
          indices = findall(mask .& (tbl.N .== N))
          terminal = indices[argmax(tbl.iteration[indices])]
          push!(
            ys,
            tbl.global_k_products[terminal] + tbl.global_m_products[terminal],
          )
        end
      else
        xs, ys = evp_terminal_series(tbl, mask, :N, quantity)
      end
      add_series!(
        ax,
        xs,
        ys;
        color = colors[index],
        marker = EVP_SENSITIVITY_MARKERS[method],
      )
    end
  end
  Legend(
    fig[2, 1:2],
    legend_line_marker_elements(methods, EVP_SENSITIVITY_MARKERS; colors),
    [EVP_SENSITIVITY_LABELS[method] for method in methods];
    orientation = :horizontal,
    nbanks = 1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig25_evp_linear_work")
end

function fig26_evp_history()
  tbl = loadtable("study9_evp_sensitivity.csv")
  combination = loadtable("study9_evp_sensitivity_combination.csv")
  mask = (tbl.experiment .== "history") .& (tbl.method .== "var_dd_history")
  depths, iterations =
    evp_terminal_series(tbl, mask, :history_depth, :iteration)
  conditions = Float64[]
  for depth in depths
    condition_mask =
      (combination.experiment .== "history") .&
      (combination.method .== "var_dd_history") .&
      (combination.history_depth .== depth)
    push!(conditions, maximum(combination.mass_condition[condition_mask]))
  end
  fig = Figure(size = (820, 360))
  Label(
    fig[0, 1:2],
    "One previous iterate supplies the useful EVP history enrichment";
    fontsize = 22,
    font = :bold,
  )
  ax_iterations = Axis(
    fig[1, 1];
    xlabel = "history depth q",
    ylabel = "outer iterations to tolerance",
  )
  add_series!(
    ax_iterations,
    depths,
    iterations;
    color = PALETTE[2],
    marker = :hexagon,
  )
  vlines!(ax_iterations, [1]; color = (:black, 0.45), linestyle = :dash)
  ax_condition = Axis(
    fig[1, 2];
    xlabel = "history depth q",
    ylabel = "max κ₂(QᵀMQ)",
    yscale = log10,
  )
  add_series!(
    ax_condition,
    depths,
    conditions;
    color = PALETTE[5],
    marker = :diamond,
  )
  vlines!(ax_condition, [1]; color = (:black, 0.45), linestyle = :dash)
  savefigs(fig, "fig26_evp_history")
end

function fig27_evp_local_work()
  tbl = loadtable("study9_evp_sensitivity.csv")
  local_stats = loadtable("study9_evp_sensitivity_local.csv")
  Ns = sort(
    unique(
      tbl.N[(tbl.experiment .== "mesh") .& (tbl.regime .== "fixed_delta_over_H")],
    ),
  )
  fig = Figure(size = (1260, 390))
  Label(
    fig[0, 1:3],
    "Local-system size, critical path, and factor storage";
    fontsize = 22,
    font = :bold,
  )

  ax_dimension = Axis(
    fig[1, 1];
    xlabel = "elements per direction, 1/h",
    ylabel = "maximum local dimension",
  )
  for (system, label, color, marker) in (
    ("schwarz_block", "AS block", PALETTE[4], :rect),
    ("vardd_augmented_pencil", "varDD augmented pencil", PALETTE[2], :circle),
  )
    values = Float64[]
    for N in Ns
      mask =
        (local_stats.experiment .== "mesh") .&
        (local_stats.regime .== "fixed_delta_over_H") .& (local_stats.N .== N) .&
        (local_stats.system .== system) .& (
          system == "schwarz_block" ? trues(length(local_stats.N)) :
          local_stats.method .== "var_dd"
        )
      push!(values, maximum(local_stats.dimension[mask]))
    end
    add_series!(ax_dimension, Ns, values; label, color, marker)
  end
  axislegend(ax_dimension; position = :lt)

  ax_critical = Axis(
    fig[1, 2];
    xlabel = "elements per direction, 1/h",
    ylabel = "critical-path local LOBPCG iterations",
  )
  for (method, color, marker) in (
    ("var_dd", PALETTE[1], :circle),
    ("var_dd_history", PALETTE[2], :hexagon),
  )
    mask =
      (tbl.experiment .== "mesh") .& (tbl.regime .== "fixed_delta_over_H") .&
      (tbl.method .== method)
    xs, ys = evp_terminal_series(tbl, mask, :N, :local_iterations_critical)
    add_series!(
      ax_critical,
      xs,
      ys;
      label = EVP_SENSITIVITY_LABELS[method],
      color,
      marker,
    )
  end
  axislegend(ax_critical; position = :lt)

  ax_factor = Axis(
    fig[1, 3];
    xlabel = "elements per direction, 1/h",
    ylabel = "maximum local factor nnz",
    yscale = log10,
  )
  for (system, label, color, marker) in (
    ("schwarz_block", "AS block", PALETTE[4], :rect),
    ("vardd_augmented_pencil", "varDD augmented pencil", PALETTE[2], :circle),
  )
    values = Float64[]
    for N in Ns
      mask =
        (local_stats.experiment .== "mesh") .&
        (local_stats.regime .== "fixed_delta_over_H") .& (local_stats.N .== N) .&
        (local_stats.system .== system) .& (
          system == "schwarz_block" ? trues(length(local_stats.N)) :
          local_stats.method .== "var_dd"
        )
      push!(values, maximum(local_stats.factor_nnz[mask]))
    end
    add_series!(ax_factor, Ns, values; label, color, marker)
  end
  axislegend(ax_factor; position = :lt)
  savefigs(fig, "fig27_evp_local_work")
end

# ---------------------------------------------------------------------------
# Fig 28: compact paper figure comparing variants of the method
# ---------------------------------------------------------------------------
function fig28_method_variants()
  tbl = loadtable("study12_variants.csv")
  methods = (
    "emdd",
    "memdd",
    "remdd",
    "emdd_multiplicity",
    "emdd_nicolaides",
    "remdd_nicolaides",
  )
  labels = Dict(
    method => tbl.label[findfirst(tbl.method .== method)] for method in methods
  )
  markers = Dict(
    "emdd" => :circle,
    "memdd" => :star5,
    "remdd" => :utriangle,
    "emdd_multiplicity" => :cross,
    "emdd_nicolaides" => :rect,
    "remdd_nicolaides" => :diamond,
  )
  colors = tab10_colors(length(methods))
  ms = sort(unique(tbl.m))
  xmax = maximum(tbl.iteration) + 2
  fig = Figure(; size = (PAPER_FULL_WIDTH, 350))
  for (column, m) in enumerate(ms)
    N = tbl.N[findfirst(tbl.m .== m)]
    ax = Axis(
      fig[1, column];
      xlabel = "outer sweep k",
      ylabel = column == 1 ? "relative residual" : "",
      yscale = log10,
      yticks = LogTicks(collect(0:-2:-10)),
      title = "m = $m, 1/h = $N",
      limits = ((0, xmax), (5e-11, 2.0)),
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.m .== m) .& (tbl.method .== method) .& (tbl.relative_residual .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        pick(tbl, :relative_residual, mask);
        color = colors[index],
        marker = markers[method],
        markersize = 5,
      )
    end
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(
      methods,
      markers;
      colors,
      markersizes = fill(5, length(methods)),
    ),
    [labels[method] for method in methods];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = true,
    labelsize = 12,
  )
  rowgap!(fig.layout, 6)
  return savefigs(fig, "fig28_method_variants")
end

# ---------------------------------------------------------------------------

function make_all_figures()
  println("plots: generating all figures from data/*.csv")
  fig01_setup()
  fig02_evp_convergence()
  fig03_evp_baseline()
  fig04_evp_haccuracy()
  fig05_robustness()
  fig06_timing()
  fig07_poisson()
  fig09_heat_warmstart()
  fig10_heat_dissipation()
  fig11_local3d()
  fig12_poisson_cmp()
  fig12b_poisson_cmp_paper()
  fig12c_poisson_nicolaides()
  fig12d_poisson_weak_scaling()
  fig13_evp_cmp()
  fig14_gp_convergence()
  fig14b_gp_convergence_paper()
  fig15_gp_ground_states()
  fig16_gp_energy_gap()
  fig17_semilinear_poisson()
  fig17b_semilinear_poisson_paper()
  fig17b_semilinear_l_shape_section61()
  fig18_poisson_scaling()
  fig19_poisson_contrast()
  fig20_poisson_history()
  fig21_semilinear_mesh_scaling()
  fig22_semilinear_newton_inner()
  fig23_semilinear_history()
  fig24_evp_scaling()
  fig25_evp_linear_work()
  fig26_evp_history()
  fig27_evp_local_work()
  fig28_method_variants()
  println("plots: done -> $(FIG_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
  make_all_figures()
end
