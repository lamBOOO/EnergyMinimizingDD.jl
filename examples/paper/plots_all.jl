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
const MARKERSIZE = 8
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
  markersize = MARKERSIZE,
  linewidth = 2.5,
)
  c = isnothing(color) ? PALETTE[1] : color
  lines!(ax, x, y; label = label, color = c, linestyle = linestyle, linewidth = linewidth)
  scatter!(ax, x, y; color = c, marker = marker, markersize = markersize)
end

function add_legend!(ax; position = :rt)
  axislegend(ax; position = position, framevisible = true)
end

function color_for(i)
  PALETTE[mod1(i, length(PALETTE))]
end

function add_partition_boundaries!(ax, owner_grid; color = (:black, 0.55), linewidth = 0.45)
  n = size(owner_grid, 1)
  xs = Float64[]
  ys = Float64[]
  function add_segment!(x1, y1, x2, y2)
    append!(xs, (x1, x2, NaN))
    append!(ys, (y1, y2, NaN))
  end
  for i in 1:n-1, j in 1:n
    owner_grid[i, j] == owner_grid[i+1, j] && continue
    x = i / n
    add_segment!(x, (j - 1) / n, x, j / n)
  end
  for i in 1:n, j in 1:n-1
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
)
  return [
    [
      LineElement(color = colors[i], linewidth = 2.5),
      MarkerElement(
        color = colors[i],
        marker = markers[method],
        markersize = MARKERSIZE,
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
    [RGBAf(0, 0, 0, mult_grid[i, j] > 1 ? 0.1f0 * mult_grid[i, j] : 0.0f0)
     for i in axes(mult_grid, 1), j in axes(mult_grid, 2)];
  )
  add_partition_boundaries!(pax, owner_grid)
  if !isempty(inset_title)
    text!(
      pax,
      0.5,
      0.96;
      text=inset_title,
      align=(:center, :top),
      fontsize=9,
      color=:white,
      font=:bold,
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
    text="density",
    align=(:center, :top),
    fontsize=9,
    color=:white,
    font=:bold,
  )
  return sax
end

function add_semilinear_solution_inset!(
  figpos,
  solutions;
  halign=0.68,
  valign=0.97,
  inset_size=0.23,
)
  N = solutions.N[1]
  values = solutions.value
  sax = Axis(
    figpos;
    width=Relative(inset_size),
    height=Relative(inset_size),
    halign=halign,
    valign=valign,
    tellwidth=false,
    tellheight=false,
    aspect=DataAspect(),
    limits=(0, 1, 0, 1),
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
  for iy = 0:N-1, ix = 0:N-1
    lower_left = node(ix, iy)
    lower_right = node(ix + 1, iy)
    upper_left = node(ix, iy + 1)
    upper_right = node(ix + 1, iy + 1)
    push!(faces, TriangleFace(lower_left, lower_right, upper_left))
    push!(faces, TriangleFace(lower_right, upper_right, upper_left))
  end
  color_limit = maximum(abs, values)
  mesh!(
    sax,
    vertices,
    faces;
    color=vertex_values,
    colormap=:balance,
    colorrange=(-color_limit, color_limit),
    shading=NoShading,
  )
  wireframe!(
    sax,
    CairoMakie.GeometryBasics.Mesh(vertices, faces);
    color=(:black, 0.14),
    linewidth=0.18,
  )
  text!(
    sax,
    0.5,
    0.96;
    text="exact u★",
    align=(:center, :top),
    fontsize=9,
    color=:white,
    font=:bold,
  )
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
)
  pmask = parts.m .== m
  pax = Axis(
    figpos;
    width=Relative(inset_size),
    height=Relative(inset_size),
    halign=halign,
    valign=valign,
    tellwidth=false,
    tellheight=false,
    aspect=DataAspect(),
    limits=(0, 1, 0, 1),
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
    color=owners,
    colormap=:Spectral_9,
    colorrange=(1, m),
    strokecolor=(:black, 0.18),
    strokewidth=0.2,
  )
  poly!(
    pax,
    polygons;
    color=[
      RGBAf(0, 0, 0, value > 1 ? 0.1f0 * value : 0.0f0) for
      value in multiplicities
    ],
    strokewidth=0,
  )
  text!(
    pax,
    0.5,
    0.96;
    text=inset_title,
    align=(:center, :top),
    fontsize=9,
    color=:white,
    font=:bold,
  )
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
  ax = Axis(fig[1, 1]; xlabel = "iteration", ylabel = "eigenvalue error", yscale = log10)
  for (i, (method, label)) in
      enumerate((("var_dd", "variational DD"), ("inverse_iteration", "inverse iteration")))
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
  lines!(ax, href, eref; label = "quadratic reference", color = :black, linestyle = :dash, linewidth = 1.5)
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
      (tbl.problem .== problem) .&
      (tbl.m .== m) .&
      (tbl.overlap .== 2) .&
      .!isnan.(tbl.time_s)
    any(mask) || continue
    Ns = pick(tbl, :N, mask)
    ts = pick(tbl, :time_s, mask)
    its = pick(tbl, :iters, mask)
    add_series!(ax1, Ns, ts; label = "$plabel, m = $m", color = color_for(ci), linestyle = ls)
    add_series!(ax2, Ns, ts ./ its; label = "$plabel, m = $m", color = color_for(ci), linestyle = ls)
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
  ax1 = Axis(fig[1, 1]; xlabel = "iteration", ylabel = "energy gap", yscale = log10)
  ax2 = Axis(fig[1, 2]; xlabel = "iteration", ylabel = "residual norm", yscale = log10)
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
  hlines!(ax2, [1e-12]; label = "tolerance", color = :black, linestyle = :dash, linewidth = 1.5)
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
  ax1 = Axis(fig[1, 1]; xlabel = "time step n", ylabel = "DD iterations per step")
  for (i, (mode, label)) in enumerate((("cold", "cold start"), ("warm", "warm start")))
    mask = wc.mode .== mode
    add_series!(ax1, pick(wc, :step, mask), pick(wc, :iters, mask); label = label, color = color_for(i))
  end
  ylims!(ax1, 0, maximum(wc.iters) + 2)
  add_legend!(ax1; position = :rc)

  taus = sort(unique(ts.tau))
  totals = [sum(pick(ts, :iters, ts.tau .== tau)) for tau in taus]
  ax2 = Axis(fig[1, 2]; xlabel = "time step size tau", ylabel = "total DD iterations (T = 0.2)", xscale = log10)
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
  ax1 = Axis(fig[1, 1]; xlabel = "t", ylabel = "energy gap to steady state", yscale = log10)
  ax2 = Axis(fig[1, 2]; xlabel = "t", ylabel = "rel. error vs direct solve", yscale = log10)
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
  subs = sort(unique(tbl.sub[tbl.kind.=="update"]))
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
    surface!(ax, xs, xs, field_matrix_with_bc(pick(tbl, :value, mask_u), N); colormap = :viridis)

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
  colors = Makie.resample_cmap(:tab10, length(methods))
  finite_res = tbl.resnorm[tbl.resnorm .> 0]
  ylims = (1e-10 * 0.5, maximum(finite_res) * 3)
  ytick_exps = sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size = (max(430 * length(ms), 1200), 350))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "iteration",
      ylabel = j == 1 ? "residual norm ‖Axₖ-b‖₂" : "",
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

# Paper-facing subset of Figure 12. Here q counts the current iterate as the
# first global vector, so q=1 is plain additive varDD and q=2 adds u_{k-1}.
function fig12b_poisson_cmp_paper()
  tbl = loadtable("study8_linear_cmp.csv")
  parts = loadtable("study8_partitions.csv")
  ms = sort(unique(tbl.m))
  methods = (
    "var_dd_additive",
    "remdd_q1",
    "var_dd_additive_history",
    "remdd_q2",
    "ras",
    "pcg_as",
    "gmres_ras",
  )
  labels = Dict(
    "var_dd_additive" => "EMDD (q = 1)",
    "remdd_q1" => "REMDD (q = 1)",
    "remdd_q2" => "REMDD (q = 2)",
    "var_dd_additive_history" => "EMDD (q = 2)",
    "ras" => "RAS",
    "pcg_as" => "CG+AS",
    "gmres_ras" => "GMRES+RAS",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "remdd_q1" => :rtriangle,
    "remdd_q2" => :ltriangle,
    "var_dd_additive_history" => :hexagon,
    "ras" => :utriangle,
    "pcg_as" => :diamond,
    "gmres_ras" => :pentagon,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  finite_res = tbl.resnorm[tbl.resnorm .> 0]
  ylims = (1e-9, maximum(finite_res) * 3)
  yticks = LogTicks(collect(1:-2:-9))
  fig = Figure(size=(PAPER_FULL_WIDTH, 330))
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel="iteration",
      ylabel=column == 1 ? "residual norm ‖Axₖ-b‖₂" : "",
      yscale=log10,
      yticks,
      title="m = $m",
      limits=((0, 100), ylims),
    )
    for (index, method) in enumerate(methods)
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.resnorm .> 0)
      add_series!(
        ax,
        pick(tbl, :solves, mask) ./ m,
        pick(tbl, :resnorm, mask);
        label=labels[method],
        color=colors[index],
        marker=markers[method],
      )
    end
    add_partition_inset!(fig[1, column], parts, m)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation=:horizontal,
    nbanks=1,
    framevisible=true,
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
  colors = Makie.resample_cmap(:tab10, length(methods))
  selected = map(method -> method in methods, tbl.method)
  finite_res = tbl.resnorm[selected .& (tbl.resnorm .> 0)]
  ylims = (1e-10 * 0.5, maximum(finite_res) * 3)
  yticks = LogTicks(collect(1:-2:-9))
  fig = Figure(; size=(max(430 * length(ms), 1200), 350))
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel="parallel local-solve batches",
      ylabel=column == 1 ? "residual norm ‖Axₖ-b‖₂" : "",
      yscale=log10,
      yticks=yticks,
      title="m = $m",
      limits=((0, 60), ylims),
    )
    for (index, method) in enumerate(methods)
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.resnorm .> 1e-14)
      add_series!(
        ax,
        pick(tbl, :solves, mask) ./ m,
        logfloor(pick(tbl, :resnorm, mask));
        label=labels[method],
        color=colors[index],
        marker=markers[method],
      )
    end
    add_partition_inset!(fig[1, column], parts, m)
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation=:horizontal,
    nbanks=2,
    framevisible=true,
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
  fig = Figure(; size=(PAPER_FULL_WIDTH, 430))
  for (column, q) in enumerate((1, 2))
    ax = Axis(
      fig[1, column];
      xlabel="number of subdomains, m",
      ylabel=column == 1 ? "parallel local-solve batches" : "",
      xscale=log2,
      xticks=(ms, string.(ms)),
      title="q = $q",
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
        LineElement(; color=style[4], linestyle=style[6], linewidth=2.5),
        MarkerElement(; color=style[4], marker=style[5], markersize=MARKERSIZE),
      ] for style in styles
    ],
    [style[3] for style in styles];
    orientation=:horizontal,
    nbanks=2,
    framevisible=true,
    labelsize=14,
  )
  rowgap!(fig.layout, 8)
  return savefigs(fig, "fig12d_poisson_weak_scaling")
end

# ---------------------------------------------------------------------------
# Fig 13: EVP residual -- comparison against one-level preconditioned baselines
# ---------------------------------------------------------------------------
function fig13_evp_cmp()
  tbl = loadtable("study9_evp_cmp.csv")
  parts = loadtable("study9_partitions.csv")
  ms = sort(unique(tbl.m))
  labels = Dict(
    "var_dd" => "varDD",
    "var_dd_history" => "varDD + history (1)",
    "lopsd_as" => "LOPSD+AS",
    "lobpcg_as" => "LOBPCG+AS",
    "jd_gmres_as" => "JD–GMRES(AS)",
    "si_lanczos_pcg_as" => "SI-Lanczos–PCG(AS)",
  )
  methods = (
    "var_dd",
    "var_dd_history",
    "lopsd_as",
    "lobpcg_as",
    "jd_gmres_as",
    "si_lanczos_pcg_as",
  )
  markers = Dict(
    "var_dd" => :circle,
    "var_dd_history" => :hexagon,
    "lopsd_as" => :rect,
    "lobpcg_as" => :utriangle,
    "jd_gmres_as" => :diamond,
    "si_lanczos_pcg_as" => :pentagon,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  finite_res = tbl.relative_residual[tbl.relative_residual .> 0]
  ylims = (1e-6 * 0.5, maximum(finite_res) * 1.5)
  ytick_exps = sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size = (max(990, 330 * length(ms)), 390))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "iteration",
      ylabel = j == 1 ? "relative residual" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = (nothing, ylims),
    )
    for (i, method) in enumerate(methods)
      mask =
        (tbl.m .== m) .&
        (tbl.method .== method) .&
        (tbl.relative_residual .> 0)
      add_series!(
        ax,
        pick(tbl, :iteration, mask),
        logfloor(pick(tbl, :relative_residual, mask));
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
    "gfdn_au_exact",
    "cg_gfdn_au_exact",
    "gfdn_au_as",
    "cg_gfdn_au_as",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gfdn_au_exact" => "exact GFDN(aᵤ) (optimal step)",
    "cg_gfdn_au_exact" => "exact CG-GFDN(aᵤ) (optimal step)",
    "gfdn_au_as" => "AS-inexact GFDN(aᵤ) (optimal step)",
    "cg_gfdn_au_as" => "AS-inexact CG-GFDN(aᵤ) (optimal step)",
  )
  markers = Dict(
    "gp_additive" => :circle,
    "gp_additive_history" => :hexagon,
    "gfdn_au_exact" => :diamond,
    "cg_gfdn_au_exact" => :cross,
    "gfdn_au_as" => :rect,
    "cg_gfdn_au_as" => :utriangle,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  positive_residuals = tbl.resnorm[tbl.resnorm .> 0]
  ylimits = (1e-7, maximum(positive_residuals) * 2)
  ytick_exps = sort(
    collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1])))
  )
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
        (tbl.method .== method) .&
        (tbl.beta .== beta) .&
        (tbl.m .== m) .&
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
    add_gp_solution_inset!(
      fig[row, column], solutions, beta; halign=0.68, inset_size=0.23
    )
    add_partition_inset!(
      fig[row, column],
      parts,
      m;
      halign=0.99,
      inset_size=0.23,
      inset_title="partition",
    )
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

function fig16_gp_energy_gap()
  tbl = loadtable("study10_gp_conv.csv")
  parts = loadtable("study10_partitions.csv")
  solutions = loadtable("study10_gp_solutions.csv")
  betas = sort(unique(tbl.beta))
  ms = sort(unique(tbl.m))
  methods = (
    "gp_additive",
    "gp_additive_history",
    "gfdn_au_exact",
    "cg_gfdn_au_exact",
    "gfdn_au_as",
    "cg_gfdn_au_as",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gfdn_au_exact" => "exact GFDN(aᵤ) (optimal step)",
    "cg_gfdn_au_exact" => "exact CG-GFDN(aᵤ) (optimal step)",
    "gfdn_au_as" => "AS-inexact GFDN(aᵤ) (optimal step)",
    "cg_gfdn_au_as" => "AS-inexact CG-GFDN(aᵤ) (optimal step)",
  )
  markers = Dict(
    "gp_additive" => :circle,
    "gp_additive_history" => :hexagon,
    "gfdn_au_exact" => :diamond,
    "cg_gfdn_au_exact" => :cross,
    "gfdn_au_as" => :rect,
    "cg_gfdn_au_as" => :utriangle,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  positive_gaps = tbl.energy_gap[tbl.energy_gap .> 0]
  ylimits = (max(minimum(positive_gaps) / 2, 1e-14), maximum(positive_gaps) * 2)
  ytick_exps = sort(
    collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1])))
  )
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
        (tbl.method .== method) .&
        (tbl.beta .== beta) .&
        (tbl.m .== m) .&
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
    add_gp_solution_inset!(
      fig[row, column], solutions, beta; halign=0.68, inset_size=0.23
    )
    add_partition_inset!(
      fig[row, column],
      parts,
      m;
      halign=0.99,
      inset_size=0.23,
      inset_title="partition",
    )
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
  xs = collect(range(-8 + 16 / N, 8 - 16 / N; length=N - 1))
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
      vertical=false,
      label="density |u|²",
      width=Relative(0.85),
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
    "nonlinear_ras",
    "anderson_ras",
    "newton_pcg_as_4",
    "newton_pcg_as_8",
    "energy_imex_pcg_as",
    "aspin",
    "raspen",
    "var_dd",
    "var_dd_history",
  )
  labels = Dict(
    "nonlinear_as" => "nAS + optimal damping",
    "nonlinear_ras" => "nRAS + optimal damping",
    "anderson_ras" => "Anderson–RAS (q = 4)",
    "newton_pcg_as_4" => "Newton–PCG(AS, 4)",
    "newton_pcg_as_8" => "Newton–PCG(AS, 8)",
    "energy_imex_pcg_as" => "energy-IMEX–PCG(AS), Δt = 1",
    "aspin" => "ASPIN",
    "raspen" => "RASPEN",
    "var_dd" => "varDD",
    "var_dd_history" => "varDD + history (1)",
  )
  markers = Dict(
    "nonlinear_as" => :circle,
    "nonlinear_ras" => :rect,
    "anderson_ras" => :cross,
    "newton_pcg_as_4" => :dtriangle,
    "newton_pcg_as_8" => :hexagon,
    "energy_imex_pcg_as" => :star5,
    "aspin" => :xcross,
    "raspen" => :pentagon,
    "var_dd" => :diamond,
    "var_dd_history" => :utriangle,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  positive_residuals = conv.relative_residual[conv.relative_residual .> 0]
  ylimits = (1e-8, maximum(positive_residuals) * 2)
  ytick_exps = sort(
    collect(floor(Int, log10(ylimits[2])):-2:ceil(Int, log10(ylimits[1])))
  )

  fig = Figure(size=(max(990, 330 * length(ms)), 540))
  Label(
    fig[0, 1:length(ms)],
    "−Δu + βu³ = f  in Ω,    β = 1,    u = 0  on ∂Ω";
    fontsize=22,
    font=:bold,
  )
  for (column, m) in enumerate(ms)
    ax = Axis(
      fig[1, column];
      xlabel="outer iteration",
      ylabel=column == 1 ? "relative residual" : "",
      title="m = $m",
      yscale=log10,
      yticks=LogTicks(ytick_exps),
      limits=(nothing, ylimits),
    )
    for (index, method) in enumerate(methods)
      mask =
        (conv.m .== m) .&
        (conv.method .== method) .&
        (conv.relative_residual .> 0)
      add_series!(
        ax,
        pick(conv, :outer, mask),
        logfloor(pick(conv, :relative_residual, mask); floor=1e-16);
        label=labels[method],
        color=colors[index],
        marker=markers[method],
      )
    end
    hlines!(ax, [1e-7]; color=:black, linestyle=:dot, linewidth=1.2)
    add_semilinear_solution_inset!(
      fig[1, column], solutions; halign=0.68, inset_size=0.23
    )
    add_triangle_partition_inset!(
      fig[1, column],
      parts,
      m;
      halign=0.99,
      inset_size=0.23,
      inset_title="partition",
    )
  end
  Legend(
    fig[2, 1:length(ms)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation=:horizontal,
    nbanks=3,
    framevisible=true,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig17_semilinear_poisson")
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
  methods = [
    "var_dd_additive",
    "var_dd_additive_history",
    "pcg_as",
    "gmres_ras",
  ]
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
  colors = Makie.resample_cmap(:tab10, length(methods))
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
  fig = Figure(size=(430 * length(panels), 390))
  for (column, (experiment, regime, parameter, title)) in enumerate(panels)
    ax = Axis(
      fig[1, column];
      xlabel=parameter == :N ? "elements per direction, 1/h" : "overlap layers ℓ",
      ylabel=column == 1 ? "parallel local-solve batches" : "",
      title,
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== experiment) .&
        (tbl.regime .== regime) .&
        (tbl.method .== method)
      xs, ys = study8_terminal_series(tbl, mask, parameter)
      add_series!(
        ax, xs, ys;
        label=labels[method], color=colors[index], marker=markers[method],
      )
    end
  end
  Legend(
    fig[2, 1:length(panels)],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation=:horizontal,
    nbanks=1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig18_poisson_scaling")
end

function fig19_poisson_contrast()
  tbl = loadtable("study8_sensitivity.csv")
  inner = loadtable("study8_inner_systems.csv")
  methods = [
    "var_dd_additive",
    "var_dd_additive_history",
    "pcg_as",
    "gmres_ras",
  ]
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
  colors = Makie.resample_cmap(:tab10, length(methods))
  fig = Figure(size=(1420, 410))
  ax_iterations = Axis(
    fig[1, 1];
    xlabel="diffusion contrast κ",
    ylabel="parallel local-solve batches to tolerance",
    xscale=log10,
    title="outer convergence",
  )
  for (index, method) in enumerate(methods)
    mask = (tbl.experiment .== "contrast") .& (tbl.method .== method)
    xs, ys = study8_terminal_series(tbl, mask, :contrast)
    add_series!(
      ax_iterations, xs, ys;
      label=labels[method], color=colors[index], marker=markers[method],
    )
  end

  ax_condition = Axis(
    fig[1, 2];
    xlabel="diffusion contrast κ",
    ylabel="estimated κ₂",
    xscale=log10,
    yscale=log10,
    title="first-sweep condition estimates",
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
    system_mask =
      (inner.experiment .== "contrast") .&
      (inner.system .== system)
    contrasts = sort(unique(inner.contrast[system_mask]))
    maxima = [
      maximum(inner.condition_estimate[
        system_mask .& (inner.contrast .== contrast)
      ]) for contrast in contrasts
    ]
    add_series!(
      ax_condition, contrasts, maxima;
      label,
      color,
      marker,
      linestyle,
    )
  end
  axislegend(ax_condition; position=:lt)

  ax_inner = Axis(
    fig[1, 3];
    xlabel="diffusion contrast κ",
    ylabel="local CG iterations",
    xscale=log10,
    title="iterative local-solve work (censored)",
    limits=(nothing, (0, 2200)),
  )
  mask =
    (inner.experiment .== "contrast") .&
    (inner.system .== "schwarz_local")
  contrasts = sort(unique(inner.contrast[mask]))
  medians = Float64[]
  maxima = Float64[]
  for contrast in contrasts
    values = inner.cg_iterations[mask .& (inner.contrast .== contrast)]
    push!(medians, median(values))
    push!(maxima, maximum(values))
  end
  add_series!(
    ax_inner, contrasts, medians;
    label="median subdomain", color=PALETTE[5], marker=:rect,
  )
  add_series!(
    ax_inner, contrasts, maxima;
    label="most difficult subdomain", color=PALETTE[6], marker=:diamond,
  )
  iteration_cap = 2000
  censored = unique(vcat(
    contrasts[medians .>= iteration_cap],
    contrasts[maxima .>= iteration_cap],
  ))
  hlines!(
    ax_inner,
    [iteration_cap];
    color=(:black, 0.55),
    linestyle=:dot,
    linewidth=1.4,
  )
  scatter!(
    ax_inner,
    censored,
    fill(iteration_cap, length(censored));
    label="censored (≥ 2000)",
    color=:black,
    marker=:utriangle,
    markersize=14,
  )
  axislegend(ax_inner; position=:lt)
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, markers; colors),
    [labels[method] for method in methods];
    orientation=:horizontal,
    nbanks=1,
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
  fig = Figure(size=(850, 370))
  Label(
    fig[0, 1:2],
    "One previous iterate provides nearly all of the history benefit";
    fontsize=22,
    font=:bold,
  )
  for (column, regime) in enumerate(regimes)
    mask =
      (tbl.experiment .== "history") .&
      (tbl.regime .== regime)
    depths, batches = study8_terminal_series(tbl, mask, :history_depth)
    ax_batches = Axis(
      fig[1, column];
      xlabel="history depth q",
      ylabel=column == 1 ? "parallel local-solve batches to tolerance" : "",
      title=titles[regime],
    )
    add_series!(
      ax_batches, depths, batches;
      color=PALETTE[2], marker=:hexagon,
    )
    vlines!(ax_batches, [1]; color=(:black, 0.45), linestyle=:dash)
  end
  savefigs(fig, "fig20_poisson_history")
end

# ---------------------------------------------------------------------------
# Figs 21--23: semilinear sensitivity studies
# ---------------------------------------------------------------------------

const SEMILINEAR_BENCHMARK_LABELS = Dict(
  "nonlinear_ras" => "nonlinear RAS",
  "anderson_ras" => "Anderson–RAS (q = 4)",
  "newton_pcg_as" => "Newton–PCG(AS, 4)",
  "energy_imex_pcg_as" => "energy-IMEX–PCG(AS)",
  "aspin" => "ASPIN",
  "raspen" => "RASPEN",
  "var_dd" => "varDD",
  "var_dd_history" => "varDD + history",
)

const SEMILINEAR_BENCHMARK_METHODS = (
  "nonlinear_ras",
  "anderson_ras",
  "newton_pcg_as",
  "energy_imex_pcg_as",
  "aspin",
  "raspen",
  "var_dd",
  "var_dd_history",
)

const SEMILINEAR_BENCHMARK_MARKERS = Dict(
  "nonlinear_ras" => :rect,
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
  colors = Makie.resample_cmap(:tab10, length(methods))
  fig = Figure(size=(1120, 500))
  Label(
    fig[0, 1:3],
    "Mesh refinement h, h/2, h/4 with fixed physical overlap";
    fontsize=22,
    font=:bold,
  )
  specifications = (
    (:outer_iterations, "outer iterations"),
    (:nonlinear_local_batches, "nonlinear local batches"),
    (:linear_as_batches, "linear AS batches"),
  )
  for (column, (quantity, ylabel)) in enumerate(specifications)
    ax = Axis(fig[1, column]; xlabel="cells per coordinate direction N", ylabel)
    for (index, method) in enumerate(methods)
      mask = mask_mesh .& (tbl.method .== method)
      any(mask) || continue
      values = pick(tbl, quantity, mask)
      all(iszero, values) && quantity != :outer_iterations && continue
      add_series!(
        ax,
        pick(tbl, :N, mask),
        values;
        color=colors[index],
        marker=SEMILINEAR_BENCHMARK_MARKERS[method],
      )
    end
    ax.xticks = Ns
  end
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, SEMILINEAR_BENCHMARK_MARKERS; colors),
    [SEMILINEAR_BENCHMARK_LABELS[method] for method in methods];
    orientation=:horizontal,
    nbanks=2,
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
  outer = [only(tbl.outer_iterations[mask .& (tbl.parameter .== p)]) for p in parameters]
  batches = [only(tbl.linear_as_batches[mask .& (tbl.parameter .== p)]) for p in parameters]
  converged = [only(tbl.converged[mask .& (tbl.parameter .== p)]) == 1 for p in parameters]
  fig = Figure(size=(780, 350))
  Label(
    fig[0, 1:2],
    "Inexact Newton trades outer steps for inner PCG(AS) work";
    fontsize=22,
    font=:bold,
  )
  for (column, (values, ylabel)) in enumerate((
    (outer, "outer Newton iterations"),
    (batches, "total linear AS batches"),
  ))
    ax = Axis(
      fig[1, column];
      xlabel="PCG(AS) steps per Newton update",
      ylabel,
      xticks=(positions, labels),
    )
    add_series!(ax, positions, values; color=PALETTE[3], marker=:dtriangle)
    failed = positions[.!converged]
    if !isempty(failed)
      scatter!(
        ax,
        failed,
        values[.!converged];
        color=:black,
        marker=:utriangle,
        markersize=14,
        label="iteration budget reached",
      )
      column == 1 && axislegend(ax; position=:rt)
    end
  end
  savefigs(fig, "fig22_semilinear_newton_inner")
end

function fig23_semilinear_history()
  tbl = loadtable("study11_semilinear_sensitivity.csv")
  algorithms = ("anderson_ras", "var_dd_history")
  titles = ("Anderson–RAS", "varDD")
  fig = Figure(size=(780, 350))
  Label(
    fig[0, 1:2],
    "Effect of multisecant and iterate history depth";
    fontsize=22,
    font=:bold,
  )
  for (column, (method, title)) in enumerate(zip(algorithms, titles))
    mask = (tbl.experiment .== "history") .& (tbl.method .== method)
    order = sortperm(tbl.parameter[mask])
    depths = tbl.parameter[mask][order]
    iterations = tbl.outer_iterations[mask][order]
    ax = Axis(
      fig[1, column];
      xlabel="history depth q",
      ylabel=column == 1 ? "outer iterations to tolerance" : "",
      title,
      xticks=depths,
    )
    add_series!(ax, depths, iterations; color=PALETTE[column+1], marker=:hexagon)
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
  "jd_gmres_as" => "JD–GMRES(AS)",
  "si_lanczos_pcg_as" => "SI-Lanczos–PCG(AS)",
)

const EVP_SENSITIVITY_METHODS = (
  "var_dd",
  "var_dd_history",
  "lobpcg_as",
  "jd_gmres_as",
  "si_lanczos_pcg_as",
)

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
  colors = Makie.resample_cmap(:tab10, length(methods))
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
  fig = Figure(size=(1320, 420))
  for (column, (experiment, regime, parameter, title)) in enumerate(panels)
    ax = Axis(
      fig[1, column];
      xlabel=parameter == :N ? "elements per direction, 1/h" : "frequency ν",
      ylabel=column == 1 ? "outer iterations to tolerance" : "",
      title,
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== experiment) .&
        (tbl.regime .== regime) .&
        (tbl.method .== method)
      xs, ys = evp_terminal_series(tbl, mask, parameter, :iteration)
      add_series!(
        ax, xs, ys;
        color=colors[index], marker=EVP_SENSITIVITY_MARKERS[method],
      )
    end
  end
  Legend(
    fig[2, 1:3],
    legend_line_marker_elements(methods, EVP_SENSITIVITY_MARKERS; colors),
    [EVP_SENSITIVITY_LABELS[method] for method in methods];
    orientation=:horizontal,
    nbanks=1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig24_evp_scaling")
end

function fig25_evp_linear_work()
  tbl = loadtable("study9_evp_sensitivity.csv")
  methods = ["lobpcg_as", "jd_gmres_as", "si_lanczos_pcg_as"]
  colors = Makie.resample_cmap(:tab10, length(methods))
  fig = Figure(size=(900, 390))
  Label(
    fig[0, 1:2],
    "Linear-preconditioner work is separate from varDD local eigenproblems";
    fontsize=22,
    font=:bold,
  )
  for (column, (quantity, ylabel)) in enumerate((
    (:linear_as_batches, "parallel linear AS batches"),
    (:global_operator_products, "global K/M operator applications"),
  ))
    ax = Axis(
      fig[1, column];
      xlabel="elements per direction, 1/h",
      ylabel,
      title=column == 1 ? "local linear solves" : "global operator work",
    )
    for (index, method) in enumerate(methods)
      mask =
        (tbl.experiment .== "mesh") .&
        (tbl.regime .== "fixed_delta_over_H") .&
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
        ax, xs, ys;
        color=colors[index], marker=EVP_SENSITIVITY_MARKERS[method],
      )
    end
  end
  Legend(
    fig[2, 1:2],
    legend_line_marker_elements(methods, EVP_SENSITIVITY_MARKERS; colors),
    [EVP_SENSITIVITY_LABELS[method] for method in methods];
    orientation=:horizontal,
    nbanks=1,
  )
  rowgap!(fig.layout, 8)
  savefigs(fig, "fig25_evp_linear_work")
end

function fig26_evp_history()
  tbl = loadtable("study9_evp_sensitivity.csv")
  combination = loadtable("study9_evp_sensitivity_combination.csv")
  mask =
    (tbl.experiment .== "history") .&
    (tbl.method .== "var_dd_history")
  depths, iterations = evp_terminal_series(
    tbl, mask, :history_depth, :iteration
  )
  conditions = Float64[]
  for depth in depths
    condition_mask =
      (combination.experiment .== "history") .&
      (combination.method .== "var_dd_history") .&
      (combination.history_depth .== depth)
    push!(conditions, maximum(combination.mass_condition[condition_mask]))
  end
  fig = Figure(size=(820, 360))
  Label(
    fig[0, 1:2],
    "One previous iterate supplies the useful EVP history enrichment";
    fontsize=22,
    font=:bold,
  )
  ax_iterations = Axis(
    fig[1, 1]; xlabel="history depth q", ylabel="outer iterations to tolerance"
  )
  add_series!(ax_iterations, depths, iterations; color=PALETTE[2], marker=:hexagon)
  vlines!(ax_iterations, [1]; color=(:black, 0.45), linestyle=:dash)
  ax_condition = Axis(
    fig[1, 2];
    xlabel="history depth q",
    ylabel="max κ₂(QᵀMQ)",
    yscale=log10,
  )
  add_series!(ax_condition, depths, conditions; color=PALETTE[5], marker=:diamond)
  vlines!(ax_condition, [1]; color=(:black, 0.45), linestyle=:dash)
  savefigs(fig, "fig26_evp_history")
end

function fig27_evp_local_work()
  tbl = loadtable("study9_evp_sensitivity.csv")
  local_stats = loadtable("study9_evp_sensitivity_local.csv")
  Ns = sort(unique(tbl.N[
    (tbl.experiment .== "mesh") .& (tbl.regime .== "fixed_delta_over_H")
  ]))
  fig = Figure(size=(1260, 390))
  Label(
    fig[0, 1:3],
    "Local-system size, critical path, and factor storage";
    fontsize=22,
    font=:bold,
  )

  ax_dimension = Axis(
    fig[1, 1]; xlabel="elements per direction, 1/h", ylabel="maximum local dimension"
  )
  for (system, label, color, marker) in (
    ("schwarz_block", "AS block", PALETTE[4], :rect),
    ("vardd_augmented_pencil", "varDD augmented pencil", PALETTE[2], :circle),
  )
    values = Float64[]
    for N in Ns
      mask =
        (local_stats.experiment .== "mesh") .&
        (local_stats.regime .== "fixed_delta_over_H") .&
        (local_stats.N .== N) .&
        (local_stats.system .== system) .&
        (system == "schwarz_block" ? trues(length(local_stats.N)) :
         local_stats.method .== "var_dd")
      push!(values, maximum(local_stats.dimension[mask]))
    end
    add_series!(ax_dimension, Ns, values; label, color, marker)
  end
  axislegend(ax_dimension; position=:lt)

  ax_critical = Axis(
    fig[1, 2];
    xlabel="elements per direction, 1/h",
    ylabel="critical-path local LOBPCG iterations",
  )
  for (method, color, marker) in (
    ("var_dd", PALETTE[1], :circle),
    ("var_dd_history", PALETTE[2], :hexagon),
  )
    mask =
      (tbl.experiment .== "mesh") .&
      (tbl.regime .== "fixed_delta_over_H") .&
      (tbl.method .== method)
    xs, ys = evp_terminal_series(
      tbl, mask, :N, :local_iterations_critical
    )
    add_series!(
      ax_critical, xs, ys;
      label=EVP_SENSITIVITY_LABELS[method], color, marker,
    )
  end
  axislegend(ax_critical; position=:lt)

  ax_factor = Axis(
    fig[1, 3];
    xlabel="elements per direction, 1/h",
    ylabel="maximum local factor nnz",
    yscale=log10,
  )
  for (system, label, color, marker) in (
    ("schwarz_block", "AS block", PALETTE[4], :rect),
    ("vardd_augmented_pencil", "varDD augmented pencil", PALETTE[2], :circle),
  )
    values = Float64[]
    for N in Ns
      mask =
        (local_stats.experiment .== "mesh") .&
        (local_stats.regime .== "fixed_delta_over_H") .&
        (local_stats.N .== N) .&
        (local_stats.system .== system) .&
        (system == "schwarz_block" ? trues(length(local_stats.N)) :
         local_stats.method .== "var_dd")
      push!(values, maximum(local_stats.factor_nnz[mask]))
    end
    add_series!(ax_factor, Ns, values; label, color, marker)
  end
  axislegend(ax_factor; position=:lt)
  savefigs(fig, "fig27_evp_local_work")
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
  fig15_gp_ground_states()
  fig16_gp_energy_gap()
  fig17_semilinear_poisson()
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
  println("plots: done -> $(FIG_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
  make_all_figures()
end
