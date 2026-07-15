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
    colorrange = (0, maximum(solutions.density)),
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

# ---------------------------------------------------------------------------
# Fig 1: problem setup illustration (partition, overlap, solutions)
# ---------------------------------------------------------------------------
function fig01_setup()
  part = loadtable("study1_partition.csv")
  gs = loadtable("study1_schroedinger.csv")
  plap = loadtable("study1_plap.csv")

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
    fig[2, 1];
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

  Npl = plap.N[1]
  ax4 = Axis(
    fig[2, 2];
    title = "p-Laplacian solution (p = 3)",
    xlabel = "x1",
    ylabel = "x2",
    aspect = DataAspect(),
    limits = (0, 1, 0, 1),
  )
  heatmap!(
    ax4,
    collect(all_nodes(Npl)),
    collect(all_nodes(Npl)),
    field_matrix_with_bc(plap.value, Npl);
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
# Fig 8: p-Laplacian convergence and accuracy
# ---------------------------------------------------------------------------
function fig08_plaplacian()
  conv = loadtable("study5_conv.csv")
  summ = loadtable("study5_summary.csv")
  ps = sort(unique(conv.p))
  fig = Figure(size = (900, 350))
  ax1 = Axis(
    fig[1, 1];
    xlabel = "iteration",
    ylabel = "energy gap to Newton reference",
    yscale = log10,
  )
  for (i, p) in enumerate(ps)
    mask = (conv.p .== p) .& (abs.(conv.energy_gap) .> 1e-13)
    add_series!(
      ax1,
      pick(conv, :iter, mask),
      logfloor(abs.(pick(conv, :energy_gap, mask)));
      label = "p = $p",
      color = color_for(i),
    )
  end
  add_legend!(ax1; position = :rt)

  ax2 = Axis(
    fig[1, 2];
    xlabel = "exponent p",
    ylabel = "rel. dof error vs Newton",
    yscale = log10,
    xticks = collect(summ.p),
    limits = ((minimum(summ.p) - 0.5, maximum(summ.p) + 0.5), (1e-7, 1e-4)),
  )
  scatter!(ax2, collect(summ.p), collect(summ.relerr); color = color_for(1), markersize = 12)
  for (pv, it, re) in zip(summ.p, summ.iters, summ.relerr)
    text!(ax2, pv, re * 3; text = "$it iters", align = (:center, :center), fontsize = 14)
  end
  savefigs(fig, "fig08_plaplacian")
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
    "var_dd_additive_history" => "additive varDD + previous",
    "var_dd_additive_mix_025" => "additive varDD, ω = 0.25",
    "var_dd_additive_mix_05" => "additive varDD, ω = 0.5",
    "var_dd_additive_mix_075" => "additive varDD, ω = 0.75",
    "var_dd_multiplicative" => "multiplicative varDD",
    "as" => "damped AS",
    "ras" => "RAS",
    "pcg_as" => "CG+AS",
  )
  methods = (
    "var_dd_additive",
    "var_dd_additive_history",
    "var_dd_additive_mix_025",
    "var_dd_additive_mix_05",
    "var_dd_additive_mix_075",
    "var_dd_multiplicative",
    "as",
    "ras",
    "pcg_as",
  )
  markers = Dict(
    "var_dd_additive" => :circle,
    "var_dd_additive_history" => :hexagon,
    "var_dd_additive_mix_025" => :cross,
    "var_dd_additive_mix_05" => :xcross,
    "var_dd_additive_mix_075" => :dtriangle,
    "var_dd_multiplicative" => :star5,
    "as" => :rect,
    "ras" => :utriangle,
    "pcg_as" => :diamond,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  finite_res = tbl.resnorm[tbl.resnorm .> 0]
  ylims = (1e-10 * 0.5, maximum(finite_res) * 3)
  ytick_exps = sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size = (330 * length(ms), 330))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "iteration",
      ylabel = j == 1 ? "residual norm ‖Axₖ-b‖₂" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = (nothing, ylims),
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

# ---------------------------------------------------------------------------
# Fig 13: EVP residual -- comparison against one-level LOPSD/LOBPCG+AS baselines
# ---------------------------------------------------------------------------
function fig13_evp_cmp()
  tbl = loadtable("study9_evp_cmp.csv")
  parts = loadtable("study9_partitions.csv")
  ms = sort(unique(tbl.m))
  labels = Dict(
    "var_dd" => "varDD",
    "var_dd_prev" => "varDD + previous",
    "var_dd_mix_025" => "varDD, ω = 0.25",
    "var_dd_mix_05" => "varDD, ω = 0.5",
    "var_dd_mix_075" => "varDD, ω = 0.75",
    "lopsd_as" => "LOPSD+AS",
    "lobpcg_as" => "LOBPCG+AS",
  )
  methods = (
    "var_dd",
    "var_dd_prev",
    "var_dd_mix_025",
    "var_dd_mix_05",
    "var_dd_mix_075",
    "lopsd_as",
    "lobpcg_as",
  )
  markers = Dict(
    "var_dd" => :circle,
    "var_dd_prev" => :hexagon,
    "var_dd_mix_025" => :cross,
    "var_dd_mix_05" => :star5,
    "var_dd_mix_075" => :dtriangle,
    "lopsd_as" => :rect,
    "lobpcg_as" => :utriangle,
  )
  colors = Makie.resample_cmap(:tab10, length(methods))
  finite_res = tbl.resnorm[.!isnan.(tbl.resnorm) .& (tbl.resnorm .> 0)]
  ylims = (1e-6 * 0.5, maximum(finite_res) * 3)
  ytick_exps = sort(collect(floor(Int, log10(ylims[2])):-2:ceil(Int, log10(ylims[1]))))
  yticks = LogTicks(ytick_exps)
  fig = Figure(size = (330 * length(ms), 330))
  for (j, m) in enumerate(ms)
    ax = Axis(
      fig[1, j];
      xlabel = "iteration",
      ylabel = j == 1 ? "residual norm ‖Axₖ-λₖxₖ‖₂" : "",
      yscale = log10,
      yticks = yticks,
      title = "m = $m",
      limits = (nothing, ylims),
    )
    for (i, method) in enumerate(methods)
      mask =
        (tbl.m .== m) .&
        (tbl.method .== method) .&
        .!isnan.(tbl.resnorm) .&
        (tbl.resnorm .> 0)
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
    "gfdn_au_as",
    "cg_gfdn_au_as",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gfdn_au_as" => "GFDN(aᵤ)+AS (optimal step)",
    "cg_gfdn_au_as" => "CG-GFDN(aᵤ)+AS (optimal step)",
  )
  markers = Dict(
    "gp_additive" => :circle,
    "gp_additive_history" => :hexagon,
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
      ylabel = column == 1 ? "β = $(Int(beta))\nresidual norm ‖rₖ‖₂" : "",
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
        pick(tbl, :solves, mask) ./ m,
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
    "gfdn_au_as",
    "cg_gfdn_au_as",
  )
  labels = Dict(
    "gp_additive" => "additive GP-varDD",
    "gp_additive_history" => "additive GP-varDD + history",
    "gfdn_au_as" => "GFDN(aᵤ)+AS (optimal step)",
    "cg_gfdn_au_as" => "CG-GFDN(aᵤ)+AS (optimal step)",
  )
  markers = Dict(
    "gp_additive" => :circle,
    "gp_additive_history" => :hexagon,
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
      ylabel = column == 1 ? "β = $(Int(beta))\nenergy gap E(uₖ)−E(u★)" : "",
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
        pick(tbl, :solves, mask) ./ m,
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
  xs = collect(interior_nodes(N))
  density_max = maximum(tbl.density)
  fig = Figure(size = (330 * length(betas), 330))
  for (column, beta) in enumerate(betas)
    mask = tbl.beta .== beta
    ax = Axis(
      fig[1, column];
      title = "β = $(Int(beta))",
      xlabel = "x₁",
      ylabel = column == 1 ? "x₂" : "",
      aspect = DataAspect(),
    )
    heatmap!(
      ax,
      xs,
      xs,
      field_matrix(pick(tbl, :density, mask), N);
      colormap = :viridis,
      colorrange = (0, density_max),
    )
  end
  Colorbar(
    fig[1, length(betas)+1];
    limits = (0, density_max),
    colormap = :viridis,
    label = "ground-state density |u|^2",
  )
  savefigs(fig, "fig15_gp_ground_states")
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
  fig08_plaplacian()
  fig09_heat_warmstart()
  fig10_heat_dissipation()
  fig11_local3d()
  fig12_poisson_cmp()
  fig13_evp_cmp()
  fig14_gp_convergence()
  fig15_gp_ground_states()
  fig16_gp_energy_gap()
  println("plots: done -> $(FIG_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
  make_all_figures()
end
