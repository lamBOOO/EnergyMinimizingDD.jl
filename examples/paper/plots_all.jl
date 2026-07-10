# Regenerates ALL Section 4 figures from data/*.csv (no solves).
# Run: julia --project=. examples/paper/plots_all.jl

isdefined(Main, :PAPER_COMMON) || include("common.jl")

using Plots
using Plots.PlotMeasures

default(
  fontfamily = "Computer Modern",
  linewidth = 2.5,
  framestyle = :box,
  grid = true,
  gridalpha = 0.12,
  labelfontsize = 12,
  tickfontsize = 10,
  legendfontsize = 10,
  titlefontsize = 13,
  dpi = 300,
)

"Save a figure as PDF (vector, for LaTeX) and PNG (preview) into figures/."
function savefigs(fig, name)
  savefig(fig, joinpath(FIG_DIR, name * ".pdf"))
  savefig(fig, joinpath(FIG_DIR, name * ".png"))
  println("  saved $(name).{pdf,png}")
  return fig
end

# Common marker style for convergence panels
const MSTYLE =
  (marker = :circle, markersize = 3.5, markerstrokewidth = 0)

# Select entries of column `col` where `mask` holds (columntable helper)
pick(tbl, col, mask) = collect(getproperty(tbl, col)[mask])

# ---------------------------------------------------------------------------
# Fig 1: problem setup illustration (partition, overlap, solutions)
# ---------------------------------------------------------------------------
function fig01_setup()
  part = loadtable("study1_partition.csv")
  gs = loadtable("study1_schroedinger.csv")
  plap = loadtable("study1_plap.csv")

  Npart = part.N[1]
  xs = interior_nodes(Npart)
  nsub = maximum(part.owner)

  p1 = heatmap(
    xs,
    xs,
    field_matrix(float.(part.owner), Npart);
    title = "METIS partition (owner)",
    color = cgrad(:tab10, nsub; categorical = true),
    colorbar = false,
    aspect_ratio = :equal,
    xlims = (0, 1),
    ylims = (0, 1),
    xlabel = "x1",
    ylabel = "x2",
  )
  p2 = heatmap(
    xs,
    xs,
    field_matrix(float.(part.mult), Npart);
    title = "overlap multiplicity",
    color = cgrad(:viridis, maximum(part.mult); categorical = true),
    colorbar_ticks = collect(1:maximum(part.mult)),
    aspect_ratio = :equal,
    xlims = (0, 1),
    ylims = (0, 1),
    xlabel = "x1",
    ylabel = "x2",
  )
  Ngs = gs.N[1]
  p3 = heatmap(
    all_nodes(Ngs),
    all_nodes(Ngs),
    field_matrix_with_bc(gs.value, Ngs);
    title = "Schroedinger ground state",
    color = :viridis,
    aspect_ratio = :equal,
    xlims = (0, 1),
    ylims = (0, 1),
    xlabel = "x1",
    ylabel = "x2",
  )
  Npl = plap.N[1]
  p4 = heatmap(
    all_nodes(Npl),
    all_nodes(Npl),
    field_matrix_with_bc(plap.value, Npl);
    title = "p-Laplacian solution (p = 3)",
    color = :viridis,
    aspect_ratio = :equal,
    xlims = (0, 1),
    ylims = (0, 1),
    xlabel = "x1",
    ylabel = "x2",
  )
  fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (880, 780), margin = 3mm)
  savefigs(fig, "fig01_setup")
end

# ---------------------------------------------------------------------------
# Fig 2: EVP convergence histories (sweep m, sweep overlap)
# ---------------------------------------------------------------------------
function conv_panel(tbl, values, labelfn; kwargs...)
  p = plot(;
    xlabel = "iteration",
    ylabel = "eigenvalue error",
    yscale = :log10,
    legend = :topright,
    kwargs...,
  )
  for v in values
    mask = (tbl.param .== v) .& (tbl.err .> 1e-13)
    plot!(
      p,
      pick(tbl, :iter, mask),
      logfloor(pick(tbl, :err, mask));
      label = labelfn(v),
      MSTYLE...,
    )
  end
  return p
end

function fig02_evp_convergence()
  tm = loadtable("study2_sweep_m.csv")
  to = loadtable("study2_sweep_olap.csv")
  p1 =
    conv_panel(tm, sort(unique(tm.param)), v -> "m = $v"; title = "overlap = 2")
  p2 = conv_panel(
    to,
    sort(unique(to.param)),
    v -> "overlap = $v";
    title = "m = 6",
  )
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig02_evp_convergence")
end

# ---------------------------------------------------------------------------
# Fig 3: EVP vs inverse-iteration baseline
# ---------------------------------------------------------------------------
function fig03_evp_baseline()
  tbl = loadtable("study2_baseline.csv")
  fig = plot(;
    xlabel = "iteration",
    ylabel = "eigenvalue error",
    yscale = :log10,
    legend = :topright,
    size = (600, 400),
  )
  for (method, label) in
      (("var_dd", "variational DD"), ("inverse_iteration", "inverse iteration"))
    mask = (tbl.method .== method) .& (tbl.err .> 1e-13)
    plot!(
      fig,
      pick(tbl, :iter, mask),
      logfloor(pick(tbl, :err, mask));
      label = label,
      MSTYLE...,
    )
  end
  savefigs(fig, "fig03_evp_baseline")
end

# ---------------------------------------------------------------------------
# Fig 4: FEM accuracy of the converged eigenvalue vs h
# ---------------------------------------------------------------------------
function fig04_evp_haccuracy()
  tbl = loadtable("study2_haccuracy.csv")
  hs = collect(tbl.h)
  errs = collect(tbl.err)
  fig = plot(
    hs,
    errs;
    xscale = :log10,
    yscale = :log10,
    xlabel = "mesh size h",
    ylabel = "eigenvalue error vs exact",
    label = "var_dd (converged)",
    legend = :topleft,
    size = (600, 400),
    MSTYLE...,
  )
  # O(h^2) reference line anchored at the finest mesh
  href = [minimum(hs), maximum(hs)]
  eref = errs[argmin(hs)] .* (href ./ minimum(hs)) .^ 2
  plot!(
    fig,
    href,
    eref;
    label = "quadratic reference",
    linestyle = :dash,
    linewidth = 1.5,
    color = :black,
  )
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

  p1 = plot(;
    xlabel = "number of subdomains m",
    ylabel = "iterations",
    title = "overlap = 2",
    legend = :topleft,
    xticks = ms,
  )
  for (ci, N) in enumerate(Ns), (problem, (ls, plabel)) in PROBLEM_STYLE
    mask = (tbl.problem .== problem) .& (tbl.N .== N) .& (tbl.overlap .== 2)
    plot!(
      p1,
      pick(tbl, :m, mask),
      pick(tbl, :iters, mask);
      label = "$plabel, N = $N",
      color = ci,
      linestyle = ls,
      MSTYLE...,
    )
  end

  olaps = sort(unique(tbl.overlap))
  p2 = plot(;
    xlabel = "overlap",
    ylabel = "iterations",
    title = "N = $N_fix",
    legend = :topright,
    xticks = olaps,
  )
  for (ci, m) in enumerate(ms), (problem, (ls, plabel)) in PROBLEM_STYLE
    mask = (tbl.problem .== problem) .& (tbl.N .== N_fix) .& (tbl.m .== m)
    plot!(
      p2,
      pick(tbl, :overlap, mask),
      pick(tbl, :iters, mask);
      label = "$plabel, m = $m",
      color = ci,
      linestyle = ls,
      MSTYLE...,
    )
  end
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig05_robustness")
end

# ---------------------------------------------------------------------------
# Fig 6: timing (overlap = 2 slice)
# ---------------------------------------------------------------------------
function fig06_timing()
  tbl = loadtable("study3_robustness.csv")
  ms = sort(unique(tbl.m))
  Nticks = sort(unique(tbl.N))
  p1 = plot(;
    xlabel = "mesh parameter N",
    ylabel = "wall-clock time [s]",
    xscale = :log10,
    yscale = :log10,
    xticks = (Nticks, string.(Nticks)),
    legend = :topleft,
    title = "time per solve",
  )
  p2 = plot(;
    xlabel = "mesh parameter N",
    ylabel = "time per iteration [s]",
    xscale = :log10,
    yscale = :log10,
    xticks = (Nticks, string.(Nticks)),
    legend = :topleft,
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
    plot!(p1, Ns, ts; label = "$plabel, m = $m", color = ci, linestyle = ls, MSTYLE...)
    plot!(p2, Ns, ts ./ its; label = "$plabel, m = $m", color = ci, linestyle = ls, MSTYLE...)
  end
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig06_timing")
end

# ---------------------------------------------------------------------------
# Fig 7: Poisson convergence histories
# ---------------------------------------------------------------------------
function fig07_poisson()
  tbl = loadtable("study4_poisson.csv")
  ms = sort(unique(tbl.m))
  p1 = plot(;
    xlabel = "iteration",
    ylabel = "energy gap",
    yscale = :log10,
    legend = :topright,
  )
  p2 = plot(;
    xlabel = "iteration",
    ylabel = "residual norm",
    yscale = :log10,
    legend = :topright,
  )
  for m in ms
    mask_e =
      (tbl.m .== m) .& (tbl.quantity .== "energy_gap") .& (tbl.value .> 1e-14)
    mask_r =
      (tbl.m .== m) .& (tbl.quantity .== "resnorm") .& (tbl.value .> 1e-14)
    plot!(
      p1,
      pick(tbl, :iter, mask_e),
      logfloor(pick(tbl, :value, mask_e));
      label = "m = $m",
      MSTYLE...,
    )
    plot!(
      p2,
      pick(tbl, :iter, mask_r),
      logfloor(pick(tbl, :value, mask_r));
      label = "m = $m",
      MSTYLE...,
    )
  end
  hline!(
    p2,
    [1e-12];
    label = "tolerance",
    linestyle = :dash,
    linewidth = 1.5,
    color = :black,
  )
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig07_poisson")
end

# ---------------------------------------------------------------------------
# Fig 8: p-Laplacian convergence and accuracy
# ---------------------------------------------------------------------------
function fig08_plaplacian()
  conv = loadtable("study5_conv.csv")
  summ = loadtable("study5_summary.csv")
  ps = sort(unique(conv.p))
  p1 = plot(;
    xlabel = "iteration",
    ylabel = "energy gap to Newton reference",
    yscale = :log10,
    legend = :topright,
  )
  for p in ps
    mask = (conv.p .== p) .& (abs.(conv.energy_gap) .> 1e-13)
    plot!(
      p1,
      pick(conv, :iter, mask),
      logfloor(abs.(pick(conv, :energy_gap, mask)));
      label = "p = $p",
      MSTYLE...,
    )
  end
  p2 = scatter(
    collect(summ.p),
    collect(summ.relerr);
    yscale = :log10,
    xlabel = "exponent p",
    ylabel = "rel. dof error vs Newton",
    label = "",
    markersize = 7,
    markerstrokewidth = 0,
    xticks = collect(summ.p),
    xlims = (minimum(summ.p) - 0.5, maximum(summ.p) + 0.5),
    ylims = (1e-7, 1e-4),
  )
  for (pv, it, re) in zip(summ.p, summ.iters, summ.relerr)
    annotate!(p2, pv, re * 3, text("$it iters", 9, :center))
  end
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig08_plaplacian")
end

# ---------------------------------------------------------------------------
# Fig 9: heat equation — warm vs cold start and tau sweep
# ---------------------------------------------------------------------------
function fig09_heat_warmstart()
  wc = loadtable("study6_warmcold.csv")
  ts = loadtable("study6_tau.csv")

  p1 = plot(;
    xlabel = "time step n",
    ylabel = "DD iterations per step",
    legend = :right,
  )
  for (mode, label) in (("cold", "cold start"), ("warm", "warm start"))
    mask = wc.mode .== mode
    plot!(
      p1,
      pick(wc, :step, mask),
      pick(wc, :iters, mask);
      label = label,
      MSTYLE...,
    )
  end
  ylims!(p1, 0, maximum(wc.iters) + 2)

  taus = sort(unique(ts.tau))
  totals = [sum(pick(ts, :iters, ts.tau .== tau)) for tau in taus]
  p2 = plot(
    taus,
    totals;
    xscale = :log10,
    xlabel = "time step size tau",
    ylabel = "total DD iterations (T = 0.2)",
    label = "",
    MSTYLE...,
  )
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig09_heat_warmstart")
end

# ---------------------------------------------------------------------------
# Fig 10: heat equation — dissipation and accuracy
# ---------------------------------------------------------------------------
function fig10_heat_dissipation()
  ts = loadtable("study6_tau.csv")
  taus = sort(unique(ts.tau))
  p1 = plot(;
    xlabel = "t",
    ylabel = "energy gap to steady state",
    yscale = :log10,
    legend = :topright,
  )
  p2 = plot(;
    xlabel = "t",
    ylabel = "rel. error vs direct solve",
    yscale = :log10,
    legend = :bottomright,
  )
  for tau in taus
    mask = ts.tau .== tau
    label = @sprintf("tau = %.1e", tau)
    plot!(
      p1,
      pick(ts, :t, mask),
      logfloor(pick(ts, :energy_gap, mask));
      label = label,
      MSTYLE...,
      markersize = 2.5,
    )
    plot!(
      p2,
      pick(ts, :t, mask),
      logfloor(pick(ts, :relerr, mask));
      label = label,
      MSTYLE...,
      markersize = 2.5,
    )
  end
  fig = plot(p1, p2; layout = (1, 2), size = (900, 350), margin = 5mm)
  savefigs(fig, "fig10_heat_dissipation")
end

# ---------------------------------------------------------------------------
# Fig 11: Schroedinger EVP — 3D series of iterate and subdomain updates
# ---------------------------------------------------------------------------
function fig11_local3d()
  tbl = loadtable("study7_local3d.csv")
  N = tbl.N[1]
  xs = all_nodes(N)
  ks = sort(unique(tbl.iter))
  subs = sort(unique(tbl.sub[tbl.kind.=="update"]))

  # include the zero Dirichlet boundary layer so surfaces reach the boundary
  surf(vals; title = "", zlims = :auto) = surface(
    xs,
    xs,
    field_matrix_with_bc(vals, N);
    title = title,
    color = :viridis,
    colorbar = false,
    xticks = false,
    yticks = false,
    zticks = true,
    zlims = zlims,
    titlefontsize = 11,
    camera = (40, 35),
  )

  panels = []
  for (ri, k) in enumerate(ks)
    mask_u = (tbl.iter .== k) .& (tbl.kind .== "solution")
    push!(
      panels,
      surf(pick(tbl, :value, mask_u); title = "iterate after sweep $k"),
    )
    # shared z-range across this sweep's updates (shows their decay)
    mask_all = (tbl.iter .== k) .& (tbl.kind .== "update")
    zmax = maximum(abs.(tbl.value[mask_all])) + 1e-12
    for s in subs
      mask = mask_all .& (tbl.sub .== s)
      push!(
        panels,
        surf(
          pick(tbl, :value, mask);
          title = ri == 1 ? "update, subdomain $s" : "",
          zlims = (-zmax, zmax),
        ),
      )
    end
  end
  ncols = 1 + length(subs)
  fig = plot(
    panels...;
    layout = (length(ks), ncols),
    size = (330 * ncols, 300 * length(ks)),
    margin = 1mm,
  )
  savefigs(fig, "fig11_local3d")
end

# ---------------------------------------------------------------------------
# Fig 12: Poisson -- comparison against one-level Schwarz baselines
# ---------------------------------------------------------------------------
function fig12_poisson_cmp()
  tbl = loadtable("study8_linear_cmp.csv")
  ms = sort(unique(tbl.m))
  labels = Dict(
    "var_dd" => "variational DD",
    "as" => "damped AS",
    "ras" => "RAS",
    "pcg_as" => "CG + AS",
  )
  panels = []
  for m in ms
    p = plot(;
      xlabel = "subdomain solves",
      ylabel = "residual norm",
      yscale = :log10,
      title = "m = $m",
      legend = m == first(ms) ? :topright : false,
    )
    for method in ("var_dd", "as", "ras", "pcg_as")
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.resnorm .> 1e-14)
      plot!(
        p,
        pick(tbl, :solves, mask),
        logfloor(pick(tbl, :resnorm, mask));
        label = labels[method],
        MSTYLE...,
      )
    end
    push!(panels, p)
  end
  fig = plot(
    panels...;
    layout = (1, length(panels)),
    size = (330 * length(panels), 350),
    margin = 4mm,
  )
  savefigs(fig, "fig12_poisson_cmp")
end

# ---------------------------------------------------------------------------
# Fig 13: EVP -- comparison against one-level LOBPCG+AS baseline
# ---------------------------------------------------------------------------
function fig13_evp_cmp()
  tbl = loadtable("study9_evp_cmp.csv")
  ms = sort(unique(tbl.m))
  labels = Dict(
    "var_dd" => "varDD",
    "lopsd_as" => "LOPSD+AS",
    "lobpcg_as" => "LOBPCG + AS",
  )
  panels = []
  for m in ms
    p = plot(;
      xlabel = "subdomain solves",
      ylabel = "eigenvalue error",
      yscale = :log10,
      title = "m = $m",
      legend = m == first(ms) ? :topright : false,
    )
    for method in ("var_dd", "lopsd_as", "lobpcg_as")
      mask = (tbl.m .== m) .& (tbl.method .== method) .& (tbl.err .> 1e-14)
      plot!(
        p,
        pick(tbl, :solves, mask),
        logfloor(pick(tbl, :err, mask));
        label = labels[method],
        MSTYLE...,
      )
    end
    push!(panels, p)
  end
  fig = plot(
    panels...;
    layout = (1, length(panels)),
    size = (330 * length(panels), 350),
    margin = 4mm,
  )
  savefigs(fig, "fig13_evp_cmp")
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
  println("plots: done -> $(FIG_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
  make_all_figures()
end
