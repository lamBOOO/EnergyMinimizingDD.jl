using VariationalDD
using CairoMakie
using LinearAlgebra
using Printf

const FEM = VariationalDD.FEMDiscretizations
const E = VariationalDD.Energies
const S = VariationalDD.Solvers

const ASSET_DIR = joinpath(@__DIR__, "src", "assets")
const PURPLE = RGBf(0.42, 0.27, 0.65)
const ORANGE = RGBf(0.86, 0.42, 0.16)
const INK = RGBf(0.12, 0.15, 0.22)
const MUTED = RGBf(0.36, 0.41, 0.49)
const GRID = RGBf(0.88, 0.90, 0.93)

function enable_mathsf!()
  mathtex = CairoMakie.Makie.MathTeXEngine
  serif = mathtex.FontFamily("NewComputerModern")
  sans = mathtex.FontFamily("TeXGyreHeros")
  fonts = copy(serif.fonts)
  fonts[:sans] = sans.fonts[:regular]
  fonts[:sansbold] = sans.fonts[:bold]
  modifiers = deepcopy(serif.font_modifiers)
  modifiers[:sf] = Dict(
    :regular => :sans,
    :italic => :sans,
    :bold => :sansbold,
    :bolditalic => :sansbold,
    :math => :sans,
  )
  family = mathtex.FontFamily(
    fonts;
    font_mapping=serif.font_mapping,
    font_modifiers=modifiers,
    special_chars=serif.special_chars,
    slant_angle=serif.slant_angle,
    thickness=serif.thickness,
  )
  mathtex.set_texfont_family!(family)
  return nothing
end

CairoMakie.activate!(; type="png")
enable_mathsf!()
set_theme!(theme_latexfonts(); fontsize=17)

function field_with_boundary(u, N)
  values = zeros(N + 1, N + 1)
  values[2:N, 2:N] .= reshape(u, N - 1, N - 1)
  return values
end

function local_candidates(energy, u, subdomains)
  return [S.inf_step(energy, u, dofs) for dofs in subdomains]
end

function one_iteration(energy, u, subdomains)
  candidates = local_candidates(energy, u, subdomains)
  next = S.combine_step(energy, hcat(u, candidates...))
  return candidates, next
end

function fmt_energy(value)
  return @sprintf("%+.5e", value)
end

function make_iteration_figure(energy, subdomains, N)
  x = collect(range(0.0, 1.0; length=N + 1))
  interior = x[2:N]
  u = vec([
    0.10 * sinpi(xi) * sinpi(yi) + 0.06 * sinpi(3xi) * sinpi(2yi) for
    xi in interior, yi in interior
  ])
  candidates, next = one_iteration(energy, u, subdomains)

  current_field = field_with_boundary(u, N)
  candidate_fields = field_with_boundary.(candidates, N)
  update_fields = candidate_fields .- Ref(current_field)
  next_field = field_with_boundary(next, N)
  field_scale = maximum(abs, vcat(vec(current_field), vec(next_field)))
  field_colorrange = (-field_scale, field_scale)
  update_scale = maximum(abs, vcat(vec.(update_fields)...))
  update_colorrange = (-update_scale, update_scale)
  index_field = reshape(collect(1:length(u)), N - 1, N - 1)
  subdomain_masks = map(subdomains) do dofs
    mask = zeros(N + 1, N + 1)
    mask[2:N, 2:N] .= in.(index_field, Ref(Set(dofs)))
    return mask
  end

  fig = Figure(; size=(1800, 820), backgroundcolor=:white)
  Label(
    fig[1, 1:6],
    L"\mathcal{W}_i(\mathsf{u}^{(k)})=\mathrm{span}\{\mathsf{u}^{(k)}\}+\mathcal{V}_i,\qquad \mathsf{y}_i^{(k)}\in\mathrm{arg\,min}_{\mathsf{y}\in\mathcal{W}_i(\mathsf{u}^{(k)})}\;\mathsf{J}(\mathsf{y})";
    fontsize=25,
    color=INK,
    padding=(0, 0, 8, 4),
  )

  current_axis = Axis(
    fig[2, 1];
    title=L"\mathsf{u}^{(0)}\qquad \mathcal{E}(\mathsf{u}^{(0)})=%$(fmt_energy(energy(u)))",
    xlabel="x",
    ylabel="y",
    aspect=DataAspect(),
    titlecolor=INK,
    backgroundcolor=RGBf(0.97, 0.98, 0.99),
  )
  current_plot = heatmap!(
    current_axis,
    x,
    x,
    current_field;
    colormap=:vik,
    colorrange=field_colorrange,
  )

  Label(
    fig[2, 2],
    "LOCAL\nLEVEL\n→";
    fontsize=20,
    font=:bold,
    color=MUTED,
    lineheight=1.25,
  )

  local_grid = GridLayout()
  fig[2, 3] = local_grid
  Label(
    local_grid[1, 1:2],
    L"\text{four independent local updates}\qquad \mathsf{y}_i^{(0)}-\mathsf{u}^{(0)}";
    fontsize=21,
    font=:bold,
    color=INK,
    padding=(0, 0, 0, 8),
  )
  for i in 1:4
    row, column = cld(i, 2), mod1(i, 2)
    axis = Axis(
      local_grid[row + 1, column];
      title=L"\mathsf{y}_{%$i}^{(0)}\qquad \mathcal{E}(\mathsf{y}_{%$i}^{(0)})=%$(fmt_energy(energy(candidates[i])))",
      aspect=DataAspect(),
      titlecolor=INK,
      titlesize=14,
      backgroundcolor=RGBf(0.97, 0.98, 0.99),
    )
    heatmap!(
      axis, x, x, update_fields[i]; colormap=:vik, colorrange=update_colorrange
    )
    contour!(
      axis, x, x, subdomain_masks[i]; levels=[0.5], color=:white, linewidth=2.5
    )
    hidedecorations!(axis)
    hidespines!(axis)
  end
  rowsize!(local_grid, 1, Fixed(48))
  rowsize!(local_grid, 2, Relative(0.5))
  rowsize!(local_grid, 3, Relative(0.5))
  rowgap!(local_grid, 10)
  colgap!(local_grid, 12)

  Label(
    fig[2, 4],
    "SECOND\nLEVEL\n→";
    fontsize=20,
    font=:bold,
    color=MUTED,
    lineheight=1.25,
  )

  next_axis = Axis(
    fig[2, 5];
    title=L"\mathsf{u}^{(1)}\qquad \mathcal{E}(\mathsf{u}^{(1)})=%$(fmt_energy(energy(next)))",
    xlabel="x",
    ylabel="y",
    aspect=DataAspect(),
    titlecolor=INK,
    backgroundcolor=RGBf(0.97, 0.98, 0.99),
  )
  heatmap!(
    next_axis, x, x, next_field; colormap=:vik, colorrange=field_colorrange
  )
  Colorbar(fig[2, 6], current_plot; label=L"\mathsf{u}_h", width=20)

  Label(
    fig[3, 1:6],
    L"\mathcal{Z}^{(k)}=\mathrm{span}\{\mathsf{u}^{(k)},\mathsf{y}_1^{(k)},\ldots,\mathsf{y}_m^{(k)}\},\qquad \mathsf{u}^{(k+1)}\in\mathrm{arg\,min}_{\mathsf{u}\in\mathcal{Z}^{(k)}}\;\mathsf{J}(\mathsf{u})\qquad(q=0)";
    fontsize=25,
    color=INK,
    padding=(0, 0, 8, 0),
  )

  colsize!(fig.layout, 1, Relative(0.29))
  colsize!(fig.layout, 2, Fixed(55))
  colsize!(fig.layout, 3, Relative(0.36))
  colsize!(fig.layout, 4, Fixed(55))
  colsize!(fig.layout, 5, Relative(0.29))
  colsize!(fig.layout, 6, Fixed(70))
  rowsize!(fig.layout, 1, Fixed(108))
  rowsize!(fig.layout, 2, Fixed(455))
  rowsize!(fig.layout, 3, Fixed(108))

  save(
    joinpath(ASSET_DIR, "variational-dd-poisson-iteration.png"),
    fig;
    px_per_unit=1.25,
  )
  return nothing
end

function make_convergence_figure(energy, subdomains)
  solution, _, energy_history, _, residual_history = S.var_dd(
    energy, subdomains; maxiter=50, tol=1e-8, verbose=false
  )
  direct_solution = energy.A \ energy.b
  minimum_energy = energy(direct_solution)
  energy_gaps = max.(energy_history .- minimum_energy, eps(Float64))

  fig = Figure(; size=(1250, 520), backgroundcolor=:white)
  Label(
    fig[1, 1:2],
    "Poisson convergence on a 16 × 16 mesh with four overlapping subdomains";
    fontsize=23,
    font=:bold,
    color=INK,
    padding=(0, 0, 4, 10),
  )

  residual_axis = Axis(
    fig[2, 1];
    xlabel="iteration k",
    ylabel=L"\|\mathsf{r}(\mathsf{u}^{(k)})\|_2",
    yscale=log10,
    title="stationarity residual",
    titlecolor=INK,
    backgroundcolor=:white,
    xgridcolor=GRID,
    ygridcolor=GRID,
  )
  iterations = collect(1:length(residual_history))
  lines!(residual_axis, iterations, residual_history; color=PURPLE, linewidth=3)
  scatter!(
    residual_axis, iterations, residual_history; color=PURPLE, markersize=8
  )
  hlines!(residual_axis, [1e-8]; color=ORANGE, linestyle=:dash, linewidth=2)
  text!(
    residual_axis,
    27.5,
    1.35e-8;
    text="tolerance",
    color=ORANGE,
    align=(:right, :bottom),
    fontsize=14,
  )

  energy_axis = Axis(
    fig[2, 2];
    xlabel="iteration k",
    ylabel=L"\mathcal{E}(\mathsf{u}^{(k)})-\mathcal{E}^{\star}",
    yscale=log10,
    title="energy error",
    titlecolor=INK,
    backgroundcolor=:white,
    xgridcolor=GRID,
    ygridcolor=GRID,
  )
  energy_iterations = collect(0:(length(energy_gaps) - 1))
  lines!(energy_axis, energy_iterations, energy_gaps; color=ORANGE, linewidth=3)
  scatter!(
    energy_axis, energy_iterations, energy_gaps; color=ORANGE, markersize=8
  )

  Label(
    fig[3, 1:2],
    @sprintf(
      "28 iterations   ·   final residual %.2e   ·   relative error to the direct FEM solution %.2e",
      residual_history[end],
      norm(solution - direct_solution) / norm(direct_solution),
    );
    fontsize=15,
    color=MUTED,
    padding=(0, 0, 12, 0),
  )
  colgap!(fig.layout, 55)

  save(joinpath(ASSET_DIR, "poisson-convergence.png"), fig; px_per_unit=1.25)
  return nothing
end

N = 16
A, _, b, subdomains, _ = FEM.FEM_Schroedinger(
  N, 4; P=x -> 0.0, f=x -> 1.0, overlap=2, partitioning=:cartesian
)
energy = E.QuadraticEnergy(A, b)

make_iteration_figure(energy, subdomains, N)
make_convergence_figure(energy, subdomains)
