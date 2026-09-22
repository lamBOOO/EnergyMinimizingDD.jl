# EnergyMinimizingDD.jl — paper reproducibility code

This repository contains the Julia implementation and numerical experiments
used in the accompanying paper on energy-minimizing domain decomposition
(EMDD). It has intentionally been reduced to the code needed to reproduce the
five published figures and the linear-source scaling table.

## Requirements

- Julia 1.10 or newer
- a LaTeX installation only if the generated table is compiled with the paper

Instantiate the package environment from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Reproduce every result

```bash
julia --project=. examples/paper/run_all.jl
```

The study scripts cache raw CSV data in `examples/paper/data/`. Figures are
written as PDF and PNG files to `examples/paper/figures/`, and the generated
table is written to `examples/paper/tables/`.

Set `FORCE=1` to recompute cached experiments. A small, inexpensive smoke run
is available for installation checks:

```bash
SMALL=1 FORCE=1 julia --project=. examples/paper/run_all.jl
```

The small run deliberately skips the publication table because that table
requires the full parameter grid.

## Paper outputs and source scripts

| Paper output | Experiment |
|---|---|
| Fig. 12b, linear source problems | `study8_linear_cmp.jl` |
| Table 18, mesh/overlap sensitivity | `study8_linear_sensitivity.jl` |
| Fig. 13, linear eigenproblem | `study9_evp_cmp.jl` |
| Fig. 14b, Gross–Pitaevskii problem | `study10_gp.jl` |
| Fig. 17b, semilinear square problem | `study11_semilinear.jl` |
| Fig. 17b, semilinear L-domain problem | `study11b_semilinear_l_shape.jl` |

All paths in the table are relative to `examples/paper/`. The shared numerical
baselines are in `common.jl`, `evp_common.jl`, and
`nonlinear_source_common.jl`. Plotting and table generation are separated from
the expensive solves in `plots_all.jl` and `tables_all.jl`.

## Run checks

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/paper/tables_all.jl --check
```

Every Julia command should be run with `--project=.` so that it uses the
repository environment.
