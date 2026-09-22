# EnergyMinimizingDD.jl

EnergyMinimizingDD.jl contains the Julia implementation and numerical
experiments for the accompanying paper on energy-minimizing domain
decomposition.

!!! warning "Project status"
    EnergyMinimizingDD.jl is experimental research software. The public API
    may evolve before a stable release.

## Installation

From the repository root, instantiate the project environment with

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Julia 1.10 or newer is required.

## Reproducing the paper results

Run every retained numerical study, generate the five paper figures, and
regenerate the paper table with

```bash
julia --project=. examples/paper/run_all.jl
```

For a smaller installation smoke test, use

```bash
SMALL=1 FORCE=1 julia --project=. examples/paper/run_all.jl
```

The small run skips the publication table because it requires the full
parameter grids. See the repository README for the mapping between paper
outputs and study scripts.

## Checks

Run the focused test suite with

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
