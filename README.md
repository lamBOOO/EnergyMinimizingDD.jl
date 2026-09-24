# EnergyMinimizingDD.jl

<p align="center">
  <strong>Energy-minimizing domain decomposition for finite-element problems in Julia</strong>
</p>

<p align="center">
  <a href="https://github.com/lamBOOO/EnergyMinimizingDD.jl/actions/workflows/ci.yml"><img src="https://github.com/lamBOOO/EnergyMinimizingDD.jl/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://julialang.org/"><img src="https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia&logoColor=white" alt="Julia 1.10 or newer"></a>
  <a href="https://lambooo.github.io/EnergyMinimizingDD.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-2B6CB0" alt="Documentation"></a>
  <a href="#project-status"><img src="https://img.shields.io/badge/status-experimental-EA8C00" alt="Experimental status"></a>
</p>

EnergyMinimizingDD.jl implements energy-minimizing domain-decomposition methods for finite-element problems. Each iteration solves independent variational problems on overlapping local spaces and then recombines the resulting candidates through a small global minimization. A common solver interface supports quadratic source problems, generalized eigenproblems, semilinear energies, and Gross–Pitaevskii models.

## Quick start: Poisson's equation

Install the current development version directly from GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/lamBOOO/EnergyMinimizingDD.jl.git")
```

The following example solves

$$
-\Delta u = 1 \quad \text{in } (0,1)^2,
\qquad u = 0 \quad \text{on } \partial(0,1)^2,
$$

with four overlapping subdomains:

```julia
using EnergyMinimizingDD, LinearAlgebra

const FEM = EnergyMinimizingDD.FEMDiscretizations
const E = EnergyMinimizingDD.Energies
const S = EnergyMinimizingDD.Solvers

A, _, b, subdomains, _ = FEM.FEM_Schroedinger(
    16, 4; P=x -> 0.0, f=x -> 1.0, overlap=2,
    partitioning=:cartesian,
)
energy = E.QuadraticEnergy(A, b)
result = S.var_dd(energy, subdomains; maxiter=50, tol=1e-8, verbose=false)

@show result.converged result.iterations norm(A * result.u - b)
# result.converged = true
# result.iterations = 28
# norm(A * result.u - b) = 5.337471100984455e-9
```

`var_dd` returns a `VarDDResult` holding the final iterate `u`, its `energy`,
the `energy_history`, `iterate_history` and `residual_history`, and the
`converged` / `iterations` status.

## Development notice

AI/LLM-based tools are used to assist with coding in this repository.
