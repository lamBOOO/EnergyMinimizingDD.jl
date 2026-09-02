# [API reference](@id api-reference)

The package currently exposes its implementation through the
`EnergyMinimizingDD.Energies`, `EnergyMinimizingDD.FEMDiscretizations`, and
`EnergyMinimizingDD.Solvers` modules. The routines below form the core variational
domain-decomposition interface.

## Core functions

```@docs
EnergyMinimizingDD.Solvers.inf_step
EnergyMinimizingDD.Solvers.combine_step
EnergyMinimizingDD.Solvers.partition_of_unity_weights
EnergyMinimizingDD.Solvers.nicolaides_coarse_basis
EnergyMinimizingDD.Solvers.var_dd
```

## Restricted and two-level variants

REMDD uses `restriction=:partition_of_unity` to multiply overlapping local
corrections by inverse multiplicity weights before global recombination:

```julia
u, history... = EnergyMinimizingDD.Solvers.var_dd(
    energy,
    overlapping_dofs;
    restriction=:partition_of_unity,
)
```

For experiments, the same multiplicity columns can also be supplied as a
coarse-space comparator. They are distinct from the recommended low-energy
Nicolaides-type basis:

```julia
Z_multiplicity = EnergyMinimizingDD.Solvers.partition_of_unity_weights(
    overlapping_dofs,
    size(K, 1),
)
Z_harmonic = EnergyMinimizingDD.Solvers.nicolaides_coarse_basis(
    K,
    core_dofs,
    overlapping_dofs,
)

# q = 1: current iterate plus the current local candidates
u_q1, history_q1... = EnergyMinimizingDD.Solvers.var_dd(
    energy,
    overlapping_dofs;
    coarse_basis=Z_harmonic,
)

# q = 2: additionally retain the preceding global iterate
u_q2, history_q2... = EnergyMinimizingDD.Solvers.var_dd(
    energy,
    overlapping_dofs;
    history_depth=1,
    coarse_basis=Z_harmonic,
)
```

For each subdomain, the raw coarse function is one on its nonoverlapping core,
zero outside its overlap, and discrete harmonic in between. The raw functions
are normalized pointwise, so the columns of `Z_harmonic` sum to the global constant
vector. Interface DOFs may occur in multiple entries of `core_dofs`; the
constructor assigns each of them to one core.

The weak-scaling example in
`examples/paper/study8_weak_scaling.jl` compares EMDD and REMDD with `q=1` and
`q=2` for three cases: no coarse space, multiplicity-PoU columns, and the
harmonic Nicolaides space. The compact executable example
`examples/nicolaides_poisson.jl` runs the same three-way comparison.
