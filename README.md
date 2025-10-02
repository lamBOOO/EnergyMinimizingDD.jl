# DDEigen

Domain Decomposition based subspace iteration for the generalized eigenproblem `K u = λ M u`.

## Features

### Nonoverlapping Domain Decomposition

By default, the domain decomposition creates overlapping subdomains where nodes at the interface between subdomains belong to multiple subdomains. This is useful for many domain decomposition algorithms, but sometimes you need a true nonoverlapping partition.

The new `nonoverlapping` parameter allows you to create partitions where interface nodes belong to exactly one subdomain:

```julia
# Default behavior (with overlap)
K, M, dofspar, VV = Setup_FEM(100, 9)
# Result: interface nodes appear in multiple subdomains

# Nonoverlapping behavior
K, M, dofspar, VV = Setup_FEM(100, 9; overlap=0, nonoverlapping=true)
# Result: interface nodes appear in only one subdomain
```

### Example

```julia
using DDEigen

# Run the eigenvalue solver with nonoverlapping domain decomposition
u, λ, history, solutions = ddm_eigen_solver(
    N=100,           # Grid size
    m=9,             # Number of subdomains
    maxiter=200,     # Maximum iterations
    tol=1e-10,       # Convergence tolerance
    overlap=0,       # No element overlap
    nonoverlapping=true  # Remove duplicate DOFs at interfaces
)
```

### Parameters

- `overlap::Int=2`: Number of element layers to add as overlap between subdomains
- `nonoverlapping::Bool=false`: If `true`, removes duplicate DOFs at subdomain interfaces, ensuring each DOF belongs to exactly one subdomain

### Use Case

This feature is particularly useful when:
- You want a strict partition: `ids = [1,2,3,4]`, `domains = [[1,2],[3,4]]`
- You need to avoid double-counting nodes at interfaces
- You're implementing algorithms that require disjoint subdomains

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/lamBOOO/dd_eigen")
```

## Usage

See `example_nonoverlapping.jl` for a complete example.
