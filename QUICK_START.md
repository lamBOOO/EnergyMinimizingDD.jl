# Quick Start: Nonoverlapping Domain Decomposition

## TL;DR

Want nonoverlapping domains where interface nodes belong to exactly one subdomain?

```julia
using DDEigen

# Just add these two parameters:
K, M, dofspar, VV = Setup_FEM(100, 9; 
    overlap=0,           # ← No element overlap
    nonoverlapping=true  # ← Remove DOF overlap
)
```

That's it! Now `dofspar[i]` and `dofspar[j]` have no shared DOFs for any i ≠ j.

## Before vs After

### Default (Overlapping)
```julia
K, M, dofspar, VV = Setup_FEM(100, 9)
# Interface nodes appear in multiple subdomains
# Example: domains = [[1,2,3], [2,3,4], ...]
```

### Nonoverlapping
```julia
K, M, dofspar, VV = Setup_FEM(100, 9; overlap=0, nonoverlapping=true)
# Interface nodes appear in only one subdomain
# Example: domains = [[1,2,3], [4,5], ...]
```

## Verification

```julia
# Check that domains are truly nonoverlapping
for i in 1:9
    for j in i+1:9
        shared = intersect(dofspar[i], dofspar[j])
        println("Domains $i and $j share: $(length(shared)) DOFs")
    end
end
# Should print: "0 DOFs" for all pairs
```

## With the Eigen Solver

```julia
u, λ, history, solutions = ddm_eigen_solver(
    N=100,
    m=9,
    maxiter=200,
    tol=1e-10,
    overlap=0,           # ← Add this
    nonoverlapping=true  # ← Add this
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overlap` | `Int` | `2` | Number of element layers to overlap |
| `nonoverlapping` | `Bool` | `false` | Remove DOF overlap at interfaces |

## FAQ

**Q: Does this break my existing code?**  
A: No! Default behavior is unchanged. The new parameters are optional.

**Q: What if I use `overlap=0` without `nonoverlapping=true`?**  
A: You'll still have DOF overlap at interfaces (from element-to-DOF conversion).

**Q: What if I use `overlap=2` with `nonoverlapping=true`?**  
A: Elements will overlap, but DOFs won't. The nonoverlapping filter is applied after element overlap.

**Q: Which subdomain gets the interface nodes?**  
A: The first subdomain that contains them (lower index gets priority).

**Q: Are all DOFs still covered?**  
A: Yes! Every DOF belongs to exactly one subdomain. No DOF is lost.

## More Info

- Full documentation: `README.md`
- Implementation details: `NONOVERLAPPING.md`
- Visual examples: `docs/nonoverlapping_diagram.txt`
- Code examples: `example_nonoverlapping.jl`
- Complete summary: `IMPLEMENTATION_SUMMARY.md`
