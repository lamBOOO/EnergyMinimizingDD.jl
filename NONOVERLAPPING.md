# Nonoverlapping Domain Decomposition Implementation

## Problem Statement

The user requested the ability to create truly nonoverlapping domain decompositions using METIS. Previously, even with `overlap=0`, nodes at the interface between subdomains would belong to both subdomains because:

1. METIS partitions **elements** into disjoint sets
2. When converting from elements to DOFs (degrees of freedom), nodes shared by elements at the interface naturally appear in multiple subdomains

### Example of the Problem

For a 1D mesh with 4 elements:
- Element 1: nodes [1, 2]
- Element 2: nodes [2, 3]  
- Element 3: nodes [3, 4]
- Element 4: nodes [4, 5]

METIS partition: elements [1,2] → domain 1, elements [3,4] → domain 2

**Before fix (overlapping):**
- Domain 1: nodes [1, 2, 3]
- Domain 2: nodes [3, 4, 5]
- **Shared node: 3**

**After fix (nonoverlapping):**
- Domain 1: nodes [1, 2, 3]
- Domain 2: nodes [4, 5]
- **No shared nodes**

## Solution

### Implementation Details

1. **Added `nonoverlapping` parameter** to `create_dofs_partition` function
   - Type: `Bool`
   - Default: `false` (maintains backward compatibility)
   
2. **Algorithm**: Process subdomains in order, assigning each node to the first subdomain that contains it
   ```julia
   if nonoverlapping
     assigned_nodes = Set{Int32}()
     for ipar = 1:npars
       freenodesp[ipar] = filter(node -> !(node in assigned_nodes), freenodesp[ipar])
       union!(assigned_nodes, freenodesp[ipar])
     end
   end
   ```

3. **Propagated parameters** through the call chain:
   - `create_dofs_partition` → accepts `nonoverlapping`
   - `Setup_FEM` → accepts `overlap` and `nonoverlapping`
   - `ddm_eigen_solver` → accepts `overlap` and `nonoverlapping`

### Key Properties

- **Backward Compatible**: Default behavior unchanged (`overlap=2`, `nonoverlapping=false`)
- **Coverage Guarantee**: All DOFs are covered (no DOF is lost)
- **Disjoint Guarantee**: No DOF appears in multiple subdomains when `nonoverlapping=true`
- **Minimal Changes**: Only ~30 lines of code changed in the core algorithm

## Usage

### Basic Example

```julia
using DDEigen

# Nonoverlapping decomposition
K, M, dofspar, VV = Setup_FEM(100, 9; overlap=0, nonoverlapping=true)

# Check that domains are truly nonoverlapping
intersection = intersect(dofspar[1], dofspar[2])
@assert isempty(intersection)  # Should be empty
```

### Full Solver Example

```julia
u, λ, history, solutions = ddm_eigen_solver(
    N=100,
    m=9,
    maxiter=200,
    tol=1e-10,
    overlap=0,           # No element overlap
    nonoverlapping=true  # Remove DOF overlap at interfaces
)
```

## Testing

Unit tests in `test/test_nonoverlapping.jl` verify:
1. Two subdomains with overlapping nodes → correctly separated
2. Three subdomains with cascading overlaps → correctly separated
3. Already nonoverlapping subdomains → remain unchanged
4. All nodes covered before and after transformation

All tests pass ✓

## Files Modified

- `solve_fem.jl`: Core implementation (3 functions modified)
- `README.md`: User documentation
- `example_nonoverlapping.jl`: Usage examples
- `test/test_nonoverlapping.jl`: Unit tests
- `NONOVERLAPPING.md`: This document

## References

This implementation addresses the GitHub issue requesting:
```
ids = [1,2,3,4]
domains = [[1,2],[3,4]]  # Nonoverlapping
```

Instead of the default:
```
domains = [[1,2,3],[2,3,4]]  # Overlapping at node 2 and 3
```
