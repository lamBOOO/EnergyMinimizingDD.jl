# Shared infrastructure for the Section 4 "Numerical results" study suite.
#
# Conventions:
# - Each study script writes its raw results to data/*.csv and skips
#   computation if the CSV already exists (override with FORCE=1).
# - SMALL=1 switches to tiny smoke-test parameters (~2 min total).
# - plots_all.jl regenerates every figure purely from the CSVs.
# - Figures are saved as PDF (vector, for LaTeX) and PNG (preview).

const PAPER_COMMON = true

using EnergyMinimizingDD.FEMDiscretizations
using EnergyMinimizingDD.Energies
using EnergyMinimizingDD.Solvers
using Gridap
using GridapDistributed
using Metis
using LinearAlgebra
using SparseArrays
using Random
using Printf
using CSV
using Tables
using FixedPointAcceleration
using NonlinearSolve
using Optim

const PAPER_DIR = @__DIR__
const DATA_DIR = mkpath(joinpath(PAPER_DIR, "data"))
const FIG_DIR = mkpath(joinpath(PAPER_DIR, "figures"))
const SMALL = get(ENV, "SMALL", "0") == "1"
const FORCE = get(ENV, "FORCE", "0") == "1"

datafile(name) = joinpath(DATA_DIR, name)

"Skip a study when all of its CSVs already exist (unless FORCE=1)."
needs_run(files...) = FORCE || !all(f -> isfile(datafile(f)), files)

savetable(name, tbl) = CSV.write(datafile(name), tbl)
loadtable(name) = Tables.columntable(CSV.File(datafile(name)))

# ---------------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------------

"Laplacian (P ≡ 0) setup on the unit square; also used for Poisson and heat."
laplace_setup(N, m, overlap; kwargs...) = FEMDiscretizations.FEM_Schroedinger(
  N,
  m;
  P = (x -> 0.0),
  f = (x -> 1.0),
  overlap = overlap,
  kwargs...,
)

"Schroedinger setup with the repo's default exponential potential."
schroedinger_setup(N, m, overlap; kwargs...) = FEMDiscretizations.FEM_Schroedinger(
  N,
  m;
  f = (x -> 1.0),
  overlap = overlap,
  kwargs...,
)

"Compute a METIS owner for every cell of an `N x N` Cartesian mesh."
function metis_cell_owners(N, m)
  model = CartesianDiscreteModel(
    (0, 1.0, 0, 1.0),
    (1.0 * N, 1.0 * N);
    isperiodic = (false, false),
  )
  g = GridapDistributed.compute_cell_graph(model)
  return Int32.(Metis.partition(g, m))
end

"""
    prolong_cell_owners(reference_owners, N)

Prolong a cellwise partition of a square reference grid to an integer uniform
refinement. Every child cell inherits its parent cell's owner, so all refined
meshes represent the same physical (possibly irregular) core partition.
"""
function prolong_cell_owners(reference_owners, N)
  reference_N = isqrt(length(reference_owners))
  reference_N^2 == length(reference_owners) || throw(DimensionMismatch(
    "reference_owners must describe a square Cartesian grid",
  ))
  N % reference_N == 0 || throw(ArgumentError(
    "N=$N must be an integer refinement of reference_N=$reference_N",
  ))
  refinement = N ÷ reference_N
  owners = Vector{Int32}(undef, N^2)
  for cell = 1:N^2
    xcell = mod1(cell, N)
    ycell = cld(cell, N)
    parent_x = cld(xcell, refinement)
    parent_y = cld(ycell, refinement)
    parent = parent_x + reference_N * (parent_y - 1)
    owners[cell] = reference_owners[parent]
  end
  return owners
end

"Overlap multiplicity for an explicitly supplied non-overlapping cell owner."
function cell_partition_overlap(N, m, overlap, owner)
  length(owner) == N^2 || throw(DimensionMismatch(
    "owner must contain one entry for every cell",
  ))
  model = CartesianDiscreteModel(
    (0, 1.0, 0, 1.0),
    (1.0 * N, 1.0 * N);
    isperiodic = (false, false),
  )
  g = GridapDistributed.compute_cell_graph(model)
  elpar = FEMDiscretizations.create_elements_partition(Int32.(owner), m)
  FEMDiscretizations.create_overlapping_elements_partition!(elpar, g, m, overlap)
  mult = zeros(Int, length(owner))
  for elems in elpar, el in elems
    mult[el] += 1
  end
  return Int.(owner), mult
end

"Non-overlapping METIS cell owner and overlap multiplicity on the N x N mesh."
function metis_cell_partition(N, m, overlap)
  return cell_partition_overlap(N, m, overlap, metis_cell_owners(N, m))
end

# ---------------------------------------------------------------------------
# Schwarz building blocks (baselines for studies 8/9), sharing the var_dd
# overlapping partition. Cost unit across all methods: one subdomain solve.
# ---------------------------------------------------------------------------

struct SchwarzData
  dofs::Vector{Vector{Int}}
  facts::Vector{Any}                 # Cholesky factors of K[dofs, dofs]
  owner_mask::Vector{Vector{Bool}}   # dof owned by this subdomain (for RAS)
  max_mult::Int                      # max overlap multiplicity (AS damping)
end

nsub(S::SchwarzData) = length(S.dofs)

function schwarz_setup(K, dofspar; core_dofs = nothing)
  dofs = [collect(Int, d) for d in dofspar]
  facts = [cholesky(Symmetric(sparse(K[d, d]))) for d in dofs]
  mult = zeros(Int, size(K, 1))
  for d in dofs, j in d
    mult[j] += 1
  end
  ownership_candidates = isnothing(core_dofs) ? dofspar : core_dofs
  owned = FEMDiscretizations.create_balanced_disjoint_dofs_partition(
    ownership_candidates, size(K, 1)
  )
  owner = zeros(Int, size(K, 1))
  for (i, d) in enumerate(owned), j in d
    owner[j] = i
  end
  masks = [owner[d] .== i for (i, d) in enumerate(dofs)]
  all(owner .> 0) || error("RAS ownership does not cover every free DOF")
  coverage = zeros(Int, size(K, 1))
  for (d, mask) in zip(dofs, masks)
    coverage[d[mask]] .+= 1
  end
  all(coverage .== 1) ||
    error("RAS ownership must restrict every free DOF exactly once")
  return SchwarzData(dofs, facts, masks, maximum(mult))
end

"Additive Schwarz application: z = sum_i R_i' K_ii^{-1} R_i r."
function apply_AS(S::SchwarzData, r)
  z = zeros(length(r))
  for (d, F) in zip(S.dofs, S.facts)
    z[d] .+= F \ r[d]
  end
  return z
end

"Restricted additive Schwarz (Cai-Sarkis): each dof updated by its owner only."
function apply_RAS(S::SchwarzData, r)
  z = zeros(length(r))
  for (i, (d, F)) in enumerate(zip(S.dofs, S.facts))
    c = F \ r[d]
    zd = view(z, d)
    zd[S.owner_mask[i]] .+= c[S.owner_mask[i]]
  end
  return z
end

"Additive Schwarz preconditioner wrapper for IterativeSolvers.jl."
struct ASPreconditioner
  S::SchwarzData
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::ASPreconditioner, x::AbstractVector)
  y .= apply_AS(P.S, x)
  return y
end

function LinearAlgebra.ldiv!(Y::AbstractMatrix, P::ASPreconditioner, X::AbstractMatrix)
  @inbounds for j in axes(X, 2)
    ldiv!(view(Y, :, j), P, view(X, :, j))
  end
  return Y
end
