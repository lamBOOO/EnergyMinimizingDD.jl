# Shared infrastructure for the Section 4 "Numerical results" study suite.
#
# Conventions:
# - Each study script writes its raw results to data/*.csv and skips
#   computation if the CSV already exists (override with FORCE=1).
# - SMALL=1 switches to tiny smoke-test parameters (~2 min total).
# - plots_all.jl regenerates every figure purely from the CSVs.
# - Figures are saved as PDF (vector, for LaTeX) and PNG (preview).

const PAPER_COMMON = true

using VariationalDD.FEMDiscretizations
using VariationalDD.Energies
using VariationalDD.Solvers
using Gridap
using GridapDistributed
using Metis
using Arpack
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using CSV
using Tables

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

"""
    timed_median(f; repeats = 3)

Wall-clock timing of `f()`: one warm-up call (also covers compilation),
then the median over `repeats` timed calls. Returns (median_seconds, result).
"""
function timed_median(f; repeats = 3)
  result = f()  # warm-up
  ts = [(@elapsed f()) for _ = 1:repeats]
  return median(ts), result
end

# ---------------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------------

"Laplacian (P ≡ 0) setup on the unit square; also used for Poisson and heat."
laplace_setup(N, m, overlap) = FEMDiscretizations.FEM_Schroedinger(
  N,
  m;
  P = (x -> 0.0),
  f = (x -> 1.0),
  overlap = overlap,
)

"Schroedinger setup with the repo's default exponential potential."
schroedinger_setup(N, m, overlap) = FEMDiscretizations.FEM_Schroedinger(
  N,
  m;
  f = (x -> 1.0),
  overlap = overlap,
)

"Non-overlapping METIS cell owner and overlap multiplicity on the N x N mesh."
function metis_cell_partition(N, m, overlap)
  model = CartesianDiscreteModel(
    (0, 1.0, 0, 1.0),
    (1.0 * N, 1.0 * N);
    isperiodic = (false, false),
  )
  g = GridapDistributed.compute_cell_graph(model)
  owner = Int.(Metis.partition(g, m))
  elpar = FEMDiscretizations.create_elements_partition(Int32.(owner), m)
  FEMDiscretizations.create_overlapping_elements_partition!(elpar, g, m, overlap)
  mult = zeros(Int, length(owner))
  for elems in elpar, el in elems
    mult[el] += 1
  end
  return owner, mult
end

"Reference lowest eigenvalue of the discrete pencil (K, M) via Arpack."
function reference_lambda(K, M)
  vals, _ = Arpack.eigs(K, M; nev = 1, which = :SM, maxiter = 1000)
  return real(vals[1])
end

"Reference lowest eigenvalue of the discrete pencil (K, M) via dense LAPACK."
function dense_reference_lambda(K, M)
  vals = eigen(Symmetric(Matrix(K)), Symmetric(Matrix(M))).values
  return minimum(vals)
end

"Free dofs of the Q1 space on the N×N Cartesian mesh are the interior nodes
in lexicographic (x1-fastest) ordering; reshape accordingly for heatmaps."
field_matrix(u, N) = Matrix(reshape(u, N - 1, N - 1)')

interior_nodes(N) = range(1 / N, 1 - 1 / N; length = N - 1)

all_nodes(N) = range(0, 1; length = N + 1)

"Pad a free-dof field with its homogeneous Dirichlet boundary values (zero)
so surface/heatmap plots extend to the actual domain boundary."
function field_matrix_with_bc(u, N)
  Z = zeros(N + 1, N + 1)
  Z[2:N, 2:N] .= field_matrix(u, N)
  return Z
end

"Floor values for semilog plots (histories hit machine precision)."
logfloor(v; floor = 1e-16) = max.(v, floor)

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

function schwarz_setup(K, dofspar)
  dofs = [collect(Int, d) for d in dofspar]
  facts = [cholesky(Symmetric(sparse(K[d, d]))) for d in dofs]
  owner = zeros(Int, size(K, 1))
  mult = zeros(Int, size(K, 1))
  for (i, d) in enumerate(dofs), j in d
    owner[j] == 0 && (owner[j] = i)
    mult[j] += 1
  end
  masks = [owner[d] .== i for (i, d) in enumerate(dofs)]
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
