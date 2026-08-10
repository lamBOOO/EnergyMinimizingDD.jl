Studies 8 and 9 use the same overlapping one-level partitions for all methods.
Study 8 shows additive varDD, additive varDD with one-vector history, and
multiplicative varDD separately. The additive local candidates are independent
and their `m` subdomain solves can run in parallel. Multiplicative varDD feeds
each local result into the next subdomain, giving it a serial critical path of
`m` local solves per sweep. The history variant requires no additional local
solves and enlarges only the small second-level problem. Study 8 additionally
compares post-combination damping with omega = 0.25, 0.5, and 0.75.
The stored cost is the number of local subdomain solves: one variational DD
sweep, one additive Schwarz stationary step, one RAS stationary step, and one
additive-Schwarz preconditioner application each count as `m` subdomain solves.
Figure 12 divides this work count by `m` and plots outer solves.

Study 8 also includes right-preconditioned, unrestarted GMRES+RAS. RAS uses
the same overlapping subdomains as varDD and a balanced disjoint restriction
constructed from the original METIS element cores; it no longer assigns an
interface degree of freedom to the first overlapping subdomain encountered.
CG+AS and GMRES+RAS each use one global matrix-vector product and one parallel
batch of `m` local solves per Krylov iteration. Their true relative residual is
used for stopping.

The Study 8 sensitivity data change one parameter at a time rather than taking
a Cartesian product. Mesh refinement is shown at `N=20,40,...,120`, both with
two fixed overlap layers and with exactly fixed relative physical overlap. In
the latter sequence `overlap=N/20` and `m=4`, hence `delta/H=0.1`. For a
clean refinement study, these runs use the same nested `2 x 2` Cartesian
subdomain geometry at every mesh level; the other studies retain METIS
partitions. For a regular decomposition, the reported estimate is

    delta/H ≈ overlap*sqrt(m)/N.

The coefficient stress test uses a single off-centre circular inclusion with
diffusion contrast `kappa`; it is a conditioning sensitivity experiment, not a
claim of high-contrast robustness. Every contrast decade from `1` to `1e6` is
included. History depths `0, 1, 2, 4, 8` are compared
for homogeneous and contrast-`1e4` cases.

The inner-system CSV distinguishes the Schwarz block `R_i*K*R_i'`, the varDD
local space in an orthonormal basis, and the first-sweep combination problem.
More precisely, Figure 19 shows the maximum over subdomains of
`kappa_2(K_i)`, where `K_i=R_i*K*R_i'`, and of `kappa_2(Q_i'*K*Q_i)`, where
the columns of `Q_i` orthonormally span the first-sweep varDD space
`span{u_0,e_j : j in I_i}`. It also shows `kappa_2(Q_c'*K*Q_c)` for an
orthonormal basis of the first second-level candidate space. Each estimate is
the ratio of its largest and smallest eigenvalue. The local varDD and Schwarz
curves nearly coincide because the additional complement direction does not
control either spectral extreme; the small combination system remains much
better conditioned.

The CSV additionally reports an unpreconditioned local-CG iteration count at
relative tolerance `1e-8`. This is explicitly a
right-hand-side-dependent iterative-work proxy, not a condition number. Exact
cached Cholesky solves remain in the actual algorithms. Raw condition numbers
of nonorthogonal varDD coefficient matrices are not reported because they are
basis-dependent; effective rank and the condition of `Q'*K*Q` are reported
instead.
The observed history spaces remain full rank. Since this adds no discriminating
information, rank is retained in the CSV but omitted from Figure 20; the figure
instead emphasizes that depth one supplies nearly all of the convergence gain.
Local CG diagnostics that do not reach the requested tolerance are capped at
2000 iterations. A dotted horizontal line and black upward triangles mark
these right-censored values; they mean "at least 2000", not convergence in
exactly 2000 iterations.

Study 9 solves the generalized symmetric eigenproblem `K*u=lambda*M*u` from
the same M-normalized initial vector and stops on the same true residual
reduction. Figure 13 compares varDD, varDD with one previous iterate, LOPSD+AS,
LOBPCG+AS [Knyazev2001], Jacobi--Davidson with an AS-preconditioned flexible
GMRES correction solve [SleijpenVanDerVorst1996, GensebergerEtAl2010], and
shift-and-invert Lanczos with accurate PCG(AS) applications
[Simoncini2005]. The last method applies `K^{-1}*M` with inner relative
tolerance `1e-10`; it is therefore an intentionally expensive classical
reference rather than a one-AS-application method.

The eigenproblem work counters remain deliberately non-equivalent:

1. a parallel batch of local generalized eigenproblems for varDD,
2. a parallel batch of local AS triangular solves for the preconditioned
   eigensolvers,
3. local LOBPCG iterations inside the varDD augmented pencils, and
4. global `K` and `M` operator applications.

They are never collapsed into a single "equivalent work" number. The local
CSV reports the local dimension, LOBPCG iterations, sparse-pencil nonzeros,
and Cholesky-factor nonzeros. The cumulative parallel critical path is the sum
over sweeps of the maximum local LOBPCG iteration count over subdomains.

The Study 9 sensitivity runs use `N=20,40,80`. They show both two fixed overlap
layers and `overlap=N/20` for fixed `delta/H=0.1`. The oscillatory test uses

    a(x,y) = 1 + (kappa-1)/2 *
             (1 + sin(2*pi*nu*x)*sin(2*pi*nu*y)),

with `kappa=1e3`, `N=64`, and resolved frequencies `nu=1,2,4,8`. It tests
rapidly varying coefficients rather than higher eigenmodes, which would
require orthogonality constraints and deflation. History depths
`q=0,1,2,4,8` are compared separately. Figures 24--27 report outer scaling,
linear-AS/global-operator work, history sensitivity, and local-system
critical-path/storage diagnostics.

All methods shown are one-level methods, so their iteration counts are expected
to grow as the number of subdomains `m` increases. Two-level/coarse-space
extensions are left as future work.
