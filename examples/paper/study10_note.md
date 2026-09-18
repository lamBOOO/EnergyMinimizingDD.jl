Study 10 uses the illustrative example from Henning--Jarlebring, section 2.3:
the domain is `[-8,8]^2`, the interaction strength is `kappa=500`, and

    V(x,y) = 0.5 * (x^2 + 4*y^2)
             + 10 * (sin(pi*x)^2 + sin(pi*y)^2).

Every method starts from their prescribed L2-normalized function
`u0(x,y)=c*(x^2-8^2)*(y^2-8^2)`. The implementation minimizes
the scale-invariant nonlinear Rayleigh quotient

    R_beta(u) = (u' K u)/(u' M u)
                + (beta/2) integral(u_h^4)/(u' M u)^2,

which equals twice the constrained physical energy evaluated at the
M-normalized state. Reduced local and second-level problems solve the nonlinear
eigenproblem by damped self-consistent-field iteration in M-orthonormal
coordinates. Every iterate is normalized. Each SCF step diagonalizes the
Hamiltonian frozen at the current density; backtracking along the resulting
SCF direction prevents the two-cycles that plain Roothaan iteration can exhibit
at large interaction strength.

The comparison uses `kappa=10,100,500` and `m=2,4,8` on a `32 x 32` Q1 mesh.
The first two rows illustrate increasing interaction strength with the same
domain, potential, mesh, and initial state; the `kappa=500` row is the section
2.3 paper benchmark.

The additive local nonlinear minimizations all start from the same global
iterate and can run in parallel. As in Studies 8 and 9, the plotted
x-coordinate is the outer iteration number. A sweep has `m` units of work but a
one-local-minimization critical path. Retaining the preceding global iterate
changes only the small second-level problem and requires no additional local
minimizations. Every method may run for up to 30 outer iterations.

The curves labeled **quadratic EMDD** replace only those local nonlinear
minimizations by generalized linear eigenproblems. At the beginning of sweep
`k`, they freeze the density at the normalized global iterate and use
`K + kappa*C(u_k)` for every independent local EVP. The combination space is
then minimized with the full nonlinear GP quotient, exactly as for the other
EMDD curves. Thus a local quadratic-EMDD update is one local SCF-like step,
while the global combination is not linearized. The `q=1,2,3,4` variants
retain zero, one, two, or three preceding global iterates, respectively.

The **tangent quadratic EMDD** curves instead form the quadratic Taylor model
of the physical GP energy at the current normalized iterate and restrict it to
the linearized mass constraint. Each local
correction lies in the enriched local tangent space and is obtained from one
sparse KKT solve enforcing the linearized mass constraint. The candidate is
then retracted to unit mass. Only the local step is quadratic: the combination
space is minimized with the full nonlinear GP quotient. The plotted variants
use `q=1,2`.

The **charge-mixed EMDD** variants use the same local linear EVPs but assemble
their frozen operator from
`rho_mix = alpha*rho_k + (1-alpha)*rho_(k-1)`, with
`alpha in {0.25,0.5,0.75}`. The first sweep has no preceding density and is
therefore identical to the unmixed frozen-density method. The `q=1,2`
variants control only the global vectors retained in the nonlinear combination
space; density mixing always retains exactly one preceding density. Both
density states are mass-normalized before mixing.

The exact GFDN(a_u) implementation is retained only as an internal reference
generator and mesh-validation utility; it is not a comparison curve. It
implements Definition 5.12. In each outer
iteration it solves `A(u_n) z_n = M*u_n` by a sparse Cholesky factorization,
forms the update in (5.28), and selects the optimal step on `[0,2]` according
to (5.30). Thus its iteration count can be compared directly with Figure 6 of
Henning--Jarlebring. With the explicitly documented `32 x 32` Q1 mesh
(`h=0.5`) and
eighth-order quadrature, the discrete energy error is approximately `1e-9`
after 30 iterations, reproducing the iteration behavior in Figure 6. The
paper does not specify the spatial mesh used for that figure; consequently,
its reported continuous/reference values `E_GS ≈ 10.8995` and
`lambda_GS ≈ 27.7133` are used as mesh-convergence checks, not exact discrete
regression values.

The separate `run_hj_mesh_validation()` routine performs 30 exact-GFDN steps
on meshes up to `N=256` (`h=16/N`). At `h=0.0625` it obtains
`E_h=10.9007980` and `lambda_h=27.7149190`, close to the quoted paper values.
The `h=1e-3` mentioned on article page 294 is not the mesh for section 2.3:
it belongs to a different one-dimensional experiment on `(-2,2)` with
`kappa=20` used in Figure 7.

The **GFDN-PCG(AS,n)** comparisons use the current energy operator
`A_u = K + beta*C(u)` as a changing Sobolev metric. They apply exactly
`n in {1,2,4}` preconditioned conjugate-gradient steps, starting from zero,
with the overlapping one-level additive Schwarz operator as preconditioner.
There is no inner stopping tolerance. One PCG step is a scalar multiple of one
AS application and therefore spans the same nonlinear line-search space as the
former one-AS curve. The memoryless variants use the projected PCG direction;
the **CG-GFDN-PCG(AS,n)** variants add the transported preceding direction with
a Fletcher--Reeves coefficient and restart whenever it ceases to be a descent
direction. In both cases the full nonlinear GP `combine_step` supplies the
energy-optimal normalized line search. Each inner PCG step costs one batch of
`m` independent local linear solves and one global `A_u` product. Thus the
shared iteration axis compares outer convergence, while the method label makes
the differing inner work explicit.

The convergence figure reports the norm of the normalized Gross--Pitaevskii
Euler--Lagrange residual

    r_k = A(u_k)u_k - lambda_k*M*u_k
        = K*u_k + kappa*C(u_k)u_k - lambda_k*M*u_k.

This is the stationarity residual for the mass-normalized problem. It was
previously labeled "projected residual" because its component in the
normalization direction vanishes, but "residual norm" is the clearer name for
the plotted quantity.

Each convergence panel also shows two small spatial insets. The colored
partition inset gives the METIS ownership and overlap for the column's value
of `m`, as in Studies 8 and 9. The viridis inset gives the reference
ground-state density `|u_star|^2` for `kappa=500`.

The beta-zero implementation dispatches directly to the generalized linear
Rayleigh quotient, providing an exact regression to the linear EVP algorithm.

The separate `run_study10_local_work()` routine records the inner SCF
statistics of every local solve performed by EMDD with `q=1` and `q=2`
(`study10_gp_local_work.csv`). It is kept out of `run_study10` because it repeats
the EMDD solves only to collect per-local-problem work. The local reduced
problems use the same projected nonlinear-eigenproblem residual tolerance as
the second-level solve. The recorded counts therefore measure frozen-density
eigensolves rather than quasi-Newton steps and must be regenerated before
making work comparisons with results produced by the former L-BFGS solver.
