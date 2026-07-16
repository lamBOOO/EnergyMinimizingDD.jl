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
M-normalized state. Reduced local and second-level problems minimize this
scale-invariant quotient directly with unconstrained L-BFGS in M-orthonormal
coordinates. Normalization is applied only to the returned representative, not
as a constraint in the reduced optimization.

The comparison uses `kappa=10,100,500`. The first two rows illustrate
increasing interaction strength with the same domain, potential, mesh, and
initial state; the `kappa=500` row is the section 2.3 paper benchmark.

The additive local nonlinear minimizations all start from the same global
iterate and can run in parallel. As in Studies 8 and 9, the plotted
x-coordinate is the outer iteration number. A sweep has `m` units of work but a
one-local-minimization critical path. Retaining the preceding global iterate
changes only the small second-level problem and requires no additional local
minimizations.

The **exact GFDN(a_u)** benchmark implements Definition 5.12. In each outer
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

For completeness, **exact CG-GFDN(a_u)** applies the same exact sparse
Cholesky metric inversion but augments the direction with the transported
preceding direction and a Fletcher--Reeves coefficient. It uses the same
descent restart and energy-optimal subspace line search as the AS-inexact
CG-GFDN curve. This isolates the effect of exact versus one-AS metric inversion
under an otherwise identical CG acceleration.

The separate `run_hj_mesh_validation()` routine performs 30 exact-GFDN steps
on meshes up to `N=256` (`h=16/N`). At `h=0.0625` it obtains
`E_h=10.9007980` and `lambda_h=27.7149190`, close to the quoted paper values.
The `h=1e-3` mentioned on article page 294 is not the mesh for section 2.3:
it belongs to a different one-dimensional experiment on `(-2,2)` with
`kappa=20` used in Figure 7.

The comparison labeled **AS-inexact GFDN(a_u)** uses the current energy operator
`A_u = K + beta*C(u)` as a changing Sobolev metric. One-level additive Schwarz
approximates its inverse with the same overlapping partitions as varDD. Only
one AS application is performed per outer iteration; the `A_u` system is not
solved to a prescribed global tolerance. The memoryless AS-inexact GFDN(a_u)
curve uses the projected preconditioned gradient; the AS-inexact CG-GFDN(a_u)
curve adds the transported preceding direction with a
Fletcher--Reeves coefficient and restarts whenever it ceases to be a descent
direction. In both cases the existing GP `combine_step` supplies an
energy-optimal normalized line search. Each baseline iteration uses `m` local
linear solves, whereas each varDD sweep uses `m` local nonlinear
minimizations. Thus the shared iteration axis compares outer convergence, not
equal wall-clock cost.

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
