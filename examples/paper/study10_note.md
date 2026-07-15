Study 10 considers the real, non-rotating Gross--Pitaevskii ground-state
problem with repulsive interaction strength beta. The implementation minimizes
the scale-invariant nonlinear Rayleigh quotient

    R_beta(u) = (u' K u)/(u' M u)
                + (beta/2) integral(u_h^4)/(u' M u)^2,

which equals twice the constrained physical energy evaluated at the
M-normalized state. Reduced local and second-level problems minimize this
scale-invariant quotient directly with unconstrained L-BFGS in M-orthonormal
coordinates. Normalization is applied only to the returned representative, not
as a constraint in the reduced optimization.

The additive local nonlinear minimizations all start from the same global
iterate and can run in parallel. As in Studies 8 and 9, the plotted
x-coordinate is the outer iteration number (the recorded number of local
solves or minimizations divided by `m`). A sweep has `m` units of work but a
one-local-minimization critical path. Retaining the preceding global iterate
changes only the small second-level problem and requires no additional local
minimizations.

The monolithic comparison uses the current energy operator
`A_u = K + beta*C(u)` as a changing Sobolev metric. One-level additive Schwarz
approximates its inverse with the same overlapping partitions as varDD. The
memoryless GFDN(a_u)+AS curve uses the projected preconditioned gradient; the
CG-GFDN(a_u)+AS curve adds the transported preceding direction with a
Fletcher--Reeves coefficient and restarts whenever it ceases to be a descent
direction. In both cases the existing GP `combine_step` supplies an
energy-optimal normalized line search. Each baseline iteration uses `m` local
linear solves, whereas each varDD sweep uses `m` local nonlinear
minimizations. Thus the shared iteration axis compares outer convergence, not
equal wall-clock cost.

The convergence figure reports the norm of the normalized Gross--Pitaevskii
Euler--Lagrange residual

    r_k = A(u_k)u_k - lambda_k*M*u_k
        = K*u_k + beta*C(u_k)u_k - lambda_k*M*u_k.

This is the stationarity residual for the mass-normalized problem. It was
previously labeled "projected residual" because its component in the
normalization direction vanishes, but "residual norm" is the clearer name for
the plotted quantity.

Each convergence panel also shows two small spatial insets. The colored
partition inset gives the METIS ownership and overlap for the column's value
of `m`, as in Studies 8 and 9. The viridis inset gives the reference
ground-state density `|u_star|^2` for the row's value of `beta`.

The beta-zero implementation dispatches directly to the generalized linear
Rayleigh quotient, providing an exact regression to the linear EVP algorithm.
For every interaction strength, the reference state is computed independently
from the same linear ground-state initial guess; no continuation in beta is
used.
