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
iterate and can run in parallel. Their plotted x-coordinate counts total local
minimizations; a sweep has `m` units of work but a one-local-minimization
critical path. Retaining the preceding global iterate changes only the small
second-level problem and requires no additional local minimizations.

The beta-zero implementation dispatches directly to the generalized linear
Rayleigh quotient, providing an exact regression to the linear EVP algorithm.
For every interaction strength, the reference state is computed independently
from the same linear ground-state initial guess; no continuation in beta is
used.
