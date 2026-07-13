# Time-dependent example: heat equation via variational domain decomposition.
#
# The semi-discrete heat equation M u̇ = -(K u - b) is the gradient flow of
#   J(u) = ½ u'K u - b'u
# in the M-inner product. A backward Euler step is the incremental
# minimization problem (minimizing movements):
#   u^{n+1} = argmin_u  J(u) + 1/(2τ) ‖u - u^n‖²_M
# whose objective is again quadratic,
#   E_τ(u) = ½ u'(K + M/τ) u - (b + M u^n/τ)'u  (+ const),
# so every time step is a QuadraticEnergy solved with the existing var_dd,
# warm-started from the previous time step.

using VariationalDD.FEMDiscretizations
using VariationalDD.Energies
using VariationalDD.Solvers
using LinearAlgebra
using Printf

# FEM setup: P ≡ 0 turns the Schroedinger operator into the plain Laplacian,
# so K is the stiffness matrix, M the mass matrix and b the load vector (f ≡ 1)
N = 16       # mesh resolution (N×N cells)
m = 4        # number of subdomains
K, M, b, dofspar, U = FEMDiscretizations.FEM_Schroedinger(
  N,
  m;
  P = (x -> 0.0),
  f = (x -> 1.0),
  overlap = 2,
)

τ = 1e-2     # time step size
nsteps = 10

ndofs = size(K, 1)
u = ones(ndofs)      # initial condition u(0) ≡ 1 in the interior
A_tau = K + M / τ    # constant since τ is fixed

# Free energy of the flow; backward Euler dissipates it monotonically
J(u) = 0.5 * dot(u, K * u) - dot(b, u)

u_direct = copy(u)   # reference trajectory via direct backward Euler solves
@printf("step %2d (t = %.3f): J = %+.6e (initial condition)\n", 0, 0.0, J(u))
for n = 1:nsteps
  # Incremental minimization problem for this time step
  b_tau = b + M * u / τ
  e_step = Energies.QuadraticEnergy(A_tau, b_tau)

  u_new, _, _, _, resnorm_hist =
    Solvers.var_dd(e_step, dofspar; maxiter = 100, tol = 1e-8, u0 = u)

  global u_direct = A_tau \ (b + M * u_direct / τ)
  err = norm(u_new - u_direct) / norm(u_direct)

  @printf(
    "step %2d (t = %.3f): %2d DD iters, J = %+.6e, rel. error vs direct = %.2e\n",
    n,
    n * τ,
    length(resnorm_hist),
    J(u_new),
    err,
  )
  global u = u_new
end

# For t → ∞ the flow relaxes to the steady state K u = b
@printf(
  "steady-state residual ‖K u - b‖ = %.2e (→ 0 for nsteps → ∞)\n",
  norm(K * u - b),
)
