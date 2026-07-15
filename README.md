# VariationalDD.jl

## AS-inexact GFDN(a_u) benchmark

Study 10 compares GP-varDD with an inexact realization of the normalized
`a_u`-gradient flow for the Gross--Pitaevskii ground-state problem. At the
current normalized iterate `u_k`, define

```text
A(u_k) = K + beta*C(u_k),
r_k    = A(u_k)u_k - lambda_k*M*u_k.
```

An exact `a_u`-gradient step would require applying `A(u_k)^(-1)`. The
benchmark does not solve that global elliptic problem to a prescribed
tolerance. Instead, it applies the one-level additive Schwarz approximation
once,

```text
z_k = B_AS(u_k) r_k,     B_AS(u_k) approximately A(u_k)^(-1),
```

followed by projection onto the mass-tangent space and an energy-optimal
normalized line search. One application of `B_AS(u_k)` consists of `m`
independent local subdomain solves, which can run in parallel. The accelerated
variant adds a Fletcher--Reeves history direction and restarts when this is no
longer a descent direction. The figures therefore call the methods
**AS-inexact GFDN(a_u)** and **AS-inexact CG-GFDN(a_u)**.

This terminology follows Remark 5.14 ("inexact GFDN(a_u)") of Patrick Henning
and Elias Jarlebring, *The Gross--Pitaevskii Equation and Eigenvector
Nonlinearities: Numerical Methods and Algorithms*, SIAM Review 67(2), 2025.
The remark explains that the elliptic inverse required by an exact
`GFDN(a_u)` step can be replaced by a small amount of linear-solver work and
that this can substantially reduce the computational cost. Our one-AS
application is a domain-decomposition realization of that inexact principle.

Reference: [Henning--Jarlebring, SIAM Review, Remark 5.14](https://doi.org/10.1137/22M1516324)
