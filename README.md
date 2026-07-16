# VariationalDD.jl

## Henning--Jarlebring GPE benchmark

Study 10 reproduces the illustrative Gross--Pitaevskii example in section 2.3
of Henning--Jarlebring: `D=[-8,8]^2`, `kappa=500`, harmonic-plus-optical
potential (2.9), and their polynomial initial state. It includes the exact
normalized `a_u`-gradient flow from Definition 5.12 with the energy-optimal
step (5.30). At the current normalized iterate `u_k`, define

```text
A(u_k) = K + kappa*C(u_k),
r_k    = A(u_k)u_k - lambda_k*M*u_k.
```

The exact benchmark solves the global elliptic problem
`A(u_k)z_k=M*u_k` by sparse Cholesky and minimizes the energy along update
(5.28). On the documented `32 x 32` Q1 mesh (`h=0.5`), its energy error after 30
iterations is approximately `1e-9`, consistent with Figure 6. Since the paper
does not state the mesh used for that figure, the reported values
`E_GS≈10.8995` and `lambda_GS≈27.7133` are treated as refinement targets.
The comparison also includes **exact CG-GFDN(a_u)**, which combines the exact
metric solve with the same Fletcher--Reeves history, descent restart, and
energy-optimal line search used by the AS-inexact CG method.
The 3x3 comparison additionally includes `kappa=10` and `kappa=100`, using the
same setup and initial state; `kappa=500` remains the paper-reproduction row.
An opt-in `run_hj_mesh_validation()` study records exact-GFDN results up to
`N=256`; there it gives `E_h=10.9007980` and `lambda_h=27.7149190`. The
`h=1e-3` statement elsewhere in the paper refers to its separate 1D Figure 7
experiment, not this two-dimensional benchmark.

The AS-inexact comparison instead applies the one-level additive Schwarz
approximation once,

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

Reference: [Henning--Jarlebring, SIAM Review, sections 2.3 and 5.2.3](https://doi.org/10.1137/22M1516324)
