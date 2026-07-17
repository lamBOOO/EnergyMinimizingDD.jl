# Manufactured exponential semilinear Poisson benchmark

Study 11 solves

```math
-\Delta u = \exp(-u) + f \quad\text{in }(0,1)^2,
\qquad u=0\quad\text{on }\partial\Omega,
```

with the manufactured solution

```math
u_\star(x,y)
=1.5\sin(\pi x)\sin(\pi y)
+0.55\sin(2\pi x)\sin(3\pi y)
+0.35\sin(3\pi x)\sin(2\pi y),
```

and `f=-Delta u_star-exp(-u_star)`. The asymmetric higher modes introduce
multiple spatial scales and mildly sign-changing structure while preserving
the homogeneous boundary condition exactly.

The equation minimizes the strictly convex energy

```math
E(u)=\int_\Omega\left(\frac12|\nabla u|^2+\exp(-u)-fu\right)\,dx,
```

whose Hessian is

```math
E''(u)[w,v]=\int_\Omega \nabla w\cdot\nabla v+\exp(-u)wv\,dx.
```

The exponential is only an instance of the package's generic
`FEM_SemilinearPoisson` assembler, parameterized by a potential and its first
two derivatives. The resulting problem is represented by the existing
`NonlinearEnergy` type and solved through the same `var_dd`, `inf_step`, and
`combine_step` interface as the other source and eigenvalue examples.

The comparison uses triangular P1 elements, two triangle-edge layers of
overlap, and `m=2,4,8` METIS subdomains. Nonlinear AS, nonlinear RAS, varDD,
and varDD with one-vector history compute the same parallel batch of tightly
solved local energy minimizers. They differ only in the global combination;
the AS/RAS implementations are shared example infrastructure rather than
package-core solvers.
