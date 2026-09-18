# Study 11: nonlinear source solver comparison

All methods solve the same triangular P1 discretization of

\[
-\Delta u+\beta u^3=f,\qquad \beta\in\{1,10,100\},\qquad
u|_{\partial\Omega}=0,
\]

from the same zero initial iterate and stop when the norm of the assembled
Euler residual has decreased by the same prescribed relative factor. The
overlapping and nonoverlapping DOF sets are exactly those used by varDD.

For varDD, each local candidate minimizes the energy over
`V_i + span{u_k}` with gradient-only L-BFGS (relative tolerance `1e-10`,
absolute tolerance `1e-12`, at most 200 iterations). The nonlinear Schwarz,
ASPIN, and RASPEN baselines retain the standard affine Dirichlet space
`u_k + V_i`; their local maps are solved by the existing damped Newton
routine.

The comparison methods are deliberately kept in the example layer rather than
the package solver API:

- **Anderson--RAS(m)** uses NonlinearSolve.jl's
  `FixedPointAccelerationJL(algorithm=:Anderson, m=m)` adapter for the
  optimally damped nonlinear RAS fixed-point map. The package owns the
  extrapolation, conditioning checks, and stopping logic; this study only
  converts its iterate history to the common output schema.
- **AS--nonlinear CG (Optim.jl)** uses Optim.jl's
  `ConjugateGradient`, which implements Hager--Zhang CG-DESCENT with the
  package's Hager--Zhang line search. Optim's supported `P`/`precondprep`
  interface refreshes the current Hessian blocks and applies one parallel
  additive-Schwarz batch to every search direction. Optim.jl does not expose
  Fletcher--Reeves or Polak--Ribiere variants, so they are not presented as
  package-generated curves.
- **Newton--PCG(AS, ν)** forms the exact current Hessian, applies `ν` steps of
  PCG with the one-level additive Schwarz preconditioner, and globalizes the
  inexact Newton direction by an Armijo energy line search. The `ν=accurate`
  experiment uses a relative inner residual tolerance of `1e-10`. The main
  comparison shows `ν=1`, `ν=2`, `ν=4`, and `ν=8` separately.
- **quadratic EMDD** freezes the second-order Taylor model of the nonlinear
  energy at the start of each outer sweep. Its local candidates and global
  recombination minimize that quadratic model over the same enriched spaces
  as plain EMDD, while stopping and convergence histories use the original
  nonlinear energy and residual. The comparison includes both `q=1` and the
  one-vector-history variant `q=2`.

The focused figure treats standard EMDD, quadratic-model EMDD, and
Anderson--RAS as the primary DD comparison. The Optim.jl AS-preconditioned
nonlinear CG method and Newton--PCG are shown as visually de-emphasized
linear-AS references. Using a quadratic
local model does not by itself give quadratic convergence to quadratic-model
EMDD.
- **energy-IMEX--PCG(AS)** is the stabilized pseudo-time linearization of
  Spicher and Wihler [SpicherWihler2026], §3.1--3.2, equations (3.1) and (3.7):
  \[
  (M/\Delta t+K)u^{n+1}=Mu^n/\Delta t+Ku^n-g(u^n).
  \]
  Here `Δt=1`; every linear problem is solved to relative tolerance `1e-10`
  by PCG(AS). This is an energy-stable global fixed-point benchmark, not a
  nonlinear domain-decomposition method.
- **RASPEN** [DoleanEtAl2016] applies Newton to the restricted nonlinear
  Schwarz correction.
  Its Jacobian action is differentiated analytically through the local
  Dirichlet solves and solved by matrix-free GMRES. This follows the RASPEN
  formulation rather than approximating the nonlinear-preconditioned Jacobian
  by finite differences.
- **ASPIN** [CaiKeyes2002] uses the classical additive nonlinear Schwarz
  correction and the standard approximate Jacobian assembled from the current
  global Hessian and its overlapping local blocks. The same energy line search
  globalizes it.
  Showing it beside RASPEN is useful precisely because RASPEN instead starts
  from the restricted Schwarz map and differentiates the nonlinear local
  problems exactly.

## Fairness and reported work

An outer iteration is shown because it is the common algorithmic convergence
measure, but it is not a common cost unit. Therefore the CSVs and figures keep
the following counters separate:

1. parallel **nonlinear local-minimization batches** (Anderson--RAS, ASPIN,
   RASPEN, and varDD),
2. parallel **linear AS-solve batches** (AS--nonlinear CG, Newton, energy-IMEX,
   ASPIN, and the RASPEN Jacobian solve, as well as the linear local batches of
   quadratic EMDD), and
3. global Jacobian/operator products.

A nonlinear local minimization is not counted as one linear triangular solve.
RASPEN additionally needs one nonlinear local batch per outer iteration; each
GMRES Jacobian action then needs a parallel batch of local linearized solves.

The controlled sensitivity study uses:

- `m = 4, 16, 64` on the `N = 64` mesh in the main convergence figure;
- `N = 16, 32, 64`, corresponding to `h, h/2, h/4`, with overlap layers
  `1, 2, 4` so that the physical overlap is fixed;
- Newton inner work `ν = 1, 2, 4, 8, accurate`;
- Anderson and varDD stored-history depths `0, 1, 2, 4, 8`. For varDD these
  correspond to paper history sizes `q = 1, 2, 3, 5, 9` because `q` also
  counts the current iterate. The observed outer iteration counts are
  `53, 22, 22, 20, 20`: `q = 2` captures the substantial improvement, and
  larger history spaces save at most two additional sweeps in this experiment.

## References

Complete BibTeX records are collected in
[`references.bib`](references.bib). The citation keys used above map as follows:

- `Anderson1965` and `WalkerNi2011`: Anderson acceleration;
- `Pulay1980`: DIIS/Pulay mixing;
- `CaiKeyes2002`: ASPIN;
- `DoleanEtAl2016`: RASPEN and its comparison with ASPIN;
- `SpicherWihler2026`: stabilized energy-IMEX finite-element iteration.
