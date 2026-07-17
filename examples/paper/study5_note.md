# Study 5: homogeneous p-Laplacian

Study 5 uses the homogeneous specialization of the p-Laplacian model problem
from Heinlein, Klawonn, and Lanser,

\[
-\nabla\!\cdot\!\left((\varepsilon^2+|\nabla u|^2)^{(p-2)/2}\nabla u\right)=1
\quad\text{in }(0,1)^2,
\qquad u=0\quad\text{on }\partial\Omega.
\]

It deliberately sets the diffusion coefficient to one and uses no coarse
space. The purpose is to compare one-level nonlinear combination rules, not
robustness with respect to coefficient contrast. The discretization uses
triangular P1 elements, two triangle-edge layers of overlap, and METIS
decompositions with `m=2,4,8` subdomains. METIS acts directly on the dual
graph of the triangular mesh. The figure uses rows `p=2,3,4` and columns
`m=2,4,8`, matching the layout of the linear-system, EVP, and GPE comparisons.
The inset draws the actual triangles and overlays their overlap multiplicity;
the `p=2` row recovers the Poisson energy.

At every outer iteration all methods compute the same batch of local energy
minimizers with the exterior degrees of freedom fixed. They differ only in the
global update:

- nonlinear AS minimizes along the raw sum of the local corrections;
- nonlinear RAS minimizes along their ownership-restricted sum. Its disjoint
  DOF ownership is induced by the original nonoverlapping METIS element cores;
  shared interface DOFs are assigned to an adjacent core with balanced
  tie-breaking before overlap is added;
- varDD minimizes the nonlinear energy over the linear span of the current
  iterate and all local solution candidates via the package's common
  `combine_step` interface;
- varDD + history additionally includes the previous global iterate.

Both Schwarz baselines use an energy-optimal scalar damping parameter. One
outer iteration therefore means one parallel batch of local nonlinear solves
plus one combination step for every method. The AS/RAS implementations are
study-local; varDD uses the same `var_dd`, `inf_step`, and `combine_step`
package interface as the linear source, linear EVP, and nonlinear EVP cases.

The curves stop at a relative gradient norm of `1e-7` or 30 outer iterations.
A monolithic safeguarded Newton solve supplies the discrete reference energy
but is not plotted as a competing method.

Reference: [Heinlein--Klawonn--Lanser, *Adaptive Nonlinear Domain
Decomposition Methods with an Application to the p-Laplacian*, SIAM Journal
on Scientific Computing 45(3), 2023](https://doi.org/10.1137/21M1433605).
