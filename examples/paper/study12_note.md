# Proposed fifth numerics subsection (not inserted into the paper)

```latex
\subsection{Variants of the method}

We finally compare several variants of the EMDD iteration on the Poisson
problem from the first experiment.  In addition to the standard additive EMDD
sweep, we consider a multiplicative sweep, in which the local minimizers are
computed successively from the most recent iterate; restricted EMDD (REMDD),
in which overlapping local corrections are weighted by the algebraic
partition of unity before the second-level minimization; enrichment by the
multiplicity partition of unity; enrichment by a discrete-harmonic Nicolaides
coarse space; and the combination of REMDD with the Nicolaides space.  We use
$q=2$ throughout so that only the named algorithmic variant changes.

The comparison follows the weak-scaling setup used for the coarse-space study.
The square domain is partitioned into a $\sqrt m\times\sqrt m$ Cartesian
subdomain grid, with $m=4,16,64$.  Each core contains ten cells per coordinate
direction and the overlap is fixed at two fine-grid layers.  Hence the global
meshes have $1/h=20,40,80$, while both $H/h=10$ and $H/\delta=5$ remain fixed.
All variants use the same all-ones initial coefficient vector and a relative
residual tolerance of $10^{-10}$.

Figure~\ref{fig:method-variants} shows a clear separation as the number of
subdomains grows.  Plain EMDD requires 16, 23, and 36 sweeps for
$m=4,16,64$, respectively, and REMDD requires 8, 17, and 34.  Enrichment by
the multiplicity partition of unity reduces these counts to 13, 23, and 26,
whereas the discrete-harmonic Nicolaides space gives 14, 22, and 23 sweeps.
The best parallel variant for the larger decompositions is the hybrid
REMDD--Nicolaides method, which requires 10, 16, and 16 sweeps and therefore
shows no deterioration between $m=16$ and $m=64$ in this experiment.  The
multiplicative method converges in 7, 10, and 20 sweeps.  Its low sweep count
must be interpreted together with its serial critical path: its $m$ local
problems are solved successively, whereas the local problems of all other
variants can be solved concurrently.  At $m=64$, REMDD--Nicolaides consequently
uses both fewer outer sweeps and a parallel local stage.  The results indicate
that the harmonic coarse space becomes increasingly valuable as global
low-frequency communication across the subdomains becomes the limiting
mechanism.

\begin{figure}[t]
  \centering
  \includegraphics[width=1.0\textwidth]{figures/fig28_method_variants}
  \caption{Convergence of EMDD variants in a weak-scaling Poisson experiment
  with $m=4,16,64$ Cartesian subdomains, $1/h=20,40,80$, two overlap layers,
  and $q=2$.  The refinement keeps $H/h=10$ and $H/\delta=5$ fixed.  A
  multiplicative sweep contains $m$ sequential local solves, whereas the local
  solves of all other variants can be performed in parallel.}
  \label{fig:method-variants}
\end{figure}
```
