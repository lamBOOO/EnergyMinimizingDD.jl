Studies 8 and 9 use the same overlapping one-level partitions for all methods.
Study 8 shows additive varDD, additive varDD with one-vector history, and
multiplicative varDD separately. The additive local candidates are independent
and their `m` subdomain solves can run in parallel. Multiplicative varDD feeds
each local result into the next subdomain, giving it a serial critical path of
`m` local solves per sweep. The history variant requires no additional local
solves and enlarges only the small second-level problem. Study 9 additionally
includes post-combination damping with omega = 0.5.
The plotted cost is the number of local subdomain solves: one variational DD
sweep, one additive Schwarz stationary step, one RAS stationary step, and one
additive-Schwarz preconditioner application each count as `m` subdomain solves.
All methods shown are one-level methods, so their iteration counts are expected
to grow as the number of subdomains `m` increases. Two-level/coarse-space
extensions are left as future work.
