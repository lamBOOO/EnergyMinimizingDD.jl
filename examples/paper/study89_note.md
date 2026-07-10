Studies 8 and 9 use the same overlapping one-level partitions for all methods.
The plotted cost is the number of local subdomain solves: one variational DD
sweep, one additive Schwarz stationary step, one RAS stationary step, and one
additive-Schwarz preconditioner application each count as `m` subdomain solves.
All methods shown are one-level methods, so their iteration counts are expected
to grow as the number of subdomains `m` increases. Two-level/coarse-space
extensions are left as future work.
