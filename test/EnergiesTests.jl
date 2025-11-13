using VariationalDomainDecomposition.Energies
using LinearAlgebra
using FiniteDiff
using Gridap
using GridapDistributed
using Metis
using IterativeSolvers
using Arpack
using Printf
using Random
using LineSearches

# Example usage:
# using .Energies
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
E = Energies.QuadraticEnergy(A, b)
x = [0.5, 0.5]
E(x)                     # Evaluate energy
Energies.gradient(E, x)  # Compute gradient
Energies.hessian(E)      # Get Hessian (constant)

# Example usage of RayleighQuotient
RQ = Energies.RayleighQuotient(A)
RQ(x)                    # Evaluate Rayleigh quotient
Energies.gradient(RQ, x) # Compute gradient
Energies.hessian(RQ, x)   # Hessian implemented

# Test

# Make matrix exactly symmetric to avoid issues
A_sym = Symmetric(A)
RQ_sym = Energies.RayleighQuotient(A_sym)

# Check gradient first
@assert FiniteDiff.finite_difference_gradient(
  z -> Energies.energy(RQ_sym, z),
  x;
  absstep = 1e-8,
) - Energies.gradient(RQ_sym, x) |> norm < 1E-6

# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(
  z -> Energies.gradient(RQ_sym, z),
  x;
  absstep = 1e-8,
) - Energies.hessian(RQ_sym, x) |> norm < 1E-6

# Check Hessian in-place
H = zeros(length(x), length(x))
Energies.hessian!(H, RQ_sym, x)
@assert FiniteDiff.finite_difference_jacobian(
  z -> Energies.gradient(RQ_sym, z),
  x;
  absstep = 1e-8,
) - H |> norm < 1E-6

# Check GeneralizedRayleighQuotient
B = [2.0 0.0; 0.0 1.0]
GRQ = Energies.GeneralizedRayleighQuotient(A_sym, Symmetric(B))
GRQ(x)                    # Evaluate generalized Rayleigh quotient
Energies.gradient(GRQ, x) # Compute gradient
Energies.hessian(GRQ, x)   # Hessian implemented
# Check gradient first
@assert FiniteDiff.finite_difference_gradient(
  z -> Energies.energy(GRQ, z),
  x;
  absstep = 1e-8,
) - Energies.gradient(GRQ, x) |> norm < 1E-6
# Check Hessian
@assert FiniteDiff.finite_difference_jacobian(
  z -> Energies.gradient(GRQ, z),
  x;
  absstep = 1e-8,
) - Energies.hessian(GRQ, x) |> norm < 1E-6
