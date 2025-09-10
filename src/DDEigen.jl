module DDEigen

using LinearAlgebra
using Gridap
using Metis
using GridapDistributed
using ThreadsX
using Arpack
using SparseArrays
using IterativeSolvers
using Plots

export Setup_FEM, create_dofs_partition, create_elements_partition,
       create_overlapping_elements_partition!, R, normalize_M!, pu_matrices,
       coarse_space_corr, inf_step, combine_step, M_norm_distance,
       ddm_eigen_solver

# Include original procedural script contents
include("../solve_fem.jl")

end # module
