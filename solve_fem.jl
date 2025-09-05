using LinearAlgebra
using Plots
using Gridap
using Metis
using GridapDistributed
using ThreadsX
using Arpack
using SparseArrays
  P(x)= exp(sqrt(x.data[1])^2+(x.data[2])^2)
function Setup_FEM(N::Int, m::Int=9) # Discretizing the domain, building mass and stiffness matrix, specifying the overlapping domains
    domain=(0, 1.0, 0, 1.0)
    partition1 = (1.0 * N, 1.0 * N)
    model = CartesianDiscreteModel(domain, partition1; isperiodic=(false, false))
    reffe = ReferenceFE(lagrangian, Float64, 1) 
    VV = TestFESpace(model, reffe, dirichlet_tags=["boundary"])
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    U = TrialFESpace(VV, 0)
    a1(u, v) = ∫(∇(u) ⋅ ∇(v) + (x -> P(x)) * u * v)dΩ
    a2(u, v) = ∫(u * v)dΩ
    K= assemble_matrix(a1, VV, U)
    M=assemble_matrix(a2, VV, U)
    g = GridapDistributed.compute_cell_graph(model)
    par = Metis.partition(g, m)
    elpar = create_elements_partition(par, m)
    create_overlapping_elements_partition!(elpar, g, m, 2)
    dofspar = create_dofs_partition(elpar, VV)
    return K,M, dofspar, VV
end

function create_dofs_partition(
  elemsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace
)
  m = sp.fe_basis.trian.model
  dim = size(m.grid_topology.n_m_to_nface_to_mfaces,2) - 1
  npars = length(elemsp)
  @debug "create nodesp"
  nodesp = [Vector{Int32}() for _ in 1:npars]
  Threads.@threads for ipar = 1:npars
    nodesp[ipar] = sort(unique(vcat([m.grid_topology.n_m_to_nface_to_mfaces[dim+1][el] for el in elemsp[ipar]]...)))
  end

  @debug "create freenodesp"
  freenodesp = copy(nodesp)
  Threads.@threads for ipar = 1:npars
    filter!(e -> e in sp.metadata.free_dof_to_node, nodesp[ipar])
  end

  @debug "create dofsp"
  reverse_map = zeros(Int32, maximum(sp.metadata.free_dof_to_node))
  for (node, freenode) in enumerate(sp.metadata.free_dof_to_node)
    reverse_map[freenode] = node
  end
  dofsp = [reverse_map[freenodesp[ipar]] for ipar = 1:npars]
  return dofsp
end

function create_elements_partition(partition::Vector{Int32}, npars::Integer) # Helper function from DDEigenlab
  nelems = length(partition)
  @debug nelems, length(partition)
  @assert nelems == length(partition)
  elemsp = [Vector{Int32}() for _ in 1:npars]
  for iel = 1:nelems
    push!(elemsp[partition[iel]], iel)
  end
  @debug nelems, sum(length.(elemsp))
  @assert nelems == sum(length.(elemsp))
  return elemsp
end

 function create_overlapping_elements_partition!(elemsp, g, npars::Integer, ol) # Helper function from DDEigenlab
  for iol = 1:ol
    @debug "overlap" iol
      Threads.@threads for ipar = 1:npars
      tmp = copy(elemsp)
      elemsp[ipar] = sort(unique(vcat([g[:, i].nzind for i in tmp[ipar]]...)))
    end
  end
end

# 2) Rayleigh quotient
function R(u::Vector{Float64}, K::AbstractMatrix, M::AbstractMatrix)
    numerator   = dot(u, K*u)
    denominator = dot(u, M*u)
    return numerator / denominator
end

# 3) Normalization in the M-norm
function normalize_M!(u::Vector{Float64}, M::AbstractMatrix)
    nu = sqrt(dot(u, M*u))
    @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
    u ./= nu
end

function pu_matrices(dofsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace) #pu as in Eigenlab
  npars = length(dofsp)
  Ri = Vector{SparseMatrixCSC}(undef, npars)
  Di = Vector{SparseMatrixCSC}(undef, npars)
  Threads.@threads for ipar = 1:npars
    Ri[ipar] = spzeros(length(dofsp[ipar]), sp.nfree)
    Di[ipar] = spzeros(length(dofsp[ipar]), length(dofsp[ipar]))
    for idof = 1:length(dofsp[ipar])
      Ri[ipar][idof, dofsp[ipar][idof]] = 1
      Di[ipar][idof, idof] = 1 / sum(map(p -> dofsp[ipar][idof] in p, dofsp))
    end
  end
  return Ri, Di
end

function coarse_space_corr(dofsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace)
  Ri,Di=pu_matrices(dofsp,sp)
  n=size(Di,1) # no. subdomains
  m=size(Ri[1],2) # no. of DOFS
  Z=zeros(Float64,m,n)
  for i=1:n
  Z[:,i]=(Ri[i]'*Di[i]*Ri[i])*ones(m) #Z as in Nicolaides in DD Book
  end
  return Z
end

# 5) Inf step on subspace D_i
function inf_step(u_current::Vector{Float64},
                  K::AbstractMatrix, M::AbstractMatrix,
                  idx_sub::AbstractVector)
    N = length(u_current)
    localdim = 1 + length(idx_sub)
    B = Matrix{Float64}(undef, N, localdim)

    # First column = current global vector
    B[:, 1] = u_current

    # Next columns = standard basis restricted to idx_sub
    for (k, j) in pairs(idx_sub)
        e = zeros(N)
        e[j] = 1.0
        B[:, 1 + k] = e
    end

    K_local = B' * (K * B)
    M_local = B' * (M * B)

    eigvals, eigvecs = eigen(K_local, M_local)
    i_min = argmin(eigvals)
    α_min = eigvecs[:, i_min]

    x_new = B * α_min
    normalize_M!(x_new, M)
    return x_new
end

# 6) Combine step
function combine_step(u_collection::Vector{Vector{Float64}},
                     K::AbstractMatrix, M::AbstractMatrix)
    B = hcat(u_collection...)
    B = Matrix(qr(B).Q)


    K_local = B' * (K * B)
    M_local = B' * (M * B)
    eigvals, eigvecs = eigen(K_local, M_local)
    i_min = argmin(eigvals)
    α_min = eigvecs[:, i_min]



    # println("eigen(K_local): ",eigen(K_local).values)
    # println("eigen(M_local): ",eigen(M_local).values)
    # println(α_min)
    x_new = B * α_min
    normalize_M!(x_new, M)
    return x_new
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::AbstractMatrix)
    w = u .- v
    return sqrt(dot(w, M*w))
end

# 7) Main iteration: store solutions & keep sign consistency
function ddm_eigen_solver(;
    N::Int=50,
    m::Int=2,
    maxiter::Int=50,
    tol::Float64=1e-8,
    #sweep::Bool=true
)

    K, M, subspaces,fesp= Setup_FEM(N,m)
    # Initial guess
    u_cur = ones((N-1)^2)
    normalize_M!(u_cur, M)

    # Track the Rayleigh quotient each iteration
    lambda_history = Float64[]
    # Also store the approximate eigenvector after each iteration
    solutions = Vector{Vector{Float64}}()

    λ_cur = R(u_cur, K, M)
    push!(lambda_history, λ_cur)
    push!(solutions, copy(u_cur))

    #println("Initial Rayleigh quotient = $λ_cur")

    #sub_int = 1:m
    for n in 1:maxiter
        # Local updates
        local_updates = Vector{Vector{Float64}}(undef, m+1)
        local_updates[1] = u_cur
        for i=1:m
            #println(i)

            # Additive
            # u_next_i = inf_step(local_updates[1], K, M, subspaces[i])

            # Multiplicative
            u_next_i = inf_step(local_updates[i], K, M, subspaces[i])

            # if dot(u_next_i, u_cur) < 0
            #     u_next_i .*= -1.0
            # end
            # normalize_M!(u_next_i, M)
            # x = range(0, 1, length = N+2)
            # u_plot = vcat(0.0, u_next_i, 0.0)

            # display(plot!(x, u_plot,
            # legend=false,
            # marker=:none))
            # sleep(1)

            local_updates[i+1] = u_next_i
        end
        #if sweep
       #     sub_int = -sub_int.+(m+1)
        #end
        coarse_basis= coarse_space_corr(subspaces,fesp)
        # Combine step
        u_new = combine_step([coarse_basis,local_updates], K, M)

        # u_new = u_cur
        # for i in 1:m
        #     u_new += (local_updates[i+1]-u_cur)
        # end
        # normalize_M!(u_cur, M)
        # x = range(0, 1, length = N+2)
        # u_plot = vcat(0.0, u_cur, 0.0)

        # display(plot!(x, u_plot,
        # legend=false,
        # marker=:none))
        # sleep(1)

        # u_new = u_cur + sum(local_updates)

        # --- SIGN FIX to avoid solution flipping from iteration to iteration ---
        if dot(u_new, u_cur) < 0
            u_new .*= -1.0
        end

        λ_new = R(u_new, K, M)
        push!(lambda_history, λ_new)
        push!(solutions, copy(u_new))

        #println(abs(λ_new - λ_cur))
        if abs(λ_new - λ_cur) < tol
            println("Converged at iteration $n with eigenvalue λ = $λ_new")
            return u_new, λ_new, lambda_history, solutions
        end

        u_cur .= u_new
        λ_cur = λ_new
    end

    println("Reached maxiter=$maxiter with final Rayleigh quotient ≈ $λ_cur")
    return u_cur, λ_cur, lambda_history, solutions
end


###############################################################################
# Run the solver
###############################################################################
N=20
m       = 9
maxiter = 200
tol     = 1e-10


u_approx, lambda_approx, lambda_history, solutions = ddm_eigen_solver(
    N=N,
    m=m,
    maxiter=maxiter,
    tol=tol
)
K,M,part=Setup_FEM(N,m)

println("Final approximate eigenvalue = $lambda_approx")

#  -- Plot 1: Convergence of the Rayleigh quotient --
iters = 0:length(lambda_history)-1

plt1 = plot(
   iters, lambda_history,
  marker = :o,
    xlabel = "Iteration",
    ylabel = "Rayleigh Quotient",
    title  = "Convergence of Eigenvalue (m=$m, N=$N)"
)
#  -- Plot 2: Convergence in the eigenvector (M-norm) --
final_sol = solutions[end]
exact_sol = eigs(K, nev=1, which=:LM)
println(typeof(exact_sol))
exact_val= exact_sol[1]
println(exact_val)
exact_vec=exact_sol[end]
distances = [
   M_norm_distance(solutions[i], exact_vec, M)
    for i in 1:length(solutions)
]
plt2 = plot(
    iters, distances,
   marker = :o,
   xlabel = "Iteration",
    ylabel = "||u^(k) - u^(exact)||_M",
    title  = "Convergence of the Eigenvector in M-norm",
   yaxis = :log
)
#inverse power method
function inverse_power_method2(K::Matrix{Float64}, M::Matrix{Float64}, u0::Vector{Float64}, maxiter::Int=100, tol::Float64=1e-10, λ::Float64=0.0)
    """
    Inverse power method to find the smallest eigenvalue and corresponding eigenvector
    of the generalized eigenvalue problem Kx = λMx.
    """
    u = copy(u0)
    normalize_M!(u, M)

    for i in 1:maxiter
        # Solve the linear system (K - λM)x = 0
        # Here we use the Rayleigh quotient as an approximation for λ

        A = K - λ * M
        u_new = A \ u

        # Normalize the new vector
        normalize_M!(u_new, M)

        λ = R(u, K, M)
        @info "Iteration $i: λ = $λ"

        # Check convergence
        dist = M_norm_distance(u_new, u, M)
        if dist < tol
            println("Converged at iteration $i with eigenvalue λ ≈ $λ")
            return u_new, λ
        end

        u = copy(u_new)
    end

    println("Reached maxiter=$maxiter with final Rayleigh quotient ≈ $λ")
    return u, λ
end
