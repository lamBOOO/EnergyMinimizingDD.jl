using LinearAlgebra
using Plots
using Gridap
using Metis
using GridapDistributed
using ThreadsX
using Arpack
using SparseArrays
using IterativeSolvers

P(x)= exp(sqrt((x.data[1])^2+(x.data[2])^2))

function Setup_FEM(N::Int, m::Int=9; overlap::Int=2, nonoverlapping::Bool=false) # Discretizing the domain, building mass and stiffness matrix, specifying the overlapping domains
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
    create_overlapping_elements_partition!(elpar, g, m, overlap)
    t1=time()
    dofspar = create_dofs_partition(elpar, VV; nonoverlapping=nonoverlapping)
    elapsed=time()-t1
    println("dofspar needs $elapsed seconds ")
    return K,M, dofspar, VV
end

function create_dofs_partition(
  elemsp::Vector{Vector{Int32}}, sp::Gridap.FESpaces.UnconstrainedFESpace;
  nonoverlapping::Bool=false
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

  # Remove duplicates if nonoverlapping mode is enabled
  if nonoverlapping
    @debug "removing duplicate nodes for nonoverlapping partition"
    # Track which nodes have been assigned to a subdomain
    assigned_nodes = Set{Int32}()
    for ipar = 1:npars
      # Keep only nodes that haven't been assigned to a previous subdomain
      freenodesp[ipar] = filter(node -> !(node in assigned_nodes), freenodesp[ipar])
      # Mark these nodes as assigned
      union!(assigned_nodes, freenodesp[ipar])
    end
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
  Z=zeros(m,n)
  for i=1:n
  Z[:,i]=(Ri[i]'*Di[i]*Ri[i])*ones(m) #Z as in Nicolaides in DD Book
  end
  return Z
end

# 5) Inf step on subspace D_i
"""
  inf_step(u_current, K, M, idx_sub)

Mathematical description
------------------------
Given symmetric positive definite matrices `K, M ∈ R^{N×N}` (stiffness and mass)
and the current M-normalized iterate `u_current ∈ R^N` (i.e. `u_current' * M * u_current = 1`),
let `S = span{ u_current, e_j : j ∈ idx_sub } ⊂ R^N`, where `e_j` are the Euclidean coordinate vectors.

This routine computes the (M-orthonormal) vector

  x_new = argmin_{ x ∈ S, x ≠ 0 }  R(x),   where   R(x) = (x' K x)/(x' M x),

restricted to the subspace `S`. Equivalently, writing any `x ∈ S` as

  x = α₀ u_current + ∑_{j ∈ idx_sub} α_j e_j  =  B α,   with   B = [u_current  |  E_sub],

and collecting coefficients `α = (α₀, (α_j)_{j∈idx_sub}) ∈ R^{1+|idx_sub|}`, the Rayleigh quotient on `S` becomes

  R(B α) = (α' (B' K B) α)/(α' (B' M B) α)  = (α' K_local α)/(α' M_local α).

Thus `α_min` is the generalized eigenvector corresponding to the smallest eigenvalue λ_min solving

  K_local α = λ M_local α,

with the normalization convention imposed afterwards by scaling `x_new` to satisfy `x_new' * M * x_new = 1`.

Implementation details
----------------------
Instead of explicitly forming the basis matrix `B`, the small dense matrices

  K_local = B' K B,   M_local = B' M B ∈ R^{(1+|idx_sub|)×(1+|idx_sub|)}

are assembled by exploiting the structure of the basis vectors (one dense vector plus coordinate vectors).
The smallest generalized eigenpair is approximated with a single-vector LOBPCG call (preconditioner `chol(K_local)`).
The resulting coefficients `α_min` yield

  x_new = α_min[1] * u_current + ∑_{k=1}^{|idx_sub|} α_min[k+1] * e_{idx_sub[k]},

followed by in-place M-normalization. Returned `x_new` satisfies

  x_new' * M * x_new = 1,    R(x_new) = λ_min = min_{x∈S \\ {0}} R(x).

Arguments
---------
* `u_current::Vector{Float64}` : Current M-normalized iterate (length N).
* `K::AbstractMatrix`          : SPD stiffness matrix.
* `M::AbstractMatrix`          : SPD mass matrix.
* `idx_sub::AbstractVector`    : Indices of degrees of freedom defining the local augmentation subspace.

Returns
-------
* `x_new::Vector{Float64}` : Updated vector in `S` minimizing the Rayleigh quotient (M-normalized).

Notes
-----
* If `idx_sub` is empty, the subspace reduces to `span{u_current}` and the function returns `u_current`.
* The step is an exact (within solver tolerance) subspace minimization of the Rayleigh quotient; it never increases the minimal value over `S`.
"""
function inf_step(u_current::Vector{Float64},
                  K::AbstractMatrix, M::AbstractMatrix,
                  idx_sub::AbstractVector, nev::Int64)
    t_build_start = time()
    localdim = 1 + length(idx_sub)

    # More efficient: avoid creating standard basis vectors explicitly
    # Instead, extract submatrix directly from K and M
    t_build = time() - t_build_start

    t_local_matrices_start = time()
    # Efficient approach: work directly with submatrices instead of building B
     # Current solution + subdomain indices

    # Extract relevant rows/columns from K and M
    if length(idx_sub) > 0
        # Build the local matrices more efficiently
        K_local = zeros(localdim, localdim)
        M_local = zeros(localdim, localdim)

        # First row/column: u_current' * K/M * [u_current, e_j1, e_j2, ...]
        K_u = K * u_current
        M_u = M * u_current

        K_local[1, 1] = dot(u_current, K_u)  # u' * K * u
        M_local[1, 1] = dot(u_current, M_u)  # u' * M * u

        # First row/column: u_current' * K/M * e_j
        for (k, j) in pairs(idx_sub)
            K_local[1, k+1] = K_u[j]  # u' * K * e_j = (K * u)[j]
            K_local[k+1, 1] = K_u[j]  # e_j' * K * u = (K * u)[j] (symmetric)
            M_local[1, k+1] = M_u[j]  # u' * M * e_j = (M * u)[j]
            M_local[k+1, 1] = M_u[j]  # e_j' * M * u = (M * u)[j] (symmetric)
        end

        # Remaining entries: e_i' * K/M * e_j = K[i,j] and M[i,j]
        for (k1, j1) in pairs(idx_sub)
            for (k2, j2) in pairs(idx_sub)
                K_local[k1+1, k2+1] = K[j1, j2]
                M_local[k1+1, k2+1] = M[j1, j2]
            end
        end
    else
        # Degenerate case: only current solution
        K_local = reshape([dot(u_current, K * u_current)], 1, 1)
        M_local = reshape([dot(u_current, M * u_current)], 1, 1)
    end

    t_local_matrices = time() - t_local_matrices_start

    t_eigen_start = time()
    @debug "Size of K_local: $(size(K_local))"
    @debug "Size of M_local: $(size(M_local))"
    K_local_sym = Symmetric(K_local);  M_local_sym = Symmetric(M_local)         # if applicable
    F = cholesky(K_local_sym)                              # ≈ A^{-1} preconditioner
    α_min=Vector{Vector{Float64}}(undef, nev)
    x_new=Vector{Vector{Float64}}(undef, nev)
    res = lobpcg(K_local_sym, M_local_sym, false, nev; P=F, tol=1e-8, maxiter=500)  # false = search smallest
    λmin = res.λ[1]
                       
    for i=1:nev
      α_min[i]=res.X[:,i]      # already B-orthonormal
    end
    # eigvals, eigvecs = eigen(K_local, M_local)
    # i_min = argmin(eigvals)
    # α_min = eigvecs[:, i_min]
    t_eigen = time() - t_eigen_start

    t_finalize_start = time()
    # Reconstruct the solution without explicit B matrix
    for i=1:nev
    x_new[i] = α_min[i][1] * u_current # Coefficient for current solution
    end  

    # Add contributions from standard basis vectors
    for i=1:nev
    for (k, j) in pairs(idx_sub)
        x_new[i][j] += α_min[i][k+1]  # Add coefficient for e_j
    end
  end
    for i=1:nev
    normalize_M!(x_new[i], M)
    end
    t_finalize = time() - t_finalize_start

    # Only print detailed timing for slow operations (> 0.01 seconds)
    if t_build + t_local_matrices + t_eigen + t_finalize > 0.01
        @debug "inf_step breakdown: build=$t_build, matrices=$t_local_matrices, eigen=$t_eigen, finalize=$t_finalize"
    end

    return x_new
end

# 6) Combine step
"""
  combine_step(u_collection, K, M)

Given a collection of M-normalized (not necessarily mutually orthogonal) vectors
`{u_i}` this forms the matrix `B = [u_1 ... u_p]`, computes an orthonormal (in
the Euclidean sense) basis `Q` of its column space via QR, and then solves the
reduced generalized eigenproblem

  (Q' K Q) α = λ (Q' M Q) α

returning the vector `x_new = Q α_min` associated with the smallest Rayleigh
quotient restricted to span(B). The output is re-normalized in the M-norm.

Mathematically this performs the exact minimization

  x_new = argmin_{x ∈ span(u_collection) \\ {0}} (x' K x)/(x' M x).

Returns the updated vector `x_new` with `x_new' * M * x_new = 1`.
"""
function combine_step(u_collection::Matrix{Float64},
                     K::AbstractMatrix, M::AbstractMatrix)
    t_qr_start = time()
    
    B = Matrix(qr(u_collection).Q)
    t_qr = time() - t_qr_start

    t_matrices_start = time()
    # Optimized matrix multiplications using temporary arrays and mul!
    localdim = size(B, 2)
    N = size(B, 1)

    # Pre-allocate temporary matrices
    temp_K = Matrix{Float64}(undef, N, localdim)
    temp_M = Matrix{Float64}(undef, N, localdim)
    K_local = Matrix{Float64}(undef, localdim, localdim)
    M_local = Matrix{Float64}(undef, localdim, localdim)

    # Use mul! for in-place operations
    mul!(temp_K, K, B)
    mul!(temp_M, M, B)
    mul!(K_local, B', temp_K)
    mul!(M_local, B', temp_M)

    t_matrices = time() - t_matrices_start

    t_eigen_start = time()
    eigvals, eigvecs = eigen(K_local, M_local)
    i_min = argmin(eigvals)
    α_min = eigvecs[:, i_min]
    t_eigen = time() - t_eigen_start

    # println("eigen(K_local): ",eigen(K_local).values)
    # println("eigen(M_local): ",eigen(M_local).values)
    # println(α_min)

    t_finalize_start = time()
    x_new = B * α_min
    normalize_M!(x_new, M)
    t_finalize = time() - t_finalize_start

    @debug "combine_step breakdown: QR=$t_qr, matrices=$t_matrices, eigen=$t_eigen, finalize=$t_finalize"
    return x_new
end

# A helper function for measuring "distance" in M-norm
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::AbstractMatrix)
    w = u .- v
    return sqrt(dot(w, M*w))
end

# 7) Main iteration: store solutions & keep sign consistency
"""
   ddm_eigen_solver(; N=50, m=2, maxiter=50, tol=1e-8)

High-level driver performing a domain decomposition enhanced iterative
minimization of the Rayleigh quotient for the generalized eigenproblem

   K u = λ M u.

Workflow per iteration k:
1. Local ("infinite") steps: For each subdomain i build the augmented subspace
  `span{u^{(k)}_{i}, e_j (j in subspace i)}` and apply `inf_step`, either
  additively or (current code) multiplicatively chained.
2. Coarse correction: Append a coarse partition of unity based basis and call
  `combine_step` to perform a global small eigen solve restricted to the span
  of all local updates plus coarse vectors.
3. Sign stabilization: Flip sign if necessary to keep consecutive iterates
  aligned (to avoid oscillations due to eigenvector indeterminacy).
4. Convergence test: stop when |λ_{k+1} - λ_k| < tol.

Returns `(u, λ, lambda_history, solutions)` where `solutions` stores all
intermediate iterates and `lambda_history` the Rayleigh quotients.
"""
function ddm_eigen_solver(;
    N::Int=50,
    m::Int=2,
    maxiter::Int=50,
    tol::Float64=1e-8,
    overlap::Int=2,
    nonoverlapping::Bool=false,
    #sweep::Bool=true
)
    setup_time=time()
    K, M, subspaces,fesp= Setup_FEM(N,m; overlap=overlap, nonoverlapping=nonoverlapping)
    elapsed_setup=time()-setup_time
    println("$elapsed_setup seconds needed for setup")
    coarse_basis= coarse_space_corr(subspaces,fesp)

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
    elapsedtot=0
    #sub_int = 1:m
    for n in 1:maxiter
      t3= time()
        nev=1
        # Local updates
        t_local_start = time()
        local_updates = zeros(size(u_cur,1), (nev*m)+1)
        local_updates[:,1] = u_cur
        for i=1:m
            t_inf_step_start = time()
            l=1
            #println(i)

            # Additive
            # u_next_i = inf_step(local_updates[1], K, M, subspaces[i])

            # Multiplicative
            u_next_i = inf_step(local_updates[:,i+(l-1)*(nev-1)], K, M, subspaces[i], nev)            # if dot(u_next_i, u_cur) < 0
            #     u_next_i .*= -1.0
            # end
            # normalize_M!(u_next_i, M)
            # x = range(0, 1, length = N+2)
            # u_plot = vcat(0.0, u_next_i, 0.0)

            # display(plot!(x, u_plot,
            # legend=false,
            # marker=:none))
            # sleep(1)

            for j=1:nev
              local_updates[:,(i-1)*nev+j+1]=u_next_i[j]
            end
            t_inf_step = time() - t_inf_step_start
            @debug "inf_step for subdomain $i took $t_inf_step seconds"
            l=l+1
        end
        t_local_updates = time() - t_local_start
        @debug "All local updates took $t_local_updates seconds"
        #if sweep
       #     sub_int = -sub_int.+(m+1)
        #end

        # Combine step with coarse correction
        t_combine_start = time()
        @debug "size of local updates is $(length(local_updates))"
        @debug "size of coarse basis is $(length(coarse_basis))"
        combined_matrix=hcat(coarse_basis,local_updates)
        u_new = combine_step(combined_matrix, K, M)
        t_combine = time() - t_combine_start
        @debug "combine_step took $t_combine seconds"

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
        time_it=time()-t3
        println("Needed $time_it seconds for iteration $n")
        elapsedtot= elapsedtot+(time()-t3)
        #println(abs(λ_new - λ_cur))
        if abs(λ_new - λ_cur) < tol
            println("Converged at iteration $n with eigenvalue λ = $λ_new")
            av_time=elapsedtot/n
            println("Average time pro iteration is $av_time seconds")
            return u_new, λ_new, lambda_history, solutions
        end

        u_cur .= u_new
        λ_cur = λ_new
    end

    println("Reached maxiter=$maxiter with final Rayleigh quotient ≈ $λ_cur")
    return u_cur, λ_cur, lambda_history, solutions
end


# Auto-run block:
# Runs when (a) the file is executed as a script, or (b) we are in an interactive
# session (e.g. VSCode Cmd+R / REPL include) unless explicitly disabled by
# setting ENV["DDEIGEN_SKIP_AUTORUN"] = "1".
# It still skips during Documenter builds (non-interactive include).
if (abspath(PROGRAM_FILE) == @__FILE__) || (isinteractive() && get(ENV, "DDEIGEN_SKIP_AUTORUN", "0") != "1")
  ###############################################################################
  # Run the solver (only when executed as a script)
  ###############################################################################
  N=100
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
end
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
