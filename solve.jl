using LinearAlgebra
using Plots

# 1) Define the 1D mesh and finite difference matrices
function laplace_eig_matrices(N::Int; m::Int=9)
    """
    Set up the mass matrix M and stiffness matrix K for the
    1D Laplace operator -u'' on (0,1) with Dirichlet boundary
    conditions using N interior points.
    """
    # Spatial step
    h = 1.0 / (N+1)

    # Stiffness matrix (K ~ -d^2/dx^2)
    diag_main = fill(2.0, N)
    diag_off  = fill(-1.0, N-1)
    K = diagm(0 => diag_main, 1 => diag_off, -1 => diag_off)
    # Scale by 1/h^2
    K .= (1/h^2) .* K

    x = range(h, 1-h, length = N)
    v = fill(2.0, N)
    σ = 0.01
    H = 1/m
    for i in 0:m-1
    #   v -= exp.(-(x.-0.5).*(x.-0.5)./(2*σ))
        v -= exp.(-abs.(x.-(H/2+i*H))./(2*σ))
    end
    K += 1000*diagm(0 => v)

    display(plot!(x, v,
            legend=false,
            marker=:none))
            sleep(1)

    
            # u_plot = vcat(0.0, u_next_i, 0.0)

    # Mass matrix: the identity times 1.0 (per your request)
    M = Matrix(I, N, N) .* 1.0

    return K, M, v
end

# 2) Rayleigh quotient
function R(u::Vector{Float64}, K::Matrix{Float64}, M::Matrix{Float64})
    numerator   = dot(u, K*u)
    denominator = dot(u, M*u)
    return numerator / denominator
end

# 3) Normalization in the M-norm
function normalize_M!(u::Vector{Float64}, M::Matrix{Float64})
    nu = sqrt(dot(u, M*u))
    @assert nu > 1e-14 "Attempting to normalize a near-zero vector."
    u ./= nu
end

# 4) Domain decomposition: subdivide into m blocks
function subspace_indices(N::Int, m::Int; overlap::Int=10)
    """
    Partition the indices 1..N into m subspaces with an overlap of 'overlap' points
    between adjacent subdomains.
    """
    # Basic size for each block (no overlap)
    size_block = div(N, m)

    subs = Vector{Vector{Int}}(undef, m)
    startidx = 1
    for i in 1:m
        # Normally stopidx would be startidx+size_block-1
        # but we’ll build the base block, then add overlap.
        stopidx = (i < m) ? (startidx + size_block - 1) : N

        # Now define the block with the overlap region
        #   - For domain i, we can extend it by 'overlap' points
        #     at the high end (except maybe for the last subdomain).
        #   - Similarly, we can shift the start backwards by 'overlap'
        #     for subdomains after the first.
        # This is just one possible pattern.
        actual_start = max(1, startidx - overlap)
        actual_stop  = min(N, stopidx + overlap)

        subs[i] = collect(actual_start:actual_stop)

        # Move on to the next block’s start
        startidx = stopidx + 1
    end
    println(subs)
    return subs
end

# 5) Inf step on subspace D_i
function inf_step(u_current::Vector{Float64},
                  K::Matrix{Float64}, M::Matrix{Float64},
                  idx_sub::Vector{Int})
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
                     K::Matrix{Float64}, M::Matrix{Float64})
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
function M_norm_distance(u::Vector{Float64}, v::Vector{Float64}, M::Matrix{Float64})
    w = u .- v
    return sqrt(dot(w, M*w))
end

# 7) Main iteration: store solutions & keep sign consistency
function ddm_eigen_solver(;
    N::Int=50,
    m::Int=2,
    maxiter::Int=50,
    tol::Float64=1e-8,
    sweep::Bool=true
)

    K, M = laplace_eig_matrices(N)

    # Initial guess
    u_cur = ones(N)
    normalize_M!(u_cur, M)

    # Sub-domain index sets
    subspaces = subspace_indices(N, m)

    # Track the Rayleigh quotient each iteration
    lambda_history = Float64[]
    # Also store the approximate eigenvector after each iteration
    solutions = Vector{Vector{Float64}}()

    λ_cur = R(u_cur, K, M)
    push!(lambda_history, λ_cur)
    push!(solutions, copy(u_cur))

    println("Initial Rayleigh quotient = $λ_cur")

    sub_int = 1:m
    for n in 1:maxiter
        # Local updates
        local_updates = Vector{Vector{Float64}}(undef, m+1)
        local_updates[1] = u_cur
        for i in sub_int
            println(i)

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
        if sweep
            sub_int = -sub_int.+(m+1)
        end

        # Combine step
        u_new = combine_step(local_updates, K, M)

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

        println(abs(λ_new - λ_cur))
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
N       = 1000
m       = 9
maxiter = 100
tol     = 1e-10

u_approx, lambda_approx, lambda_history, solutions = ddm_eigen_solver(
    N=N,
    m=m,
    maxiter=maxiter,
    tol=tol
)

println("Final approximate eigenvalue = $lambda_approx")

###############################################################################
# Make the plots
###############################################################################
#  -- Plot 1: Final approximate eigenfunction --
x = range(0, 1, length = N+2)
u_plot = vcat(0.0, u_approx, 0.0)

plt1 = plot(
    x, u_plot,
    marker    = :o,
    xlabel    = "x",
    ylabel    = "u(x)",
    title     = "DDM Approx. Eigenfunction (λ ≈ $lambda_approx)",
    label     = "Final Eigenfunction"
)

#  -- Plot 2: Convergence of the Rayleigh quotient --
iters = 0:length(lambda_history)-1
plt2 = plot(
    iters, lambda_history,
    marker = :o,
    xlabel = "Iteration",
    ylabel = "Rayleigh Quotient",
    title  = "Convergence of Eigenvalue (m=$m, N=$N)"
)

#  -- Plot 3: Convergence in the eigenvector (M-norm) --
#  We measure the distance of each solution from the final solution
K, M = laplace_eig_matrices(N)
final_sol = solutions[end]
exact_sol = eigen(K)
exact_val = exact_sol.values[1]
exact_vec = exact_sol.vectors[:, 1]
distances = [
    M_norm_distance(solutions[i], exact_vec, M)
    for i in 1:length(solutions)
]
plt3 = plot(
    iters, distances,
    marker = :o,
    xlabel = "Iteration",
    ylabel = "||u^(k) - u^(exact)||_M",
    title  = "Convergence of the Eigenvector in M-norm",
    yaxis = :log
)

#  -- Plot 4: Overlay all iteration solutions --
plt4 = plot(title="All Iteration Solutions (m=$m, N=$N)")
for i in 1:length(solutions)
    u_iter_withBC = vcat(0.0, solutions[i], 0.0)
    plot!(x, u_iter_withBC,
    # label="Iter $(i-1)",
    legend=false,
    marker=:none)
end

# Display them in sequence
display(plt1)
display(plt2)
display(plt3)
display(plt4)

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
