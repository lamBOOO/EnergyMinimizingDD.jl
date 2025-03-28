using LinearAlgebra
using Plots

# 1) Define the 1D mesh and finite difference matrices
function laplace_eig_matrices(N::Int)
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

    # Mass matrix: the identity times 1.0 (per your request)
    M = Matrix(I, N, N) .* 1.0

    return K, M
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
function subspace_indices(N::Int, m::Int)
    size_block = div(N, m)
    subs = Vector{Vector{Int}}(undef, m)
    startidx = 1
    for i in 1:m
        stopidx = (i < m) ? (startidx + size_block - 1) : N
        subs[i] = collect(startidx:stopidx)
        startidx = stopidx + 1
    end
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
    K_local = B' * (K * B)
    M_local = B' * (M * B)
    eigvals, eigvecs = eigen(K_local, M_local)
    i_min = argmin(eigvals)
    α_min = eigvecs[:, i_min]

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
    tol::Float64=1e-8
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

    for n in 1:maxiter
        # Local updates
        local_updates = Vector{Vector{Float64}}(undef, m+1)
        local_updates[1] = u_cur
        for i in 1:m
            u_next_i = inf_step(local_updates[i], K, M, subspaces[i])
            local_updates[i+1] = u_next_i
        end

        # Combine step
        u_new = combine_step(local_updates, K, M)

        # --- SIGN FIX to avoid solution flipping from iteration to iteration ---
        if dot(u_new, u_cur) < 0
            u_new .*= -1.0
        end

        λ_new = R(u_new, K, M)
        push!(lambda_history, λ_new)
        push!(solutions, copy(u_new))

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
N       = 30
m       = 3
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
