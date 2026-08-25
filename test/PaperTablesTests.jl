using Test

isdefined(Main, :check_generated_tables) ||
  include(joinpath(@__DIR__, "..", "examples", "paper", "tables_all.jl"))

@testset "generated paper table check" begin
  methods = (
    "var_dd_additive",
    "var_dd_additive_history",
    "pcg_as",
    "gmres_ras",
  )
  mesh_sizes = (20, 40, 60, 80, 100, 120)
  overlap_layers = (1, 2, 4, 8)
  rows = NamedTuple[]

  for (method_index, method) in enumerate(methods)
    for N in mesh_sizes
      push!(rows, (
        experiment="mesh",
        regime="fixed_layers",
        method,
        N,
        overlap=0,
        local_batches=N + method_index,
        relative_residual=1e-11,
      ))
      push!(rows, (
        experiment="mesh",
        regime="fixed_delta_over_H",
        method,
        N,
        overlap=0,
        local_batches=N + 2 * method_index,
        relative_residual=1e-11,
      ))
    end
    for overlap in overlap_layers
      push!(rows, (
        experiment="overlap",
        regime="layer_sweep",
        method,
        N=64,
        overlap,
        local_batches=10 * overlap + method_index,
        relative_residual=1e-11,
      ))
    end
  end

  mktempdir() do temporary_dir
    data_file = joinpath(temporary_dir, "study8_sensitivity.csv")
    committed_dir = joinpath(temporary_dir, "committed")
    CSV.write(data_file, rows)
    make_fig18_table(; output_dir=committed_dir, data_file)

    @test isnothing(check_generated_tables(; data_file, committed_dir))

    open(joinpath(committed_dir, "fig18_poisson_scaling.csv"), "a") do io
      write(io, "stale\n")
    end
    @test_throws ErrorException check_generated_tables(; data_file, committed_dir)
  end
end
