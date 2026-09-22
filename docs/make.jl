import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.instantiate()

using Documenter
using EnergyMinimizingDD

makedocs(
  sitename="EnergyMinimizingDD",
  modules=[EnergyMinimizingDD],
  repo="github.com/lamBOOO/EnergyMinimizingDD.jl/blob/{commit}{path}#{line}",
  checkdocs=:none,
  format=Documenter.HTML(),
  pages=["Home" => "index.md"],
)

deploydocs(repo="github.com/lamBOOO/EnergyMinimizingDD.jl.git")
