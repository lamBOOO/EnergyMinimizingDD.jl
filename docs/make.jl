import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.instantiate()

using Documenter
using VariationalDD

makedocs(
    sitename = "VariationalDD",
    modules = [VariationalDD],
    checkdocs = :none,
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)
