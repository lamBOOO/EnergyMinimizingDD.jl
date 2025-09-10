import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.instantiate()

using Documenter
using DDEigen

makedocs(
    sitename = "DDEigen",
    modules = [DDEigen],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)
