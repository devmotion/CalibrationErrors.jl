using CalibrationErrors
using Literate: Literate
using Pkg: Pkg

if haskey(ENV, "GITHUB_ACTIONS")
    # Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
    ENV["JULIA_DEBUG"] = "Documenter"
    # Bypass the accept download prompt
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
end

EXAMPLES = joinpath(@__DIR__, "..", "examples")
OUTPUT = joinpath(@__DIR__, "src", "examples")

ispath(OUTPUT) && rm(OUTPUT; recursive=true)
mkpath(joinpath(OUTPUT, "figures"))

# Add links to binder and nbviewer below the first heading of level 1
function preprocess(content)
    sub = s"""
        \0
        #
        # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/examples/@__NAME__.ipynb)
        # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)
    """
    return replace(content, r"^# # [^\n]*"m => sub; count=1)
end

# Save current project directory
PROJECT_PATH = Pkg.project().path

for file in readdir(EXAMPLES; join=true)
    endswith(file, ".jl") || continue
    Literate.markdown(file, OUTPUT; documenter=true, preprocess=preprocess)
    Literate.notebook(file, OUTPUT)
end

# Reset project directory
Pkg.activate(PROJECT_PATH)

using Documenter
using JSON: JSON

makedocs(;
    modules=[CalibrationErrors],
    authors="David Widmann <david.widmann@it.uu.se>",
    repo="https://github.com/devmotion/CalibrationErrors.jl/blob/{commit}{path}#L{line}",
    sitename="CalibrationErrors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/CalibrationErrors.jl",
        assets=String[],
    ),
    pages=[
        "index.md",
        "introduction.md",
        "ece.md",
        "kce.md",
        "others.md",
        "Examples" =>
            joinpath.("examples", filter(x -> endswith(x, ".md"), readdir(OUTPUT))),
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/devmotion/CalibrationErrors.jl.git", push_preview=true)
