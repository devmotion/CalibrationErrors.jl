using CalibrationErrors
using Documenter

makedocs(;
    modules = [CalibrationErrors],
    authors = "David Widmann <david.widmann@it.uu.se>",
    repo = "https://github.com/devmotion/CalibrationErrors.jl/blob/{commit}{path}#L{line}",
    sitename = "CalibrationErrors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/CalibrationErrors.jl",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/devmotion/CalibrationErrors.jl.git",
    push_preview = true,
)
