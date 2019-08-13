using Documenter
using CalibrationErrors
using Literate

# define directories
const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const OUTPUT = joinpath(@__DIR__, "src", "generated")

# recreate output directory
rm(OUTPUT; force=true, recursive=true)
mkpath(OUTPUT)

# generate Markdown and Jupyter notebook
for file in readdir(EXAMPLES)
    endswith(file, ".jl") || continue

    fullpath = joinpath(EXAMPLES, file)

    Literate.markdown(fullpath, OUTPUT)
    Literate.notebook(fullpath, OUTPUT)
end

makedocs(
sitename = "CalibrationErrors.jl",
    pages = [
        "Home" => "index.md",
        "Background" => "background.md",
        "Calibration" => "calibration.md",
        "Estimators" => "estimators.md",
        "Examples" => [
            "Distribution" => "generated/distribution.md"
        ]
    ]
)
