using Documenter
using CalibrationErrors
using Literate

# define directories
const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const OUTPUT = joinpath(@__DIR__, "src", "generated")

# recreate output directory
rm(OUTPUT; force=true, recursive=true)
mkpath(OUTPUT)

# fix align environments inside of equation environments in notebooks
# by using $$...$$ instead of \begin{equation}...\end{equation}
replace_math(content) = replace(content, r"```math(.*?)```"s => s"$$\1$$")

# generate Markdown and Jupyter notebook
for file in readdir(EXAMPLES)
    endswith(file, ".jl") || continue

    fullpath = joinpath(EXAMPLES, file)

    Literate.markdown(fullpath, OUTPUT)
    Literate.notebook(fullpath, OUTPUT; preprocess = replace_math)
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

deploydocs(
    repo = "github.com/devmotion/CalibrationErrors.jl.git",
)