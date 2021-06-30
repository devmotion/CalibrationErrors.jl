# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# Run examples asynchronously
const EXAMPLES_SRC = joinpath(@__DIR__, "..", "examples")
const LITERATEJL = joinpath(@__DIR__, "literate.jl")
processes = map(filter!(isdir, readdir(EXAMPLES_SRC; join=true))) do example
    scriptjl = joinpath(example, "script.jl")
    return run(
        pipeline(
            `$(Base.julia_cmd()) $LITERATEJL $scriptjl $EXAMPLES_OUT`;
            stdin=devnull,
            stdout=devnull,
            stderr=stderr,
        );
        wait=false,
    )::Base.Process
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")

# Build documentation
using CalibrationErrors
using Documenter

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
            map(filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT))) do x
                return joinpath("examples", x)
            end,
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/devmotion/CalibrationErrors.jl.git",
    push_preview=true,
    devbranch="main",
)
