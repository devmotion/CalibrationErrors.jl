### Process examples
# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
examples = filter!(isdir, readdir(joinpath(@__DIR__, "..", "examples"); join=true))
let script = "using Pkg; Pkg.activate(ARGS[1]); Pkg.instantiate()"
    for example in examples
        if !success(`$(Base.julia_cmd()) -e $script $example`)
            error(
                "project environment of example ",
                basename(example),
                " could not be instantiated",
            )
        end
    end
end
# Run examples asynchronously
processes = let literatejl = joinpath(@__DIR__, "literate.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $literatejl $(basename(example)) $EXAMPLES_OUT`;
                stdin=devnull,
                stdout=devnull,
                stderr=stderr,
            );
            wait=false,
        )::Base.Process
    end
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
