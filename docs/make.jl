using Documenter
import JSON

if haskey(ENV, "GITHUB_ACTIONS")
    # Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
    ENV["JULIA_DEBUG"] = "Documenter"
    # Bypass the accept download prompt
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
end

using Literate, CalibrationErrors

EXAMPLES = joinpath(@__DIR__, "..", "examples")
OUTPUT = joinpath(@__DIR__, "src", "examples")

ispath(OUTPUT) && rm(OUTPUT; recursive=true)

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

for file in readdir(EXAMPLES; join=true)
    endswith(file, ".jl") || continue
    Literate.markdown(file, OUTPUT; documenter=true, preprocess=preprocess)
    Literate.notebook(file, OUTPUT)
end

# Avoid font caching warning in docs
using CairoMakie
CairoMakie.activate!()
scatter(rand(10), rand(10))

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
        "index.md",
        "introduction.md",
        "ece.md",
        "kce.md",
        "Examples" => joinpath.(
            "examples", filter(x -> endswith(x, ".md"), readdir(OUTPUT)),
        ),
    ],
)

# Obtain PR from commit message
function pr_from_commit()
    # Check if action is caused by bors on the "trying" branch
    get(ENV, "GITHUB_ACTOR", "") == "bors[bot]" || return
    get(ENV, "GITHUB_REF", "") == "refs/heads/trying" || return

    # Parse event payload
    event_path = get(ENV, "GITHUB_EVENT_PATH", nothing)
    event_path === nothing && return
    event = JSON.parsefile(event_path)
    if haskey(event, "head_commit") && haskey(event["head_commit"], "message")
        m = match(r"^Try #(.*):$", event["head_commit"]["message"])
        m === nothing && return
        return m.captures[1]
    end

    return
end

# Enable preview on the trying branch
function deploy_config()
    if haskey(ENV, "GITHUB_ACTIONS")
        repo = get(ENV, "GITHUB_REPOSITORY", "")
        if repo == "devmotion/CalibrationErrors.jl" && (pr = pr_from_commit()) !== nothing
            return Documenter.GitHubActions(
                "devmotion/CalibrationErrors.jl",
                "pull_request",
                "refs/pull/$pr/merge",
            )
        end
    end

    return Documenter.auto_detect_deploy_system()
end

deploydocs(;
    repo = "github.com/devmotion/CalibrationErrors.jl.git",
    deploy_config = deploy_config(),
    push_preview = true,
)
