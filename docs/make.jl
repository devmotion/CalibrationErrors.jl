using Documenter
import JSON

if haskey(ENV, "GITHUB_ACTIONS")
    # Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
    ENV["JULIA_DEBUG"] = "Documenter"
    # Bypass the accept download prompt
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
end

using CalibrationErrors

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
    ],
)

# custom configuration if on the trying branch to enable previews
@show ENV
function deployconfig()
    get(ENV, "GITHUB_REPOSITORY", "") == "devmotion/CalibrationErrors.jl" || return
    get(ENV, "GITHUB_REF", "") == "refs/heads/trying" || return
    get(ENV, "GITHUB_ACTOR", "") == "bors[bot]" || return

    event_path = get(ENV, "GITHUB_EVENT_PATH", nothing)
    event_path === nothing && return
    event = JSON.parsefile(event_path)
    haskey(event, "head_commit") || return
    @show event["head_commit"]
    haskey(event["head_commit"], "message") || return
    @show event["head_commit"]["message"]

    m = match(r"^Try #(.*):$", event["head_commit"]["message"])
    m === nothing && return

    pr = m.captures[1]
    config = Documenter.GitHubActions(
        "devmotion/CalibrationErrors.jl",
        "pull_request",
        "refs/pull/$pr/merge",
    )
    return config
end

config = deployconfig()
deploydocs(;
    repo = "github.com/devmotion/CalibrationErrors.jl.git",
    deployconfig = config === nothing ? Documenter.auto_detect_deploy_system() : config,
    push_preview = true,
)
