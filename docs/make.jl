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

    event_path = get(ENV, "GITHUB_EVENT_PATH", nothing)
    event_path === nothing && return
    event = JSON.parsefile(event_path)
    @show event
    haskey(event, "push") || return
    @show event["push"]
    haskey(event["push"], "commits") || return
    @show event["push"]["commits"]
    haskey(event["push"]["commits"][1], "message") || return
    @show event["push"]["commits"][1]["message"]

    m = match(r"^Try (.*):$", event["push"]["commits"][1]["message"])
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
if config === nothing
    # fallback
    deploydocs(;
        repo = "github.com/devmotion/CalibrationErrors.jl.git",
    )
else
    deploydocs(
        config;
        repo = "github.com/devmotion/CalibrationErrors.jl.git",
        push_preview = true,
    )
end
