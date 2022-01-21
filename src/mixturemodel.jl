function CalibrationErrors.unsafe_skce_eval_targets(
    κtargets::Kernel, p::AbstractMixtureModel, y, p̃::AbstractMixtureModel, ỹ
)
    p_components = components(p)
    p_probs = probs(p)
    p̃_components = components(p̃)
    p̃_probs = probs(p̃)

    probsi = p_probs[1]
    componentsi = p_components[1]
    s =
        probsi *
        p̃_probs[1] *
        CalibrationErrors.unsafe_skce_eval_targets(
            κtargets, componentsi, y, p̃_components[1], ỹ
        )
    for j in 2:length(p̃_components)
        s +=
            probsi *
            p̃_probs[j] *
            CalibrationErrors.unsafe_skce_eval_targets(
                κtargets, componentsi, y, p̃_components[j], ỹ
            )
    end

    for i in 2:length(p_components)
        probsi = p_probs[i]
        componentsi = p_components[i]

        for j in 2:length(p̃_components)
            s +=
                probsi *
                p̃_probs[j] *
                CalibrationErrors.unsafe_skce_eval_targets(
                    κtargets, componentsi, y, p̃_components[j], ỹ
                )
        end
    end

    return s
end
