function sqwasserstein(μ, ν, C, optimizer)
    P = optimal_transport_map(μ, ν, C, optimizer)
    return LinearAlgebra.dot(P, C)
end

"""
    optimal_transport_map(μ, ν, C, optimizer)

Solve the discrete optimal transport problem with source `μ`, target `ν`, and
cost matrix `C` as a linear programming (LP) problem with the given `optimizer`.

More concretely, this function returns a solution `P` of the LP problem
```math
\\begin{aligned}
\\min_{p} c^T p & \\\\
\\text{subject to } A_1p &= μ \\\\
A_2p &= ν \\\\
0 &\\leq p
\\end{aligned}
```
where
```math
\\begin{aligned}
p &= [P_{1,1},P_{2,1},\\ldots,P_{n,1},P_{2,1},\\ldots,P_{n,m}]^T, \\\\
c &= [C_{1,1},C_{2,1},\\ldots,C_{n,1},C_{2,1},\\ldots,C_{n,m}]^T, \\\\
A_1 &= \\begin{bmatrix}
1_n^T \\otimes I_m 
\\end{bmatrix}, \\\\
A_2 &= \\begin{bmatrix}
I_n \\otimes 1_m^T
\\end{bmatrix}.
\\end{aligned}
```

A possible choice of `optimizer` is `Tulip.Optimizer()` in the `Tulip` package.
"""
function optimal_transport_map(μ, ν, C, model::MOI.ModelLike)
    nμ = length(μ)
    nν = length(ν)
    size(C) == (nμ, nν) || error("size of `C` does not match size of `μ` and `ν`")
    nC = length(C)

    # define variables
    x = MOI.add_variables(model, nC)
    xmat = reshape(x, nμ, nν)

    # define objective function
    T = eltype(C)
    zero_T = zero(T)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(vec(C), x), zero_T),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for xi in x
        MOI.add_constraint(model, MOI.SingleVariable(xi), MOI.GreaterThan(zero_T))
    end

    # add constraints for source
    for (xs, μi) in zip(eachrow(xmat), μ)
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(μi), xi) for xi in xs], zero(μi)
        )
        MOI.add_constraint(model, f, MOI.EqualTo(μi))
    end

    # add constraints for target
    for (xs, νi) in zip(eachcol(xmat), ν)
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(νi), xi) for xi in xs], zero(νi)
        )
        MOI.add_constraint(model, f, MOI.EqualTo(νi))
    end

    # compute optimal solution
    MOI.optimize!(model)
    p = MOI.get(model, MOI.VariablePrimal(), x)
    P = reshape(p, nμ, nν)

    return P
end
