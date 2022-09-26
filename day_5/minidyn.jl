using MacroTools
using LabelledArrays
using StaticArrays
using ForwardDiff

struct PModel{states, controls, shocks, parameters, n_s, n_x, n_e, n_p}
    s::SLArray{Tuple{n_s}, Float64, 1, n_s, states}
    x::SLArray{Tuple{n_x}, Float64, 1, n_x, controls}
    e::SLArray{Tuple{n_e}, Float64, 1, n_e, shocks}
    p::SLArray{Tuple{n_p}, Float64, 1, n_p, parameters}
end

get_states(model::PModel{A,B,C,D,E,F,G,H}) where A where B where C where D where E where F where G where H = A
get_controls(model::PModel{A,B,C,D,E,F,G,H}) where A where B where C where D where E where F where G where H = B
get_shocks(model::PModel{A,B,C,D,E,F,G,H}) where A where B where C where D where E where F where G where H = C
get_parameters(model::PModel{A,B,C,D,E,F,G,H}) where A where B where C where D where E where F where G where H = D


function transform_variables(ex,g,group)
    if @capture ex x_[t]
        ind = 2
    elseif @capture ex x_[t+i_]
        ind = 2+i
    elseif @capture ex x_[t-i_]
        ind = 2-i
    else
        return ex
    end
    if x in group
        return :($g[$ind].$x)
    else
        return ex
    end
end

function dynexpr(mod, expr)
    # mod = eval(m)
    states = get_states(mod)
    controls = get_controls(mod)
    expr = MacroTools.prewalk(
            u->transform_variables(u, :s, states),
            expr
        )
    expr = MacroTools.prewalk(
        u->transform_variables(u, :x, controls),
        expr
    )
    return expr
end

macro dyneq(expr)
    modelname = expr.args[1].args[2]
    funtype = expr.args[1].args[1]
    body = expr.args[2]
    mod = eval(modelname)
    new_body = dynexpr(mod, body)
    syms_params = symbols(mod.p)

    if funtype==:transition
        tranfun = :(
            function transition(model, s::Tuple, x::Tuple, E::Tuple)

                $( Expr(:(=), Expr(:tuple, syms_params...), :(model.p)) )
                
                # S = @LArray [0.0, 0.0] (:k,:z)
                $new_body
                # SLVector(S)
            end
        )
    else
        
        function rewrite_equations(expr, i)
            if @capture expr lhs_ == rhs_
                new_expr = :(out[$i] = ($rhs)-($lhs) )
                return new_expr, i+1
            else
                return (expr, i)
            end

        end

        i=1
        for (k, a) in enumerate(new_body.args)
            ex, i = rewrite_equations(a, i)
            new_body.args[k] = ex
        end
        n_x = length(get_controls(model))

        @assert (i-1) == n_x

        tranfun = :(

            function arbitrage(model, s::Tuple, x::Tuple, out)

                $( Expr(:(=), Expr(:tuple, syms_params...), :(model.p)) )

                $new_body
                
                SVector(out)

            end
        )
    end

    return tranfun
end


function transition(model, s::SLArray, x::SLArray, E::SLArray)
    ## begin: hack
    vt = (s[1] + x[1] + E[1])
    v = SVector( (vt*0 for i=1:length(s))...)
    S = MVectorLike(s, v)
    ## end: hack
    transition(model, (s, S, nothing), (x, nothing, nothing), (nothing, E, nothing))
    SLVector(S)
end

function LVectorLike(m0::SLArray{Tuple{d}, T, 1, d, M}, m) where d where T where M
    tt = eltype(m)
    TT = SLArray{Tuple{d}, tt, 1, d, M}
    return TT(m...)
end
function MVectorLike(m0::SLArray{Tuple{d}, T, 1, d, M}, m) where d where T where M
    v = LVectorLike(m0, m)
    vv = LVector(v)
end

function transition(model, s::SVector, x::SVector, E::SVector)
    s_ = LVectorLike(model.s, s)
    x_ = LVectorLike(model.x, x)
    E_ = LVectorLike(model.e, E)
    S_ = transition(model, s_, x_, E_)
    SVector(S_...)
end


function arbitrage(model, s::SLArray, x::SLArray, S::SLArray, X::SLArray)
    ## begin: hack
    vt = (s[1] + x[1]  + S[1] + X[1])
    out = MVector( (vt*0 for i=1:length(x))...)
    # return out
    arbitrage(model, (nothing, s, S), (nothing, x, X), out)
    return SVector(out...)
end

function arbitrage(model, s::SVector, x::SVector,  S::SVector, X::SVector)
    s_ = LVectorLike(model.s, s)
    x_ = LVectorLike(model.x, x)
    S_ = LVectorLike(model.s, S)
    X_ = LVectorLike(model.x, X)
    arbitrage(model, s_, x_, S_, X_)
end


function perturb(model::PModel)

    s_ = SVector(model.s)
    x_ = SVector(model.x)
    e_ = SVector(model.e)

    A = ForwardDiff.jacobian(u->arbitrage(model, u, x_, s_, x_), s_)
    B = ForwardDiff.jacobian(u->arbitrage(model, s_, u, s_, x_), x_)
    C = ForwardDiff.jacobian(u->arbitrage(model, s_, x_, u, x_), s_)
    D = ForwardDiff.jacobian(u->arbitrage(model, s_, x_, s_, u), x_)
    
    E = ForwardDiff.jacobian(u->transition(model, u, x_, e_), s_)
    F = ForwardDiff.jacobian(u->transition(model, s_, u, e_), x_)
    Q = ForwardDiff.jacobian(u->transition(model, s_, x_, u), e_)
    

    (;A,B,C,D,E,F,Q)

end


function steady_state(model)

    (;s,x,e) = model
    f = arbitrage(model,s,x,s,x)
    g = transition(model,s,x,e) - s

    @assert maximum(abs,f)<1e-10
    @assert maximum(abs,g)<1e-10
    return (;s, x)

end
