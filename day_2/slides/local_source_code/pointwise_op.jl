using LinearAlgebra         # import LinearAlgebra library

f(x) = 2x                   # simple function definition
f(x) = norm(x)              # norm defined in LinearAlgebra
g(x) = sum(x + x.^2)        # dot for pointwise operations
α, β = 2.0, -2.0            # unicode symbols

q(x) = sin(cos(x))          # another function

x = rand(5)
println(q(5))                 # OK
println(q.(x))                # OK
println(q(x))                 # Error! 

