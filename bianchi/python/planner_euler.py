"""
Overborrowing, planner's problem, Euler approach.

Note: not currently converging although it used to under slightly different
initial conditions.

"""

from decentralized import *

# Create model and unpack 
model = create_overborrowing_model()
σ, η, β, ω, κ, R, b_grid, yN, yT, P = model
n_B, n_y =  len(b_grid), len(yN)
b_grid_min, b_grid_max = b_grid[0], b_grid[-1]

# Reshape
b = np.reshape(b_grid, (n_B, 1))
YT = np.reshape(yT, (1, n_y))
YN = np.reshape(yN, (1, n_y))

# Solve decentralized to get initial conditions
c_de, bp_de = solve_decentralized_equilibrium(model)
c_bind_de = compute_consumption_at_constraint(c_de, model)

# Calculate utility differential when consumption is constrained
exp_mu_de = compute_exp_marginal_utility(c_de, bp_de, model)
mu_constrained_de = compute_marginal_utility(c_bind_de, model)
constrained_euler_diff_de = mu_constrained_de - exp_mu_de

# Get binding indices
idx_bind_de = compute_binding_indicies(exp_mu_de, c_bind_de, model)

euler_diff_de = np.where(idx_bind_de, constrained_euler_diff_de, 0.0)
lagrange_de = euler_diff_de

def psi(c):
    return ((1 - ω) / ω) * (η + 1) * κ * (c / YN)**η

# Set some initial conditions
lagrange_sp = lagrange_de / (1 - psi(c_de))  
idx_bind_sp = idx_bind_de
c_bind = c_bind_de   # Added by JS

c = np.copy(c_de)
bp = np.copy(bp_de)

α = 0.02
tol = 1e-5
euler_tol = 1e-7
error = tol + 1
current_iter = 0
iter_max = 100_000
print_step = 10


while error > tol and current_iter < iter_max:
 
    old_c = np.copy(c)
    old_bp = np.copy(bp)

    mup = compute_marginal_utility(c, model)

    lfh_sp = mup + lagrange_sp * psi(c)

    rhs_sp = compute_exp_marginal_utility(lfh_sp, bp, model)

    # This is equation u_T +mu*psi= β*R E [(u_T(t+1)+mu_t+1 ]+mu_t
    lagrange_sp = lfh_sp - β * R * rhs_sp 
    lagrange_sp[idx_bind_sp == 0] = 0

    for i in range(n_B):
        for j in range(n_y):
       
            if lagrange_sp[i, j] >= euler_tol:   # Borrowing constraint binds
                c[i, j] = c_bind[i, j]
                idx_bind_sp[i, j] = 1

            else:   # Use the Euler equation   
                def euler_diff(cc):
                    return (ω*cc**(-η) + (1-ω)*yN[j]**(-η))**(σ/η-1/η-1) \
                            * ω * cc**(-η-1) - rhs_sp[i, j]
                c0 = c[i, j]
                c[i, j] = root(euler_diff, c0).x
                idx_bind_sp[i, j] = 0
           
    bp = R * b + YT - c
    bp[bp > b_grid_max] = b_grid_max
    bp[bp < b_grid_min] = b_grid_min
 
    price = ((1 - ω) / ω) * (c / YN)**(1+η)

    # Check collateral constraint
    c = R * b + YT - np.maximum(bp, -κ * (price * YN + YT))
    price = ((1 - ω) / ω) * (c / YN)**(1+η)
    bp = R * b + YT - c

    error = d_infty(c, old_c)

    # Add smoothing
    bp = α*bp + (1-α)*old_bp
    c = α*c + (1-α)*old_c

    # Compute c_bind
    bmax_collat = -κ * (price * YN + YT)
    c_bind = R * b + YT - bmax_collat

    # Compute error and update iteration
    current_iter += 1
    if current_iter % print_step == 0:
        print(f"Current error = {error} at iteration {current_iter}.")


# Here's my current output.  You can see it doesn't quite correspond to Fig 1 in the paper.
#
# FO: I think we are taking different paths for decentralised so that is fine - I can't get the planner to converge though.

fig, ax = plt.subplots()
y_point = 0
ax.plot(b_grid, b_grid, 'k--')
ax.plot(b_grid, bp_de[:, y_point], label='decentralized')
ax.plot(b_grid, bp[:, y_point], label='planner')
ax.legend()
plt.show()





