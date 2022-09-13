"""
Re-implements the investment problem using JAX.

We test

1. VFI
2. VFI with Anderson acceleration
3. HPI
4. OPI 

"""
import numpy as np
import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple

# Use 64 bit floats with JAX in order to match NumPy/Numba code
jax.config.update("jax_enable_x64", True)


def successive_approx(T,                     # Operator (callable)
                      x_0,                   # Initial condition
                      tolerance=1e-6,        # Error tolerance
                      max_iter=10_000,       # Max iteration bound
                      print_step=25,         # Print at multiples
                      verbose=False):        
    x = x_0
    error = tolerance + 1
    k = 1
    while error > tolerance and k <= max_iter:
        x_new = T(x)
        error = np.max(np.abs(x_new - x))
        if verbose and k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        x = x_new
        k += 1
    if error > tolerance:
        print(f"Warning: Iteration hit upper bound {max_iter}.")
    elif verbose:
        print(f"Terminated successfully in {k} iterations.")
    return x


# == Primitives and Operators == #

# A namedtuple definition for storing parameters and grids
Model = namedtuple("Model", 
                   ("β", "a_0", "a_1", "γ", "c",
                    "y_size", "z_size", "y_grid", "z_grid", "Q"))

def create_investment_model(
        r=0.01,                              # Interest rate
        a_0=10.0, a_1=1.0,                   # Demand parameters
        γ=25.0, c=1.0,                       # Adjustment and unit cost 
        y_min=0.0, y_max=20.0, y_size=100,   # Grid for output
        ρ=0.9, ν=1.0,                        # AR(1) parameters
        z_size=150):                         # Grid size for shock
    """
    A function that takes in parameters and returns an instance of Model that
    contains data for the investment problem.
    """
    β = 1/(1+r) 
    y_grid = np.linspace(y_min, y_max, y_size)  
    mc = qe.tauchen(ρ, ν, n=z_size)
    z_grid, Q = mc.state_values, mc.P

    model = Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
                  y_size=y_size, z_size=z_size,
                  y_grid=y_grid, z_grid=z_grid, Q=Q)
    return model


def create_investment_model_jax():
    "Build a JAX-compatible version of the investment model."

    model = create_investment_model()
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model

    # Break up parameters into static and nonstatic components
    constants = β, a_0, a_1, γ, c
    sizes = y_size, z_size
    arrays = y_grid, z_grid, Q

    # Shift arrays to the device (e.g., GPU)
    arrays = tuple(map(jax.device_put, arrays))
    return constants, sizes, arrays


def B(v, constants, sizes, arrays):
    """
    A vectorized version of the right-hand side of the Bellman equation 
    (before maximization), which is a 3D array representing

        B(y, z, y′) = r(y, z, y′) + β Σ_z′ v(y′, z′) Q(z, z′)."

    for all (y, z, y′).
    """

    # Unpack 
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    # Compute current rewards r(y, z, yp) as array r[i, j, ip]
    y  = jnp.reshape(y_grid, (y_size, 1, 1))    # y[i]   ->  y[i, j, ip]
    z  = jnp.reshape(z_grid, (1, z_size, 1))    # z[j]   ->  z[i, j, ip]
    yp = jnp.reshape(y_grid, (1, 1, y_size))    # yp[ip] -> yp[i, j, ip]
    r = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    # Calculate continuation rewards at all combinations of (y, z, yp)
    v = jnp.reshape(v, (1, 1, y_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = jnp.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return r + β * EV


def compute_r_σ(σ, constants, sizes, arrays):
    """
    Compute the array r_σ[i, j] = r[i, j, σ[i, j]], which gives current
    rewards given policy σ.
    """

    # Unpack model
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    # Compute r_σ[i, j]
    y = jnp.reshape(y_grid, (y_size, 1))
    z = jnp.reshape(z_grid, (1, z_size))
    yp = y_grid[σ]
    r_σ = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    return r_σ


def T(v, constants, sizes, arrays):
    "The Bellman operator."
    return jnp.max(B(v, constants, sizes, arrays), axis=2)


def get_greedy(v, constants, sizes, arrays):
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, constants, sizes, arrays), axis=2)


def T_σ(v, σ, constants, sizes, arrays):
    "The σ-policy operator."

    # Unpack model
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    r_σ = compute_r_σ(σ, constants, sizes, arrays)

    # Compute the array v[σ[i, j], jp]
    zp_idx = jnp.arange(z_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, z_size))
    σ = jnp.reshape(σ, (y_size, z_size, 1))
    V = v[σ, zp_idx]      

    # Convert Q[j, jp] to Q[i, j, jp] 
    Q = jnp.reshape(Q, (1, z_size, z_size))

    # Calculate the expected sum Σ_jp v[σ[i, j], jp] * Q[i, j, jp]
    Ev = np.sum(V * Q, axis=2)

    return r_σ + β * np.sum(V * Q, axis=2)


def R_σ(v, σ, constants, sizes, arrays):
    """
    The value v_σ of a policy σ is defined as 

        v_σ = (I - β P_σ)^{-1} r_σ

    Here we set up the linear map v -> R_σ v, where R_σ := I - β P_σ. 

    In the investment problem, this map can be expressed as

        (R_σ v)(y, z) = v(y, z) - β Σ_z′ v(σ(y, z), z′) Q(z, z′)

    Defining the map as above works in a more intuitive multi-index setting
    (e.g. working with v[i, j] rather than flattening v to a one-dimensional
    array) and avoids instantiating the large matrix P_σ.

    """

    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    # Set up the array v[σ[i, j], jp]
    zp_idx = jnp.arange(z_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, z_size))
    σ = jnp.reshape(σ, (y_size, z_size, 1))
    V = v[σ, zp_idx]

    # Expand Q[j, jp] to Q[i, j, jp]
    Q = jnp.reshape(Q, (1, z_size, z_size))

    # Compute and return v[i, j] - β Σ_jp v[σ[i, j], jp] * Q[j, jp]
    return v - β * np.sum(V * Q, axis=2)


def get_value(σ, constants, sizes, arrays):
    "Get the value v_σ of policy σ by inverting the linear map R_σ."

    # Unpack 
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    r_σ = compute_r_σ(σ, constants, sizes, arrays)

    # Reduce R_σ to a function in v
    partial_R_σ = lambda v: R_σ(v, σ, constants, sizes, arrays)

    return jax.scipy.sparse.linalg.bicgstab(partial_R_σ, r_σ)[0]


# == Matrix versions == #


def compute_P_σ(σ, constants, sizes, arrays):
    """
    Compute the transition probabilities across states as a multi-index array

        P_σ[i, j, ip, jp] = (σ[i, j] == ip) * Q[j, jp]

    """

    # Unpack model
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    yp_idx = jnp.arange(y_size)
    yp_idx = jnp.reshape(yp_idx, (1, 1, y_size, 1))
    σ = jnp.reshape(σ, (y_size, z_size, 1, 1))
    A = jnp.where(σ == yp_idx, 1, 0)
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))
    P_σ = A * Q
    return P_σ


def get_value_matrix_version(σ, constants, sizes, arrays):
    """
    Get the value v_σ of policy σ via

        v_σ = (I - β P_σ)^{-1} r_σ

    In this version we flatten the multi-index [i, j] for the state (y, z) to
    a single index m and compute the vector r_σ[m] and matrix P_σ[m, mp]
    giving transition probabilities across the single-index state.  Then we
    solve the above equation using matrix inversion.

    """

    # Unpack 
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    # Obtain ordinary (multi-index) versions of r_σ and P_σ 
    r_σ = compute_r_σ(σ, constants, sizes, arrays)
    P_σ = compute_P_σ(σ, constants, sizes, arrays)

    # Reshape r_σ and P_σ for a single index state
    n = y_size * z_size
    P_σ = jnp.reshape(P_σ, (n, n))
    r_σ = jnp.reshape(r_σ, n)

    # Solve
    v_σ = jnp.linalg.solve(np.identity(n) - β * P_σ, r_σ)

    # Return as multi-index array
    return jnp.reshape(v_σ, (y_size, z_size))


def T_σ_matrix_version(v, σ, constants, sizes, arrays):
    "The σ-policy operator, single index version."

    # Unpack model
    β, a_0, a_1, γ, c = constants
    y_size, z_size = sizes
    y_grid, z_grid, Q = arrays

    # Obtain ordinary (multi-index) versions of r_σ and P_σ 
    r_σ = compute_r_σ(σ, constants, sizes, arrays)
    P_σ = compute_P_σ(σ, constants, sizes, arrays)

    # Reshape r_σ and P_σ for a single index state
    n = y_size * z_size
    P_σ = jnp.reshape(P_σ, (n, n))
    r_σ = jnp.reshape(r_σ, n)
    v = jnp.reshape(v, n)

    # Iterate with T_σ using matrix routines
    new_v = r_σ + β * P_σ @ v

    # Return as multi-index array
    return jnp.reshape(new_v, (y_size, z_size))


# == JIT compiled versions == #

B = jax.jit(B, static_argnums=(2,))
compute_r_σ = jax.jit(compute_r_σ, static_argnums=(2,))
T = jax.jit(T, static_argnums=(2,))
get_greedy = jax.jit(get_greedy, static_argnums=(2,))

get_value = jax.jit(get_value, static_argnums=(2,))

T_σ = jax.jit(T_σ, static_argnums=(3,))
R_σ = jax.jit(R_σ, static_argnums=(3,))

get_value_matrix_version = jax.jit(get_value_matrix_version, static_argnums=(2,))
T_σ_matrix_version = jax.jit(T_σ_matrix_version, static_argnums=(3,))

# == Solvers == #

def value_iteration(model, tol=1e-5):
    "Implements VFI."

    constants, sizes, arrays = model
    _T = lambda v: T(v, constants, sizes, arrays)
    vz = jnp.zeros(sizes)

    v_star = successive_approx(_T, vz, tolerance=tol)
    return get_greedy(v_star, constants, sizes, arrays)

def policy_iteration(model, matrix_version=False):
    "Howard policy iteration routine."

    constants, sizes, arrays = model
    if matrix_version:
        _get_value = get_value_matrix_version
    else:
        _get_value = get_value

    vz = jnp.zeros(sizes)
    σ = jnp.zeros(sizes, dtype=int)
    i, error = 0, 1.0
    while error > 0:
        v_σ = _get_value(σ, constants, sizes, arrays)
        σ_new = get_greedy(v_σ, constants, sizes, arrays)
        error = jnp.max(np.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error {error}.")
    return σ

def optimistic_policy_iteration(model, tol=1e-5, m=10, matrix_version=False):
    "Implements the OPI routine."
    constants, sizes, arrays = model
    if matrix_version:
        _T_σ = T_σ_matrix_version
    else:
        _T_σ = T_σ

    v = jnp.zeros(sizes)
    error = tol + 1
    while error > tol:
        last_v = v
        σ = get_greedy(v, constants, sizes, arrays)
        for _ in range(m):
            v = _T_σ(v, σ, constants, sizes, arrays)
        error = jnp.max(np.abs(v - last_v))
    return get_greedy(v, constants, sizes, arrays)


# == Tests == #

def quick_timing_test():
    model = create_investment_model_jax()
    print("Starting HPI.")
    qe.tic()
    out = policy_iteration(model)
    elapsed = qe.toc()
    print(out)
    print(f"HPI completed in {elapsed} seconds.")
    print("Starting VFI.")
    qe.tic()
    out = value_iteration(model)
    elapsed = qe.toc()
    print(out)
    print(f"VFI completed in {elapsed} seconds.")
    print("Starting OPI.")
    qe.tic()
    out = optimistic_policy_iteration(model, m=100)
    elapsed = qe.toc()
    print(out)
    print(f"OPI completed in {elapsed} seconds.")

