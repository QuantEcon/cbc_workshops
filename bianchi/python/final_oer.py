"""
States:
        z = (y, r) : the exogenous state

        d : the endogenous state


Bellman equation (value aggregator)

    B(d, z, dp, v) = u(c(d, z, dp)) + β Σ_zp v(dp, zp) Π(z, zp)

    y, r = z

    cT = y + dp/(1 + r) - d 

    c = (ω * cT ** (1 - 1/ξ) + (1 - ω) * (ħ**α)**(1 - 1/ξ))**(1/((1 - 1/ξ))) 
    
    u(c) = (c**(1 - σ) - 1)/(1 - σ) 

The grids are

    y, R, d, dp = y_grid[j], 1 + r_grid[j], d_grid[i],  d_grid[ip]

"""

import numpy as np
from collections import namedtuple
import quantecon as qe
from numba import njit, prange, int32
import jax
import jax.numpy as jnp
import sys
import tpm


# == Provide support functions == #

def successive_approx(T,                     # Operator (callable)
                      x_0,                   # Initial condition
                      tolerance=1e-5,        # Error tolerance
                      max_iter=10_000,       # Max iteration bound
                      print_step=25,         # Print at multiples
                      verbose=True):        
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


@njit
def argmax(list_object):
    "Return the index of the largest element of `list_object`."
    max_val = -np.inf
    argmax_index = None
    for i, x in enumerate(list_object):
        if x > max_val:
            max_val = x
            argmax_index = i
    return argmax_index



# == Define the OER model == #

Model = namedtuple(
    'Model', 
         ('σ'       # Inverse of intertemporal elasticity of consumption
          'ħ',      # H bar = Full endowment t units of labor 
          'ω',      # Share of tradables in consumption aggregator
          'ξ',      # Elasticity of subs., tradables vs nontradables
          'α',      # Labor share in notraded output
          'β',      # Annual subjective discount factor
          'π',      # Annual inflation target
          'ψ',      # Annual growth rate
          'γ',      # Parameter governing DNWR including inflation and growth
          'r_star', # Average annual real interest rate
          'Π',      # Transition probability matrix
          'Z',      # Array of exogenous states 
          'd_n',    # Number of points in debt grid
          'y_n',    # Number of points in tradable output grid
          'r_n',    # Number of points in foreign interest rate grid
          'd_grid', # Bond grid
          'y_grid', # Grid for tradable output
          'r_grid') # Grid for interest rate
)

def create_OER_model(σ=5.0, ħ=1.0, ω=0.19, ξ=0.43 ,α=0.75,
                     β=0.9571,
                     γ=0.96, 
                     π=0,
                     ψ=0,
                     d_min=-5, d_max=5.4, d_n=501,
                     y_n=21, 
                     r_n=11,
                     r_star=0.021048,
                     num_sim=1_000_000, 
                     burn=100_000, 
                     estimate=False):
    
    """
    Creates a parameterization with default values for the 
    Optimal Exchange Rate Model.
    """
    π = 1 + π
    γ = γ/((1 + π)*(1 + ψ))
    Θ = np.array([[0.72137370, -0.1323976], 
                  [0.0028990, 0.9705325]])
    Σ = np.array([[0.00116490, -0.0000131], 
                  [-0.0000131, 0.0001328]])
    N = np.array((y_n, r_n))
    
    size_error_message = 'Matrix in file does not have the same dimension as implied by inputs. You need to discretize the VAR again.'

    # Try to load the matrix in file
    try:
        data = np.load('markov_tran.npz')
        Π = data['Π']
        Z = data['Z']
        n_ck = data['N']
        n_ck = n_ck[0] * n_ck[1]
        # Check whether the stored matrix conforms to y_n, r_n
        assert y_n * r_n == n_ck, size_error_message

    # If necessary, discretize the VAR process and save
    except FileNotFoundError:
        print("Generating exogenous state Markov matrix.\n")
        Π, Xvec, Z = tpm.tpm(Θ, Σ , N, T=num_sim, Tburn=burn)
        # Make sure matrices are row major
        Π = np.ascontiguousarray(Π) 
        Z = np.ascontiguousarray(Z) 
        # Save data 
        np.savez("markov_tran.npz", Π=Π, Xvec=Xvec, Z=Z, N=N)
        print("\nDone.\n")

    y_grid = np.exp(Z[:,0]) 
    r_grid = np.exp(Z[:,1]) * (1 + r_star) - 1

    ny = len(y_grid)
    d_grid = np.linspace(d_min, d_max, d_n)
    
    return Model(σ=σ, ħ=ħ, ω=ω, ξ=ξ, α=α, β=β, γ=γ, π=π, ψ=ψ, r_star=r_star, 
                 d_n=d_n, y_n=y_n, r_n=r_n, 
                 Π=Π, Z=Z, d_grid=d_grid, y_grid=y_grid, r_grid=r_grid)
    

# == Numba version == #

@njit
def B(i, j, ip, v, model):
    """
    The right-hand side of the Bellman equation with candidate value v.

    """
    # Unpack
    d_grid, y_grid, r_grid = model.d_grid, model.y_grid, model.r_grid
    ω, ξ, ħ, α, β, σ = model.ω, model.ξ, model.ħ, model.α, model.β, model.σ
    r_star = model.r_star
    Π, Z =  model.Π, model.Z

    # Obtain values of exogenous and endogenous state
    z = Z[j, :]  
    y, r = np.exp(z[0]), np.exp(z[1]) * (1 + r_star) - 1
    d, dp = d_grid[i],  d_grid[ip]

    # Compute consumption and then evaluate
    cT = y + dp / (1 + r) - d # Consumption of tradable goods
    c = (ω * cT**(1 - 1/ξ) + (1 - ω) * (ħ**α)**(1 - 1/ξ))**(1/((1 - 1/ξ))) 
    if c > 0:
        return (c**(1 - σ) - 1)/ (1 - σ) + β * np.dot(v[ip, :], Π[j, :]) 
    return - np.inf


@njit(parallel=True)
def T(v, model):
    "The Bellman operator."
    d_size, z_size = len(model.d_grid), len(model.Z)
    v_new = np.empty_like(v)
    for i in prange(d_size):
        for j in range(z_size):
            v_new[i, j] = max([B(i, j, ip, v, model) for ip in range(d_size)])
    return v_new


@njit(parallel=True)
def get_greedy(v, model):
    "Compute a v-greedy policy."
    d_size, z_size = len(model.d_grid), len(model.Z)
    σ = np.empty_like(v, dtype=int32)
    for i in prange(d_size):
        for j in range(z_size):
            σ[i, j] = argmax([B(i, j, ip, v, model) for ip in range(d_size)])
    return σ


def value_iteration(model, tol=1e-8):
    "Implements VFI."
    vz = np.zeros((len(model.d_grid), len(model.y_grid)))
    v_star = successive_approx(lambda v: T(v, model), vz, tolerance=tol)
    return get_greedy(v_star, model)



# == JAX version == #




# == Tests == #

if __name__ == '__main__':
 
    model = create_OER_model(estimate=True)
    print("Starting VFI.")
    qe.tic()
    out = value_iteration(model)
    elapsed = qe.toc()
    print(out)
    print(f"VFI completed in {elapsed} seconds.")

    out.shape

