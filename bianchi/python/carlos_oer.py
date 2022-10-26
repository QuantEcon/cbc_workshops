# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.12 ('base')
#     language: python
#     name: python3
# ---

# ## Optimal Exchange Rate Regime
#
# This is my attempt to replicate the results for the OER model

import numpy as np
from collections import namedtuple
import quantecon as qe
from numba import njit, prange, int32
import sys
import tpm

# +

Model = namedtuple(
    'Model', ('σ', # Inverse of intertemporal elasticity of consumption
              'ħ', # H bar = Full endowment t units of labor 
              'ω', # Share of tradables in the consumption aggregator
              'ξ', # Elasticity of substition between tradables and nontradables
              'α', # Labor share in notraded output
              'β', # Annual subjective discount factor
              'π', # Annual Inflation Target
              'ψ', # Annual Growth rate
              'γ', # Annual Parameter governing DNWR including inflation and growth
              'DMIN', # Lower Bound Debt Grid
              'DMAX', # Upper Bound Debt Grid (Determined by Natural debt limit)
              'DN', # Number of points in Debt Grid
              'WMIN', # Lower Bound Wage Grid
              'WMAX', # Upper Bound Wage Grid
              'WN', # Number of points in Wage Grid
              'YN', # Number of points in the grid for the Tradable Output
              'RN', # Number of points in the grid for the Foreign Interest Rate
              'RSTAR', # Average Annual Real Interest Rate
              'Θ', # Autocorrelation Matrix for Stochastic Processes
              'Σ', # Variance - Covariance Matrix for Stochastic Processes
              'NUMSIM', # Number of periods to be simulated
              'BURN', # Burn-in periods in the simulations
              'Π', # Transition probability matrix
              'S', # Vector of states * Full array of possibilities
              'Xvec', # Simulated path for discrete VAR
              'dgrid', # Bond grid
              'ygrid', # Grid for Tradable Output
              'rgrid' # Grid for Interest Rate
              ))

# -

# ## Create the Model
#
# Creates and instance of the OER model, including the discretize VAR and needed simulations for $Y_t^T$ and $r_t$

# +
def create_OER_model(σ=5.0, ħ=1.0, ω=0.19, ξ=0.43 ,α=0.75,
                    β=0.9571,
                    γ=0.96, 
                    π=0,
                    ψ=0,
                    DMIN=-5, DMAX=5.4, DN=501,
                    WMIN=0.1,WMAX=14.3,WN=500,
                    YN=21, RN=11,
                    RSTAR=0.021048,
                    NUMSIM = 1_000_000, BURN = 100_000, estimate=False):
    
    "Creates a parameterization with default values for the Optimal Exchange Rate Model."
    π = 1 + π
    γ = γ/((1 + π)*(1 + ψ))
    Θ = np.array([[0.72137370, -0.1323976], [0.0028990, 0.9705325]])
    Σ = np.array([[0.00116490, -0.0000131], [-0.0000131, 0.0001328]])
    
    N = np.array([YN, RN])
    
    # If needed, discretize the VAR process
    # Otherwise, load the matrix in file
    if estimate:
        Π, Xvec, S = tpm.tpm(Θ, Σ , N, T=NUMSIM, Tburn=BURN)
        np.savez("markov_tran.npz", Π=Π, Xvec=Xvec, S=S, N=N)
    
    
    # Check whether the stored matrix conforms to the dimensions specified by YN and RN
    data = np.load('markov_tran.npz')
    Π = data['Π']
    Nck = data['N']
    Nck=Nck[0]*Nck[1]
    if YN*RN != Nck:
        print('Error: Matrix in file does not have the same dimension as implied by inputs. You need to discretize the VAR again. ', file=sys.stderr)
        sys.exit()

    Xvec = data['Xvec']
    S = data['S']
    
    # Shift Π from column to row major
    Π = np.ascontiguousarray(Π)


    rgrid = np.exp(S[:,1])*(1 + RSTAR) - 1
    ygrid = np.exp(S[:,0]) 
    ny = len(ygrid)
    dgrid = np.linspace(DMIN, DMAX, DN)
    
    return Model(σ=σ, ħ=ħ, ω=ω, ξ=ξ, α=α, β=β, γ=γ, π=π, ψ=ψ,
                  DMIN=DMIN, DMAX=DMAX, DN=DN,
                  WMIN=WMIN, WMAX=WMAX, WN=WN, YN=YN, RN=RN,
                  RSTAR=RSTAR, 
                  Θ=Θ, Σ=Σ,
                  NUMSIM=NUMSIM, BURN=BURN,
                  Π=Π, S=S, Xvec=Xvec, 
                  dgrid=dgrid, ygrid=ygrid, rgrid=rgrid)
    

# -

# ## Setting-up the solution method
#
# First we solve the OER with VFI. Here is the right-hand side of the Bellman Equation:

@njit
def B(i, j, ip, v, model):
    """
    The right-hand side of the Bellman equation with candidate value v and
    arguments w[i], y[j], wp[ip].

    """
    dgrid, ygrid, rgrid = model.dgrid, model.ygrid, model.rgrid
    ω, ξ, ħ, α, β, σ = model.ω, model.ξ, model.ħ, model.α, model.β, model.σ
    Π =  model.Π
    y, R, d, dp = ygrid[j], 1 + rgrid[j], dgrid[i],  dgrid[ip]
    cT = y + dp/R - d # Consumption of tradable goods
    c = (ω * cT ** (1 - 1/ξ) + (1 - ω) * (ħ ** α) ** (1 - 1/ξ)) ** (1/((1 - 1/ξ))) 
    
    if c > 0:
        return (c**(1 - σ) - 1)/ (1 - σ) + β * np.dot(v[ip, :], Π[j, :]) 
    return - np.inf


# ## Now we set up the Bellman operator

@njit(parallel=True)
def T(v, model):
    "The Bellman operator."
    bsize, ysize = len(model.dgrid), len(model.ygrid)
    v_new = np.empty_like(v)
    for i in prange(bsize):
        for j in range(ysize):
            v_new[i, j] = max([B(i, j, ip, v, model) for ip in range(bsize)])
    return v_new


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


@njit(parallel=True)
def get_greedy(v, model):
    "Compute a v-greedy policy."
    bsize, ysize = len(model.dgrid), len(model.ygrid)
    σ = np.empty_like(v, dtype=int32)
    for i in prange(bsize):
        for j in range(ysize):
            σ[i, j] = argmax([B(i, j, ip, v, model) for ip in range(bsize)])
    return σ


def successive_approx(T,                     # Operator (callable)
                      x_0,                   # Initial condition
                      tolerance=1e-8,        # Error tolerance
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


def value_iteration(model, tol=1e-8):
    "Implements VFI."
    vz = np.zeros((len(model.dgrid), len(model.ygrid)))
    v_star = successive_approx(lambda v: T(v, model), vz, tolerance=tol)
    return get_greedy(v_star, model)


# +
model = create_OER_model(estimate=True)

print("Starting VFI.")
qe.tic()
out = value_iteration(model)
elapsed = qe.toc()
print(out)
print(f"VFI completed in {elapsed} seconds.")
# -

out.shape
