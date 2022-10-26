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

import numpy as np
from collections import namedtuple
import quantecon as qe
from numba import njit, prange, int32
import sys
import tpm
import jax
import jax.numpy as jnp
from jax import jit
from quantecon.optimize.scalar_maximization import brent_max
from functools import partial


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
              'rgrid',
              'dmat', # Debt grid as a matrix
              'ymat', # Income grid as a matrix
              'dpmat' # Grid for debt tomorrow as matrix
              ))

# -

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
    
    Π = np.reshape(Π, (ny, ny, 1))
    d = np.reshape(dgrid, (DN, 1, 1))
    y = np.reshape(ygrid, (1, ny, 1))
    dp = np.reshape(dgrid, (1, 1, DN))


    
    
    return Model(σ=σ, ħ=ħ, ω=ω, ξ=ξ, α=α, β=β, γ=γ, π=π, ψ=ψ,
                  DMIN=DMIN, DMAX=DMAX, DN=DN,
                  WMIN=WMIN, WMAX=WMAX, WN=WN, YN=YN, RN=RN,
                  RSTAR=RSTAR, 
                  Θ=Θ, Σ=Σ,
                  NUMSIM=NUMSIM, BURN=BURN,
                  Π=Π, S=S, Xvec=Xvec, 
                  dgrid=dgrid, ygrid=ygrid, rgrid=rgrid, dmat = d, ymat = y, dpmat = dp)
    

@jit
def T_vec(v,model):
    
    dgrid, ygrid, rgrid = model.dgrid, model.ygrid, model.rgrid
    ω, ξ, ħ, α, β, σ = model.ω, model.ξ, model.ħ, model.α, model.β, model.σ
    d, y = model.dmat, model.ymat
    R = jax.device_put(1/(1 + rgrid))
    Π = jax.device_put(model.Π)
    ny = len(model.Π)
    dp = jax.device_put(jnp.copy(d))
    vp = jnp.dot(v, Π)
    cT = dp*R + y - d
    c = (ω * cT ** (1 - 1/ξ) + (1 - ω) * (ħ ** α) ** (1 - 1/ξ)) ** (1/(1 - 1/ξ)) 
    m = jnp.where(c > 0, (c**(1 - σ) - 1)/ (1 - σ) + β * vp, -jnp.inf)
    
    return jnp.max(m, axis=2)


def vfi_iterator(v_init, model, tol=1e-6, max_iter=50_000):
    error = tol + 1
    i = 0
    v = v_init
    while error > tol and i < max_iter:
        new_v = T_vec(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        v = new_v

        if i % 100 == 0:
            print(f"Iteration {i}")
        i += 1

    if i == max_iter:
        print(f"Warning: iteration hit upper bound {max_iter}.")
    else:
        print(f"\nConverged at iteration {i}.")
    return v


model = create_OER_model(DN=501)
vz = np.zeros((len(model.dgrid), len(model.ygrid)))
vz = jax.device_put(vz)
out = vfi_iterator(vz, model)

#T_vec_jit = jax.jit(T_vec)
out = vfi_iterator(vz, model)

pi = model.Π
pi = np.reshape(pi, (218,218,1))
pi.shape
