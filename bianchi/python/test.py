import numpy as np

try:
    x = np.load('markov_tran.npz')
    print('Loaded from file.')
    assert False, "Assertion triggered."
except FileNotFoundError:
    print('File not found')
