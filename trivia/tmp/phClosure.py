#!/usr/bin/env python3
# To play to integer of phase closre 
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# Define a wrap operator to wrap values into [-pi, pi)
def wrapper(val):
    return (val+np.pi)%(2*np.pi) - np.pi

# Three SAR acquisitions: ti, tj, tk
# Unwrapped interferometric phases for one pixel (radian):
d_ij = 4
d_jk = 3
d_ik = 1015

# phase closure of the triplet
C_ijk = d_ij + d_jk - d_ik
print('Phase closure, C_ijk = ', C_ijk)

# wrapped C_ijk
print('Wrapped phase closure, wrap(C_ijk) = ', wrapper(C_ijk))

# Diff
print('C_ijk - wrap(C_ijk) = ', C_ijk - wrapper(C_ijk))

# int of phase closure
C_int = (C_ijk - wrapper(C_ijk)) / (2*np.pi)
print('C_int = {:.1f}'.format(C_int))

# %%
