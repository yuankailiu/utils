#%%
#!/usr/bin/env python3
############################################################
# This script implement screw dislocation model
# in Paul Segall 2010 Chp. 2
# YKL @ 2023-04-25
############################################################

import numpy as np
import matplotlib.pyplot as plt

def disloc_confined_slip(x, s, d1, d2):
    """
    Eqn. 2.36 of Segall 2010
    Case-specific for postseismic slip with confined slip between depths d1 and d2 (d1 > d2).
    But also be viewed as a generalized screw dislocation for both coseismic (d2=0) or interseismic (d1=inf)
        For confined slip between depths d1 and d2, the displacements at the free surface are:
        u_3 = (-s/np.pi) * (np.arctan2(x_1,d1) - np.arctan2(x_1,d2))
            u_3 : displacement at surface (x_2=0) in the x_3 (antiplane) direction
            s   : slip (positive as left-lateral)
            x_1 : distance along x_1 direction
    """
    u_3 = (-s/np.pi) * (np.arctan2(x,d1) - np.arctan2(x,d2))
    return u_3


#%%

x = np.linspace(-100, 100, 200)
s = 10


# interseismic slip
plt.figure()
plt.title('Interseismic slip')
d1 = np.inf
for d2 in np.arange(0,20,4):
    u = disloc_confined_slip(x, s, d1, d2)
    plt.plot(x, u, label=fr'$d_2=${d2}')
plt.legend()
plt.show()

# %%
# Coseismic slip
plt.figure()
plt.title('Coseismic slip')
d2 = 0
for d1 in np.arange(0,20,4):
    u = disloc_confined_slip(x, s, d1, d2)
    plt.plot(x, u, label=fr'$d_1=${d1}')
plt.legend()
plt.show()
# %%

# Postseimic slip
plt.figure()
plt.title('Postseismic slip')
d2 = 30
for d2 in np.arange(10,40,5):
    d1 = d2+5
    u = disloc_confined_slip(x, s, d1, d2)
    plt.plot(x, u, label=fr'{d2}~{d1} km')
plt.legend()
plt.show()
# %%

# combination of inter- post-seismic
u_inter = disloc_confined_slip(x, s=5, d1=np.inf, d2=0)
u_post = disloc_confined_slip(x, s=20 , d1=30,    d2=20)
plt.figure()
plt.plot(x, u_inter, label='interseismic')
plt.plot(x, u_post, label='postseismic')
plt.plot(x, u_inter+u_post, label='Total')
plt.legend()
plt.show()
# %%

def exp_relaxation(t, tau):
    s = 1 - np.exp(-(t-t[0])/tau)
    return s

t = np.arange(1995, 2025)

s = exp_relaxation(t, tau=7)
plt.figure()
plt.plot(t, s)
plt.show()

per_early = (s[t==1996]-s[t==1995])[0]
per_late = (s[t==2022]-s[t==2014])[0]
baer_est = 400

print(f'Later/early postseismic ratio = {per_late/per_early:.2f}')
print(f'Early postseimic is {baer_est} mm')
print(f'Late postseismic is {baer_est*per_late/per_early:.2f} mm')

# given 1995-1996 early postseismic is 500 mm need to have relaxation time of 20 yr
# to allow total 140~160 mm of postseismic occur during 2014-2022 (20mm/yr for 7~8 yrs)
# are these values reasonble? Relaxation time, early postseimic slip rate form Baer et al.,?

# %%
