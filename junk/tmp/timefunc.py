#!/usr/bin/env python3
#%%

import numpy as np
from matplotlib import pyplot as plt

t = np.arange(0,1000)

# linear velocity 
vi = 2
F1 = vi * t
F1=0*t

# step function
di = 1
Ti = 400
F2 = di * (t-t[Ti] > 0)

# exp function
aei = 1
Tei = 200
tauei = 100
F3 = aei * (t-t[Tei] > 0) * (1-np.exp(-(t-Tei)/tauei))

# log function
ali = 1
Tli = 200
tauli = 100
old = np.seterr(invalid='ignore', divide='ignore')
F4 = ali * (t-t[Tli] > 0) * np.nan_to_num(np.log(1+(t-Tli)/tauli), nan=0, neginf=0)
np.seterr(**old)
## Sum of all tim fuctions
F = F1 + F2 + F3 + F4

## Plot
plt.figure(figsize=[5,3])
plt.plot(t, F, label='Total')
plt.plot(t, F1, label='linear')
plt.plot(t, F2, label='step')
plt.plot(t, F3, label='exp')
plt.plot(t, F4, label='log')
plt.xlabel('Time [e.g., days]')
plt.ylabel('Velocity [e.g., mm]')
plt.legend()
plt.show()
# %%

from mintpy.utils import ptime

exp_list = [['20201231', '20210101'], [60, 365]]
t_exps   = ptime.yyyymmdd2years(exp_list[0])
tau_exps = np.array(exp_list[1]) / 365.25

for i, (t_exp, tau_exp) in enumerate(zip(t_exps, tau_exps)):
    print('no.{}\t{:.16f}\ttau:{:.16f} years'.format(i, t_exp, tau_exp))
# %%
exp_list = ['20200101','60','20210601','120.5']

model = dict()
model['exp'] = exp_list

if len(model['exp']) % 2 != 0:
    raise ValueError('num of exponential model parameters is odd! Specify each exp func with first a onset date AND followed by a char time\n'+
                    '\te.g.,  --exp  20061014 60 \n' +
                    '\t  or   --exp  20110311 80.5 20120928 200.8    ... and so on')
else:
    tmp_arr = np.reshape(model['exp'], (int(len(model['exp'])/2), 2))
    model['exp'] = [list(tmp_arr[:,0]), list(tmp_arr[:,1].astype(float))]
    print(model['exp'])
# %%
''.join([])
# %%
