#!/usr/bin/env python3

#%%
import numpy as np

t1 = '20190710'
t2 = '20200625'

t1 = '{}-{}-{}'.format(t1[:4], t1[4:6], t1[6:])
t2 = '{}-{}-{}'.format(t2[:4], t2[4:6], t2[6:])


date_obj = np.arange(t1, t2, dtype='datetime64[D]').astype('M8[D]').astype('O')

[obj.strftime('%Y%m%d') for obj in date_obj]
# %%
import datetime as dt
a = ['20160228', '20200102']
[dt.datetime.strptime(aa, '%Y%m%d') for aa in a]

# %%
aa=54.6

if isinstance(aa, float):
    aa_check = np.array([aa])
    print('is float')
else:
    aa_check = np.array(aa)

len(aa_check)
# %%
aa=[12.3, 12.3, 2312.0]
bb = ['a','d','t']
cc=[1,2,3]
print(list(zip(bb,aa,cc)))
# %%
e2 = 20

# check empty e2 due to the rank-deficient G matrix for sigularities.
e2 = np.array(e2)
if e2.size == 0:
    print('size=0')
# %%
