#!/usr/bin/env python3

#%%

from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import dataUtils as du
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({'font.size': 22})

from geopy.distance import geodesic

my_json = './init.json'
meta = du.read_meta(my_json)
trench = du.read_plate_bound(meta['PLATE_BOUND_FILE'], meta['PLATE_BOUND_NAME'], v=False)

lalo =  [  59.2734, -145.0874]
a, b =  [  59.2108, -145.0296], [  59.2108, -145.0296]

plt.figure()
plt.scatter(lalo[1], lalo[0])
plt.scatter(a[1],a[0],c='k')
plt.scatter(b[1],b[0],c='k')
plt.show()

tl = 5
k = 2

array = np.asarray(trench)
value = np.asarray(lalo)

lat0 = value[0]
lon0 = value[1]
sub  = (array[:,0]>lat0-tl)*(array[:,0]<lat0+tl)*(array[:,1]>lon0-tl)*(array[:,1]<lon0+tl)

if np.sum(sub) < 3:
    print('less than 3 candidates')

sub_arr = array[sub]
dist = []

if len(value) == 3:
    # 3D distance
    for i in range(len(sub_arr)):
        dist.append(np.sqrt(geodesic(sub_arr[i,:2], value[:2]).km**2 + np.abs(sub_arr[i,2]-value[2])**2))
elif len(value) == 2:
    # 2D distance
    for i in range(len(sub_arr)):
        dist.append(geodesic(sub_arr[i], value).km)

dist = np.array(dist)

dupl = 0
while True:
    idxs = np.argsort(dist)[:k+dupl]
    dupl = k-len(set(dist[idxs]))
    if dupl <= 0:
        idxs = du.find_indeces(dist, sorted(list(set(dist[idxs]))))
        break

nk_idxs = du.find_indeces(array, sub_arr[idxs])
print(nk_idxs)
print(dist[idxs])





#%%
file = './Slab2_AComprehe/alu_slab2_dep_02.23.18.xyz'

pdxyz, xyz = du.read_xyz(file)
xyz = np.vstack((xyz[:,1], (xyz[:,0]-360), xyz[:,2])).T

# %%
k = 3
hypo = [55, -160, -30]
print('Given hypocenter:\n{}\n'.format(hypo))
lalo, dist = du.find_nearest_k(xyz, hypo, k=k)

print('Closest triangle:')
for i in range(k): print(lalo[i], 'Dist = {:.2f} km'.format(dist[i]))

# %%
# setup your projections
transformer = Transformer.from_crs(4326, 2100)

points = [hypo[:2]]
hypo_p = []
for pt in transformer.itransform(points):
    print('{:.3f} m  {:.3f} m'.format(*pt))
    hypo_p.append([pt[0]/1e3, pt[1]/1e3, hypo[2]])
hypo_p = np.array(hypo_p[0])

points = lalo[:,:2]
grids = []
i=0
for pt in transformer.itransform(points):
    print('{:.3f} m  {:.3f} m'.format(*pt))
    grids.append([pt[0]/1e3, pt[1]/1e3, lalo[i,2]])
    i += 1
grids = np.array(grids)

# %%

n_dist, n_vec = du.dist_3Dplane(grids, hypo_p)
print('Distance to plane = {:.2f} km'.format(n_dist))
print('Normal vector = {}'.format(n_vec))

hypo_q = hypo_p + n_dist * n_vec

# %%
poly = np.concatenate((grids, grids[0,:].reshape([1,3])))
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(projection='3d')
ax.scatter(hypo_p[0], hypo_p[1], hypo_p[2], color='r')
for i in range(3):
    ax.scatter(grids[i,0], grids[i,1], grids[i,2], marker='s', color='b')
ax.plot(poly[:,0], poly[:,1], poly[:,2], color='b')
ax.plot_trisurf(grids[:,0], grids[:,1], grids[:,2], color='lightgrey', alpha=0.4)
ax.plot([hypo_p[0],hypo_q[0]], [hypo_p[1],hypo_q[1]], [hypo_p[2],hypo_q[2]], color='r')
ax.scatter(hypo_q[0], hypo_q[1], hypo_q[2], color='r', marker='^')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# %%
