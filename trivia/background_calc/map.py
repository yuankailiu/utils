#!/usr/bin/env python3
#%%
# ykliu @ June20 2021
# Doing calculation of spatial bins

import pygmt
import numpy as np
import matplotlib.pyplot as plt
import dataUtils as du
from select_cat import run_select
from pyproj import Transformer
from itertools import cycle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({'font.size': 16})


def gmt_map(lons, lats, mags, deps, meta, pts=None, slab_model=None, title=' ', cmap='inferno', show=False):
    """make GMT map of the region with seismicity plotted"""
    # Specify the approx fault trace
    extent      = meta['EXTENT']
    start_lalo  = meta['FAULT_START']   # fault trace start
    end_lalo    = meta['FAULT_END']     # fault trace end
    projection  = meta['MAP_PROJ']      # get GMT map projection

    font = "6p,Helvetica-Bold"
    plt.rcParams.update({'font.size': 12})
    fig  = pygmt.Figure()
    grid = pygmt.datasets.load_earth_relief(resolution="30s", region=extent)
    fig.grdimage(grid=grid, projection=projection, frame="a", cmap="bathy", shading=0.5)
    fig.colorbar(
        frame    = ["a2000", "y+lm"],
        position = "n0.7/0.05+w4c/0.3c",
    )
    fig.coast(
        resolution = 'high',
        region     = '{}/{}/{}/{}'.format(extent[0], extent[1], extent[2], extent[3]),
        projection = projection,
        land='lightgray',
        #water='lightblue',
        borders    = '1/0.2p',
        shorelines = ['1/0.1p', '2/0.1p'],
        frame      = 'a',
        map_scale  = 'n0.5/0.1+w500k+c56',
    )
    pygmt.makecpt(cmap=cmap, series=[0, np.max(deps)], reverse=True)
    fig.plot(
        x     = lons,
        y     = lats,
        sizes = 0.006*(1.6**mags),
        color = deps,
        cmap  = True,
        style = 'cc',
        #pen   = 'black',
        #transparency = 80,
    )
    #fig.contour(
    #    data = slab_model_xyz,
    #    skip = True,
    #    level = 50,
    #    pen = "0.2p",
    #    transparency = 60,
    #)
    if slab_model is not None:
        print('Using a slab model from : {}'.format(slab_model))
        fig.grdcontour(
            grid = slab_model,
            annotation = 20,
            interval = 10,
            transparency = 60,
            limit = "-100/0"
        )
    fig.colorbar(
        frame    = ["x+lDepth", "y+lkm"],
        position = "n0.12/0.65+w3c/0.2c+h",
    )
    if False: # plot projection straight line
        fig.plot(
            y   = [start_lalo[0], end_lalo[0]],
            x   = [start_lalo[1], end_lalo[1]],
            pen = "0.5p,black,-",
        )
    fig.plot(
        data = 'trench.gmt',
        pen = "2p,white,4_3:2p",
    )

    for key in meta:
        if key.startswith('LOC_'):
            loc = key.split('_')[1]
            yx  = meta[key]
            fig.plot(x=yx[1], y=yx[0], style="s0.18c", pen="0.5p,black", color="white")
            fig.text(x=yx[1]+1.2, y=yx[0]-0.25, text=loc, font=font)
        elif key.startswith('PROFILE'):
            loc = key.split('_')[1]
            yx  = meta[key]
            fig.plot(x=[yx[0][1],yx[1][1]], y=[yx[0][0],yx[1][0]], pen="0.3p,black")
            fig.text(x=yx[0][1]+0.2, y=yx[0][0]-0.1, text=loc, font=font)

    fig.meca(spec="alaska_gcmt.txt", scale='8p', convention='mt')

    if pts is not None:
        pts = np.array(pts)
        cyc = cycle(['red', 'blue', 'green', 'yellow', 'purple'])
        for pt in pts:
            plat = pt[0]
            plon = pt[1]
            fig.plot(
                x     = plon,
                y     = plat,
                color = next(cyc),
                style = 't0.3c',
                pen   = '0.5p,black',
            )
        cyc = cycle(['red', 'blue', 'green', 'yellow', 'purple'])
        for pt in pts:
            plat = pt[2]
            plon = pt[3]
            fig.plot(
                x     = plon,
                y     = plat,
                color = next(cyc),
                style = 'c0.25c',
                pen   = '0.5p,black',
            )
    if show:
        print('Just show figure in matplotlib')
        fig.show()
    else:
        print('Save it to PNG and show the PNG file')
        fig.savefig('{}/{}.png'.format(meta['PIC_DIR'], title), transparent=True, show=True, dpi=300)



#%%
if __name__ == '__main__':

    ## =====================  0. Read metadata  ============================ ##
    my_json = './init.json'
    meta = du.read_meta(my_json)
    slab_model_xyz = './Slab2_AComprehe/alu_slab2_dep_02.23.18.xyz'
    slab_model = './Slab2_AComprehe/alu_slab2_dep_02.23.18.grd'


    ## =====================  2. Read final catalog  ======================== ##
    cat = du.read_cat('outcat_add_mc.csv')
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat

    dep_max = 200
    select = dep<=dep_max
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = np.array(cat).T[select].T
    dtime_s = dtime_s.astype('float')
    lat  = lat.astype('float')
    lon  = lon.astype('float')
    dep  = dep.astype('float')
    mag  = mag.astype('float')

    dist = du.epi2projDist(lon, lat, meta['FAULT_START'], meta['FAULT_END'])


    ## =====================  3. Make GMT plot  ======================== ##
    gmt_map(lon, lat, mag, dep, meta, slab_model=slab_model, title='Alaska_70km_meca_slab')

    print('done GMT plotting')

#%%
    ## =====================  4. Make profiles  ======================== ##
    _, tmp     = du.read_xyz(meta['SLAB_DEPTH'])
    _, tmp_thk = du.read_xyz(meta['SLAB_THICK'])
    xyz = np.vstack((tmp[:,1], tmp[:,0]-360, tmp[:,2], tmp_thk[:,2])).T

    for key in meta:
        if key.startswith('PROFILE'):
            loc = key.split('_')[1]
            yx  = meta[key]

            n = 50
            slab = []
            for prof_grid in np.vstack((np.linspace(yx[0][0], yx[1][0], n), np.linspace(yx[0][1], yx[1][1], n))).T:
                tmp = du.find_nearest_k(xyz, prof_grid, k=1)[0][0]
                if np.array(tmp).size != 1:
                    slab.append(tmp)
            slab = np.array(slab)
            slab_x = du.epi2projDist(slab[:,1], slab[:,0], yx[0], yx[1])

            dist_pf, idx = du.epi2projDist(lon, lat, yx[0], yx[1], dperMax=0.2)
            print('profile {} at {}; {} events'.format(loc, yx, len(dist_pf)))
            plt.figure(figsize=[10,3])
            plt.scatter(dist_pf, -dep[idx], fc='cornflowerblue', ec='k', lw=0.4)
            plt.plot(slab_x, slab[:,2], c='k')
            plt.plot(slab_x, slab[:,2]-slab[:,3], c='k')
            plt.ylabel('Depth [km]')
            plt.xlabel('Distance to trench [km]')
            plt.xlim(0,500)
            plt.ylim(-200,0)
            plt.title('Profile {}'.format(loc))
            plt.show()



#%%
    ## =====================  5. Hypos from slab  ====================== ##
    print('loop slab mesh and hypocenters')
    dd = []
    for i in range(len(lat)):
        if i%5000 == 0:
            print('at {} event'.format(i))

        hypo = np.array([lat[i], lon[i], -dep[i]])
        dd.append(du.find_nearest_k(xyz, hypo, k=1)[1][0])

        # get distance to the nearest triangular mesh
        if False:
            # setup your projections
            transformer = Transformer.from_crs(4326, 2100)
            points = [hypo[:2]]
            hypo_p = []
            for pt in transformer.itransform(points):
                hypo_p.append([pt[0], pt[1], hypo[2]])
            hypo_p = np.array(hypo_p[0])

            points = lalo[:,:2]
            grids = []
            j=0
            for pt in transformer.itransform(points):
                grids.append([pt[0], pt[1], lalo[j,2]])
                j += 1
            grids = np.array(grids)
            dd.append(du.dist_3Dplane(grids, hypo_p)[0])
    dd = np.array(dd)
    print('done with looping')


# %%
    print('slab distance histogram')
    cut1 = 12
    cut2 = 25
    sel_near = dd<cut1
    sel_mid  = (dd>=cut1)*(dd<cut2)
    sel_far  = dd>cut2

    msg = 'Total # = {} \n'.format(len(mag))
    msg += 'near, # = {}\n'.format(np.sum(sel_near))
    msg += 'mid, # = {}\n'.format(np.sum(sel_mid))
    msg += 'far, # = {}\n'.format(np.sum(sel_far))

    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(dd, bins=50, range=[0,60], fc='lightgrey')
    plt.axvline(x = cut1, c='k', ls='--')
    plt.axvline(x = cut2, c='k', ls='--')
    plt.xlabel('Distance from slab [km]')
    plt.ylabel('# count of hypos')
    plt.text(0.6, 0.5, msg, transform=ax.transAxes)
    plt.title('Slab distance')
    plt.show()

    print('depth histogram / slab distance')
    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(dep[sel_near], bins=50, range=[0,dep_max], alpha=0.7, label='near')
    n, bins, patches = plt.hist(dep[sel_mid], bins=50, range=[0,dep_max], alpha=0.7, label='mid')
    n, bins, patches = plt.hist(dep[sel_far], bins=50, range=[0,dep_max], alpha=0.7, label='far')
    plt.xlabel('Depth [km]')
    plt.ylabel('# count of hypos')
    plt.legend(loc='upper right')
    plt.title('Event depths')
    plt.show()

# %%
    plt.figure(figsize=[8,6])
    plt.scatter(dd, -dep, fc='lightgray', ec='k', lw=0.4, alpha=0.8)
    plt.ylabel('Event depth [km]')
    plt.xlabel('Distance to slab surface [km]')
    plt.show()


# %%
    print('slab depth histogram')
    cut0 = 18
    cut1 = 50
    cut2 = 90
    sel_surf  = dep<cut0
    sel_shal  = (dep>=cut0)*(dep<cut1)
    sel_inte  = (dep>=cut1)*(dep<cut2)
    sel_deep  = dep>cut2

    msg =  'surface  # = {}\n'.format(np.sum(sel_surf))
    msg += 'shallow  # = {}\n'.format(np.sum(sel_shal))
    msg += 'interm   # = {}\n'.format(np.sum(sel_inte))
    msg += 'deep     # = {}\n'.format(np.sum(sel_deep))

    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(dep, bins=50, range=[0,200], fc='lightgrey')
    plt.axvline(x = cut0, c='k', ls='--')
    plt.axvline(x = cut1, c='k', ls='--')
    plt.axvline(x = cut2, c='k', ls='--')
    plt.xlabel('Depth [km]')
    plt.ylabel('# count of hypos')
    plt.text(0.6, 0.5, msg, transform=ax.transAxes)
    #plt.title('Slab distance')
    plt.show()

    print('depth histogram / slab distance')
    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(dd[sel_surf], bins=50, range=[0,60], alpha=0.5, label='surface')
    n, bins, patches = plt.hist(dd[sel_shal], bins=50, range=[0,60], alpha=0.5, label='shallow')
    n, bins, patches = plt.hist(dd[sel_inte], bins=50, range=[0,60], alpha=0.5, label='interm')
    n, bins, patches = plt.hist(dd[sel_deep], bins=50, range=[0,60], alpha=0.5, label='deep')
    plt.xlabel('Distance to slab [km]')
    plt.ylabel('# count of hypos')
    plt.legend(loc='upper right')
    #plt.title('Event depths')
    plt.show()
# %%
