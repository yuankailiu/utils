#!/usr/bin/env python3
## Python script to look at phase closure tripelets for ionosphere corrections
## Ollie 2022

import h5py
import random
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load data
stack = h5py.File('./inputs/ifgramStack.h5','r')
iono_stack = stack['ionoPhase']
# iono_stack = stack['unwrapPhase']
# iono_stack = stack_msk['ionoPhase']
n_pairs = iono_stack.shape[0]
print('Number of pairs:', n_pairs)

# Load connected component
stack = h5py.File('./inputs/ifgramStack.h5','r')
conn_stack = stack['connectComponent']

# Load coherence
coh = h5py.File('./avgSpatialCoh.h5','r')
coh = coh['coherence'][:]

# Load a mask to mask the triplet closure phases
mask = h5py.File('./waterMask.h5', 'r')    # waterMask
mask = mask['waterMask'][:]
#mask = 1*(coh>=0.8)     # use avg coherence


# Need to put ionosphere in the same reference frame
ref_x = int(stack.attrs['REF_X'])
ref_y = int(stack.attrs['REF_Y'])

# Get wavelength for scaling from radians (Work in cm)
wvl = float(stack.attrs["radarWavelength"])*100 # scale into cm
print('wavelength: {} cm'.format(wvl))
print('REF_X: {} pix'.format(ref_x))
print('REF_Y: {} pix'.format(ref_y))


# igram_dates = stack_msk['date'][:]
igram_dates = [[i[0].decode('utf-8'),i[1].decode('utf-8')] for i in stack['date'][:]]
aqn_dates = [j for i in igram_dates for j in i] # https://stackoverflow.com/questions/8327856/how-to-extract-nested-lists
aqn_dates = list(set(aqn_dates))
aqn_dates.sort()
print('Number of acquisition dates:', len(aqn_dates))

# compute num of triplets (trip_num <= len(aqn_dates)-2)
trip_num = len(aqn_dates)-2
print('Num of triplets compute: {}'.format(trip_num))
# For T115 we have 638 pairs
# Only 638/2 ionosphere pairs

triplet_sum = np.zeros_like(iono_stack[0,:,:])
trips = []
skip_trip = 0
print('Dimension: ', triplet_sum.shape, '\n')

for i,date in enumerate(aqn_dates):
    # TODO stop before getting to the end
    AB_dates = [aqn_dates[i],aqn_dates[i+1]]
    AB_idx = igram_dates.index(AB_dates)
    AC_dates = [aqn_dates[i],aqn_dates[i+2]]
    AC_idx = igram_dates.index(AC_dates)
    BC_dates = [aqn_dates[i+1],aqn_dates[i+2]]
    BC_idx = igram_dates.index(BC_dates)

    AB = iono_stack[AB_idx,:,:] - iono_stack[AB_idx,ref_y,ref_x] # Remove reference point
    AC = iono_stack[AC_idx,:,:] - iono_stack[AC_idx,ref_y,ref_x]
    BC = iono_stack[BC_idx,:,:] - iono_stack[BC_idx,ref_y,ref_x]
    if False:   # mask out different connected components
        AB[conn_stack[AB_idx,:,:]!=1] = np.nan
        AC[conn_stack[AC_idx,:,:]!=1] = np.nan
        BC[conn_stack[BC_idx,:,:]!=1] = np.nan

    triplet = (AC - (AB+BC))*wvl/(4*np.pi) # scale from rad to cm

    # apply the mask to the triplet closure phase
    triplet[mask==0] = np.nan

    # check if weird all NaN array (problematic interferogram)
    if np.isnan(triplet).all():
        print('Skip all NaN triplet no. {}'.format(i+1))
        skip_trip += 1
        if np.isnan(AB).all():
            print('  all nan in pair_{}, date_{}'.format(AB_idx, AB_dates))
        if np.isnan(AC).all():
            print('  all nan in pair_{}, date_{}'.format(AC_idx, AC_dates))
        if np.isnan(BC).all():
            print('  all nan in pair_{}, date_{}'.format(BC_idx, BC_dates))
        continue

    # Cummulative sum of closure phase
    triplet_sum += triplet
    trips.append(triplet)
    print('triplet no. {}'.format(i+1))

    # Stop if we've hit a limit
    if i+1>trip_num:
        break
    elif i+1==len(aqn_dates)-2:
        trip_num = i+1
        break

print('Non-NaN mean of summed triplets: ', np.nanmean(triplet_sum), ' cm')

# TODO
# Identify unique dates
# Construct the phase triplet with those dates
# Check for missing ionosphere data
# Calculate triple
# Plot some of these

# TODO
# Share y label
# Put dates as title
# Use consistent color scale
# Add color bar

# for i in range(trip_num):

#     AB_dates = dates[4*i]
#     A_AB = AB_dates[0]
#     B_AB = AB_dates[1]
#     AC_dates = dates[4*i+1]
#     A_AC = AC_dates[0]
#     C_AC = AC_dates[1]
#     BC_dates = dates[4*i+4]
#     B_BC = AC_dates[0]
#     C_BC = AC_dates[1]
#     print(AB_dates,AC_dates,BC_dates)

#     AB = iono_stack[4*i,:,:] - iono_stack[4*i,ref_y,ref_x] # Remove reference point
#     AC = iono_stack[4*i+1,:,:] - iono_stack[4*i+1,ref_y,ref_x]
#     BC = iono_stack[4*i+4,:,:] - iono_stack[4*i+4,ref_y,ref_x]
#     triplet = AC - (AB+BC)
#     triplet_sum += triplet

    # For each triplet, read the date and plot them
    # fig,axes = plt.subplots(1,4,sharey='row')
    # axes[0].imshow(AC,cmap='RdBu_r',interpolation='none')
    # axes[1].imshow(AB,cmap='RdBu_r',interpolation='none')
    # axes[2].imshow(BC,cmap='RdBu_r',interpolation='none')
    # axes[3].imshow(triplet,cmap='RdBu_r',interpolation='none')

# Find limits
# cmax = np.nanmax(triplet_sum)
# cmin = np.nanmin(triplet_sum)

# plot with no interpolation
fig,axes = plt.subplots(figsize=[8,5], ncols=2, sharey=True,  gridspec_kw = {'wspace':0.01})
axes[0].set_title('Average spatial coherence')
im = axes[0].imshow(coh,cmap='viridis',interpolation='none',vmin=0,vmax=1)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Coherence', rotation=270, labelpad=15)
axes[1].set_title('Triplet sum for {} triplets'.format(trip_num-skip_trip))
im = axes[1].imshow(triplet_sum,cmap='RdBu_r',interpolation='none',vmin=-3, vmax=3)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Phase triplet sum (cm)', rotation=270, labelpad=15)
plt.savefig('phaseTrips_map.pdf')
plt.close()


# scatter phase closure against coherence
x = coh[~np.isnan(triplet_sum)]
y = triplet_sum[~np.isnan(triplet_sum)]
n = 10000
print('Random sample {} from {} non-NaN pixels for a scatter plot'.format(n, len(x)))
samps = random.sample(list(np.arange(len(x))), n)
x = x[samps]
y = y[samps]
plt.figure()
plt.scatter(x, y, s=2)
plt.title('Triplet sum against coherence')
plt.xlabel('Coherence')
plt.ylabel('Triplet sum (cm)')
plt.savefig('phaseTrips_coh.pdf')
plt.close()


# plot the median histogram of closure phase of each date
trips = np.array(trips)
medians = np.nanmedian(trips, axis=(1,2))
plt.figure()
plt.hist(medians, bins=30)
plt.xlabel('Median of phase triplet closure phase [cm]')
plt.ylabel('Count')
plt.savefig('phaseTrips_med.pdf')
plt.close()


# plot the mean histogram of closure phase of each date
means = np.nanmean(trips, axis=(1,2))
plt.figure()
plt.hist(means, bins=30, color='coral')
plt.xlabel('Mean of phase triplet closure phase [cm]')
plt.ylabel('Count')
plt.savefig('phaseTrips_mean.pdf')
plt.close()

stack.close()