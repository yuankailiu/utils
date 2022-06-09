## Python script to look at phase closure tripelets for ionosphere corrections 
## I forget how to use it...
## ykliu @ APRIL 2022

import h5py 
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np



# Load data
stack = h5py.File('ifgramStack.h5','r')
# stack = h5py.File('ifgramStack_msk.h5','r')
iono_stack = stack['ionoPhase']
# iono_stack = stack['unwrapPhase']
# iono_stack = stack_msk['ionoPhase']

# Load coherence
coh = h5py.File('../avgSpatialCoh_radar.h5','r')
# coh = h5py.File('../avgSpatialCoh_radar_msk.h5','r')
coh = coh['coherence'][:]


# Need to put ionosphere in the same reference frame 

# ref_x = 200
# ref_y = 400

ref_x = int(stack.attrs['REF_X'])
ref_y = int(stack.attrs['REF_Y'])


# Get wavelength for scaling from radians 
# Work in cm 
wvl = float(stack.attrs["radarWavelength"])*100 # scale into cm

trip_num = 155
# For T115 we have 638 pairs 
# Only 638/2 ionosphere pairs 

# igram_dates = stack_msk['date'][:]
igram_dates = [[i[0].decode('utf-8'),i[1].decode('utf-8')] for i in stack['date'][:]]
aqn_dates = [j for i in igram_dates for j in i] # https://stackoverflow.com/questions/8327856/how-to-extract-nested-lists
aqn_dates = list(set(aqn_dates))
aqn_dates.sort()

triplet_sum = 0
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
    triplet = (AC - (AB+BC))*wvl/(4*np.pi) # scale from rad to cm
    triplet_sum += triplet

    # Stop if we've hit a limit
    if i>trip_num:
        break
    elif i==len(aqn_dates)-2:
        trip_num = i
        break



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
fig,axes = plt.subplots(1,2)
im = axes[0].imshow(coh,cmap='viridis',interpolation='none',vmin=0,vmax=1)
cbar = fig.colorbar(im, ax=axes[0])
cbar.set_label('Coherence')
im = axes[1].imshow(triplet_sum,cmap='RdBu_r',interpolation='none',vmin=-15, vmax=15)
cbar = fig.colorbar(im, ax=axes[1])
cbar.set_label('Phase triplet sum (cm)')
plt.title('Triplet sum for {} triplets'.format(trip_num))
plt.savefig('coherence_phase_trips.pdf')

# # Plot phase closure against coherence 
# fig = plt.figure()
# plt.scatter(coh.flatten(),triplet_sum.flatten(),s=0.1)
# plt.title('Triplet sum against coherence')
# plt.xlabel('Coherence')
# plt.ylabel('Triplet sum (cm)')



stack.close()
plt.ion()
plt.show()
