# Quick calculation of seismicity metrics
# ykliu @ Mar31 2021

#%%
import os
import pygmt
import pickle
import pandas as pd
import matplotlib.dates as mdates
from geopy.distance import geodesic, distance
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.geodetics import kilometers2degrees
from obspy.geodetics import degrees2kilometers
from obspy.geodetics import locations2degrees
import datetime
from datetime import timedelta
from shapely.geometry import Point as shpPoint
from shapely.geometry.polygon import Polygon as shpPolygon 
from geopy.distance import geodesic
plt.rcParams.update({'font.size': 22})


#%% Define functions
def maxc_Mc(mag_arr, plot='no', save='yes', title='', Mc=None, range=[0,4]):
    bin_size = 0.1
    fc = 'lightskyblue'
    ec = 'k'
    k, bin_edges = np.histogram(mag_arr,np.arange(-1,10,bin_size))
    centers      = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    correction   = 0.20
    
    if Mc == None:
        Mc = centers[np.argmax(k)] + correction
    else:
        Mc = Mc
    
    xloc  = 0.50
    yloc1 = 0.90
    yloc2 = 0.70
    yloc3 = 0.55

    if plot == 'yes':
        fig ,ax = plt.subplots(figsize=[18,5], ncols=2)
        ax[0].bar(bin_edges[:-1], k, width=bin_size, fc=fc, ec=ec)
        ax[0].axvline(x=Mc, c='k', ls='--', lw=3)
        ax[0].set_xlim(range[0],range[1])
        ax[0].text(xloc, yloc1, r'$M_{min} = %.3f$'%(Mc), transform=ax[0].transAxes)
        ax[0].text(xloc, yloc2, r'N($M \geq M_{min}$) = %d'%(len(mag_arr[mag_arr>=Mc])), transform=ax[0].transAxes)
        ax[0].text(xloc, yloc3, r'N($M < M_{min}$) = %d'%(len(mag_arr[mag_arr<Mc])), transform=ax[0].transAxes)        
        ax[0].set_xlabel('Magnitude')
        ax[0].set_ylabel('# events')
        ax[0].text(0.35, 1.03, title, fontsize=22, transform=ax[0].transAxes)                

        ax[1].bar(bin_edges[:-1], np.cumsum(k)[-1]-np.cumsum(k), width=bin_size, fc=fc, ec=ec)
        #ax[1].bar(bin_edges[:-1], 100*(1-np.cumsum(k)/np.cumsum(k)[-1]), width=bin_size, fc=fc, ec=ec)
        ax[1].axvline(x=Mc, c='k', ls='--', lw=3)
        ax[1].set_xlim(range[0],range[1])
        ax[1].text(xloc, yloc1, r'$M_{min} = %.3f$'%(Mc), transform=ax[1].transAxes)
        ax[1].text(xloc, yloc2, r'N($M \geq M_{min}$) = %d'%(len(mag_arr[mag_arr>=Mc])), transform=ax[1].transAxes)
        ax[1].text(xloc, yloc3, r'N($M < M_{min}$) = %d'%(len(mag_arr[mag_arr<Mc])), transform=ax[1].transAxes)          
        ax[1].set_xlabel('Magnitude')
        ax[1].set_ylabel('Cumulaive % events')
        ax[1].set_ylabel('# events')
        ax[1].text(0.35, 1.03, title, fontsize=22, transform=ax[1].transAxes)
        ax[1].set_yscale('log')
        plt.show()
        if save == 'yes':
            fig.savefig('{}/magfreq_dist_{}.png'.format(picDir,title), dpi=300, bbox_inches='tight')
        
    elif plot == 'no':
        pass
    else:
        print('Plot keyword is either "yes" or "no"')
    return Mc  


def epoch_Mc(mag, obspyDT, Nt=10, plot='no', title=''):
    Ndt = (obspyDT[-1]-obspyDT[0])/Nt

    epochs = []
    for i in range(int(Nt)):
        epochs.append(np.where(np.array(obspyDT)>=obspyDT[0]+Ndt*i)[0][0])

    Mcs = []
    for i in range(Nt):
        if i==Nt-1:
            sub_mag = mag[epochs[i]:-1]
        else:
            sub_mag = mag[epochs[i]:epochs[i+1]]
        Mcs.append(maxc_Mc(sub_mag, plot=plot, title=title, range=[2,8])) 
    return epochs, Ndt, Mcs


def pt_projline(lalo, start_lalo, end_lalo):
    la = lalo[0]                    # lalo = array(N, 2)
    lo = lalo[1]                    # lalo = array(N, 2)
    start_lalo = np.array(start_lalo) # start_lalo = np.array([lat, lon])
    end_lalo   = np.array(end_lalo)   # end_lalo   = np.array([lat, lon])
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]
    u    = (end_lalo-start_lalo)  # u = np.array([lat, lon])
    v    = (lalo-start_lalo)      # v = np.array([lat, lon])
    un   = np.linalg.norm(u)
    vn   = np.linalg.norm(v)
    cos  = u.dot(v.T) / (un*vn).flatten()
    dpar = vn*cos                 # distance parallel to line
    new_lalo = start_lalo.reshape(1,2) + (dpar*((u/un).reshape(2,1))).T
    dper = np.linalg.norm(new_lalo-lalo, axis=1) # distance perpendicular to line
    return new_lalo, dpar, dper


def pts_projline(lalo, start_lalo, end_lalo):
    la = lalo[:,0]                    # lalo = array(N, 2)
    lo = lalo[:,1]                    # lalo = array(N, 2)
    start_lalo = np.array(start_lalo) # start_lalo = np.array([lat, lon])
    end_lalo   = np.array(end_lalo)   # end_lalo   = np.array([lat, lon])
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]
    u    = (end_lalo-start_lalo)  # u = np.array([lat, lon])
    v    = (lalo-start_lalo)      # v = np.array([lat, lon])
    un   = np.linalg.norm(u)
    vn   = np.linalg.norm(v, axis=1)
    cos  = u.dot(v.T) / (un*vn).flatten()
    dpar = vn*cos                 # distance parallel to line
    new_lalo = start_lalo.reshape(1,2) + (dpar*((u/un).reshape(2,1))).T
    dper = np.linalg.norm(new_lalo-lalo, axis=1) # distance perpendicular to line
    return new_lalo, dpar, dper


def make_segment(start_lalo, end_lalo, seg_length, seg_width):
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]

    dcos = np.diff(lon12) / (np.diff(lon12)**2 + np.diff(lon12)**2)**0.5
    dsin = np.diff(lat12) / (np.diff(lon12)**2 + np.diff(lon12)**2)**0.5
    seg_deg   = kilometers2degrees(seg_length)
    total_deg = locations2degrees(lat12[1], lon12[1], lat12[0], lon12[0])
    nbin      = int(np.round(total_deg/seg_deg))
    print('  number of segments made: ',nbin-1)
    
    lon_seg = np.linspace(lon12[1], lon12[0], nbin)
    lat_seg = np.linspace(lat12[1], lat12[0], nbin)

    width = kilometers2degrees(seg_width)
    lon_west = lon_seg - width*dsin
    lat_west = lat_seg + width*dcos      
    lon_east = lon_seg + width*dsin
    lat_east = lat_seg - width*dcos
    return lon_seg, lat_seg, lon_west, lat_west, lon_east, lat_east, dcos, dsin   


## Calculate seismicity metrics within a certain fault bin
# input: 1) catalog array
#        2) array id of events within bin
#        3) array id of events > Mc
def seis_metric(catalog, bin_id, mc_id, t_id='default', nout=False):
    # get events within a time period
    if t_id=='default':
        t_id=np.ones(len(bin_id), dtype=bool)

    # selections
    select = mc_id * bin_id * t_id
    bevid, bobDT, bpyDT, bdt, blat, blon, bdep, bmag = np.array(catalog).T[select].T
    if nout is True:
        print('Selected seismicity number = {}'.format(np.sum(select)))

    # order events by origin times for interevent time calc
    bevid, bobDT, bpyDT, bdt, blat, blon, bdep, bmag = np.array(catalog).T[select][np.argsort(bdt)].T

    # calculate interevent times & mean seismic rate [num/sec]
    bint = bdt[1:]-bdt[:-1]  # interevent times in each fault bin [sec]

    # Return NaN if only has one or zero event in the bin
    if (len(bdt)<2) or (len(bint)==1):
        return (bevid,bmag,bdt,bint,blat,blon,bdep,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    elif (bdt[-1]-bdt[0]<1e-6):
        return (bevid,bmag,bdt,bint,blat,blon,bdep,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

    ## seismic moment (Sum within 15-km window; N*m)
    moment = np.sum(10**(1.5*(bmag.astype(float))+9.05))

    ## b-value:
    meanMag = np.mean(bmag)
    b = (np.log10(np.exp(1)))/(meanMag - Mc)
    # Bootstrapping b for NN=10000 sets
    NN = 10000
    b_boot = []
    for k in range(NN):
        new_bmag = np.random.choice(bmag, replace=True, size=len(bmag))
        meanMag  = np.mean(new_bmag)
        b_boot.append((np.log10(np.exp(1)))/(meanMag - Mc))
    lb = np.percentile(b_boot,  2.5)    
    ub = np.percentile(b_boot, 97.5)   

    ## Event rate:
    meanRate = len(bdt)/(bdt[-1]-bdt[0])   # mean seismicity rate in each fault bin [num/sec]

    # calculate metrics of interevent times
    int_avg = np.mean(bint)    # sec
    int_std = np.std(bint)     # sec
    int_var = np.var(bint)     # sec^2

    # calculate COV, background rate (BR), mainshock fraction (BF)
    cov = int_std / int_avg    # non-dimension
    BR  = (int_avg/int_var)    # num/sec
    BF  = BR / meanRate        # non-dimension
    BR  = BR * 365.25 * 86400  # num/year

    # Bootstrapping COV, BR, BF for NN sets
    Cov_boot = []
    BR_boot  = []
    BF_boot  = []
    for k in range(NN):
        new_dt = np.array(sorted(np.random.choice(bint, replace=True, size=len(bint))))
        Cov_boot.append(np.std(new_dt)/np.mean(new_dt))
        BR_boot.append(365.25*86400*np.mean(new_dt)/np.var(new_dt))
        BF_boot.append(BR_boot[k]/meanRate/(365.25 * 86400))
    lc = np.percentile(Cov_boot,  2.5)    
    uc = np.percentile(Cov_boot, 97.5)   
    lr = np.percentile(BR_boot,  2.5)    
    ur = np.percentile(BR_boot, 97.5)            
    lf = np.percentile(BF_boot,  2.5)    
    uf = np.percentile(BF_boot, 97.5)        
    return (bevid,bmag,bdt,bint,blat,blon,bdep,cov,BR,BF,lc,uc,lr,ur,lf,uf,b,lb,ub,moment)


## Plot Background seismicity versus creep rate:
def plot_result(fb_mid, data, L, U, x, y, ystr, pjd, ylim=[None,None], curve=False, lc='r', fc='lightpink', titstr='metric'):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios':[1,1.5], 'hspace': 0.18}, sharex=True, figsize=[18,18])
    ax1 = ax[0]
    ax2 = ax1.twinx()
    if curve is False:
        ln1, = ax1.plot(fb_mid, data, '-o', color=lc, lw=3, mfc=lc, mec='k', mew=2, markersize=15)
    elif curve is True:
        ln1, = ax1.plot(fb_mid, data, c=lc, lw=5)
    ax1.fill_between(fb_mid, L, U, fc=fc, ec='grey', ls='--', lw=2, alpha=0.5)
    ln2, = ax2.plot(x, np.abs(y), c='k', lw=5, zorder=0)
    ax1.set_ylabel(ystr, color=lc)
    ax2.set_ylabel('InSAR line-of-sight\ncreep rate [mm/year]', color='k', rotation=270, labelpad=50)
    #ax1.set_xlim(0, xlim[1]-xlim[0])
    ax1.set_ylim(ylim[0], ylim[1])        
    ax2.set_ylim(0,1)
    ax1.xaxis.set_tick_params(which='both', labelbottom=False)
    plt.legend([ln1, ln2], [ystr, 'Creep rate'], loc='upper left', frameon=False)

    ax3 = ax[1]
    from scipy.ndimage import gaussian_filter
    xedges = np.linspace(0, 1900, 190)
    yedges = np.linspace(0, (UTC(endtime)-UTC(starttime))/86400, int((UTC(endtime)-UTC(starttime))/86400/60))

    Dt = np.array(dt)[t_id*sel_m]
    Dt = Dt - Dt[0]

    H, xedges, yedges = np.histogram2d(pjd[t_id*sel_m], np.array(Dt)/86400, bins=(xedges, yedges))
    H = H.T
    print('Seismicity heatmap:')
    print(' Time bin: {:.2f} days'.format(yedges[1]-yedges[0]))
    print(' Space bin: {:.2f} km'.format(xedges[1]-xedges[0]))
    scale = 30/(yedges[1]-yedges[0]) * 1/(xedges[1]-xedges[0])
    H = H * scale
    #H0 = np.mean(H, axis=0)
    #H  = (H-H0)/H0
    Hf = gaussian_filter(H, sigma=[1.5, 1.5])
    yedgdt = []
    for i in range(len(yedges)):
        yedgdt.append((UTC(starttime) + yedges[i]*86400).datetime)
    X, Y = np.meshgrid(xedges, yedgdt)
    im = ax3.pcolormesh(X, Y, np.log10(10*Hf), cmap='Reds', vmin=-1, vmax=1.5)
    m_id = mag>=6
    select = t_id * m_id
    sc = ax3.scatter(pjd[select], np.array(pyDT)[select], s=2.9**mag[select], ec='k', fc='none', lw=1.5)
    
  
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.20, 0.015, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'log(Seismicity intensity) [$\frac{1}{month \times 10km}$]', rotation=270, labelpad=40)
    ax3.set_xlabel('Along-strike distance [km]')
    ax3.set_ylabel('Time [calender year]')
    #ax3.set_xlim(0, xlim[1]-xlim[0])
    ax3.set_ylim(UTC(starttime).datetime, UTC(endtime).datetime)

    #ax1.set_title('{:} ({:})'.format(faultName, titstr), pad=50)
    plt.savefig('{:}/Fig_M3.0_{:}.png'.format(picDir, titstr), bbox_inches='tight', dpi=300)
    plt.rcParams.update({'font.size': 22})  
    plt.show()
    plt.close()



#%%
picDir = './pics'
if not os.path.exists(picDir):
    os.makedirs(picDir)

# %% Read catalog
if False:
    filename = './isc-gem/isc-gem-cat.csv'
    pdcat    = pd.read_csv(filename, header=97)
    c        = pdcat.to_numpy()
    time     = c[:,0]
    lat      = c[:,1].astype('float')
    lon      = c[:,2].astype('float')
    dep      = c[:,7].astype('float')
    mag      = c[:,10].astype('float')
    evid     = c[:,-1]

if False:
    # Read the Servicio Sismologico Nacional (SSN)
    filename = './catalogs/SSNMX_catalogo_19000101_20210510_lat12d14_22d12_lon-109d39_-83d59.csv'
    pdcat    = pd.read_csv(filename, dtype='str', header=12)
    pdcat    = pdcat[pdcat["Profundidad"].str.contains("en")==False]
    pdcat    = pdcat[pdcat["Magnitud"].str.contains("no")==False]
    c        = pdcat.to_numpy()
    lat      = c[:,3].astype('float')
    lon      = c[:,4].astype('float')
    dep      = c[:,5].astype('float')
    mag      = c[:,2].astype('float')

if True:
    subcat = './alaska_catalog/alaska_usgs.csv'
    pdcat  = pd.read_csv(subcat, dtype='str')
    #pdcat  = pdcat[pdcat["Depth"].str.contains("en")==False]
    #pdcat  = pdcat[pdcat["Mag"].str.contains("no")==False]
    c      = pdcat.to_numpy()
    time   = c[:,0]
    lat    = c[:,1].astype('float')
    lon    = c[:,2].astype('float')
    dep    = c[:,3].astype('float')
    mag    = c[:,4].astype('float')
    obDT   = []
    pyDT   = []
    dt     = []
    evid   = c[:,11].astype('str')
    for i in range(len(time)):
        obDT.append(UTC(time[i]))
        pyDT.append(UTC(time[i]).datetime)
        dt.append((UTC(time[i]) - UTC(time[0])))
    cat = (evid, obDT, pyDT, dt, lat, lon, dep, mag)

# Geographic region of interest
#extent = [-108, -89, 11, 21]
#sel_a  = (lon>=extent[0]) & (lon<=extent[1]) & (lat>=extent[2]) & (lat<=extent[3])


#%% plot Mc history
starttime = '19800101'
endtime   = '20210615'
t_id = (np.array(obDT)>=UTC(starttime)) * (np.array(obDT)<UTC(endtime))
print(np.sum(t_id))
Nt = 40
epochs, Ndt, Mcs = epoch_Mc(mag[t_id], np.array(obDT)[t_id], Nt, plot='no')
fc1 = 'r'
fc2 = 'lightskyblue'
bin_day = Ndt/86400

#%%
fig, ax1 = plt.subplots(figsize=[14,14])
ax2 = ax1.twinx()
#ax1.bar(np.array(pydt)[epochs], np.array(Mcs), Ndt/86400, align='edge', fc=fc1, ec='k', lw=1, alpha=0.5)
ax1.scatter(pyDT, mag+np.random.uniform(-0.05, 0.05, len(mag)), marker='o', fc='grey', s=10, zorder=0)
ax1.scatter(np.array(pyDT)[t_id][epochs]+timedelta(days=Ndt/86400/2), np.array(Mcs), s=100, c=fc1, ec='k')
ax1.plot(np.array(pyDT)[t_id][epochs]+timedelta(days=Ndt/86400/2), np.array(Mcs), lw=2, c=fc1 ,zorder=0)
ax2.hist(np.array(pyDT)[t_id], bins=Nt, fc=fc2, ec='k', alpha=0.6)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.text(0.1, 0.8, 'Binning ~{:.0f} day'.format(Ndt/86400), c=fc1, transform=ax2.transAxes)
ax1.text(0.1, 0.9, 'Binning ~{:.0f} day'.format(bin_day), c=fc2, transform=ax2.transAxes)
ax2.set_xlabel('Year')
ax2.set_ylabel('# events', rotation=270)
ax1.set_ylabel('Magnitude of \ncompleteness', color=fc1, labelpad=50)
#ax2.set_ylim(0.7,2.0)
#ax1.set_yticks(np.arange(0,5500,1000))
ax1.set_yticks(np.arange(1,8,0.5))
ax1.grid(True, alpha=0.8)
fig.savefig('{}/McHistory_250k.png'.format(picDir), dpi=300, bbox_inches='tight')


#%%Look at overall magnitude distribution
proposed_Mc = maxc_Mc(mag, plot='yes', title='250k', Mc=None, range=[0,8.5])
print('Guess a Mc = {:.2f}'.format(proposed_Mc)) 

# Events selection
Mc  = 2.5
#dep_id = dep<=50
sel_m = mag >= Mc
sel_t = np.array(obDT)>=UTC('19900101')

proposed_Mc = maxc_Mc(mag[sel_t], plot='yes', title='250k', Mc=2.5, range=[0,8.5])
print('Propose a Mc = {:.2f}'.format(proposed_Mc)) 


#%% point selection

sel_d = (np.array(dep) > 0) & (np.array(dep) < 50)
sel   = sel_t * sel_m * sel_d
slon  = lon[sel]
slat  = lat[sel]
sdep  = dep[sel]
smag  = mag[sel]


# %% Make the regional map
# Specify the approx fault trace
extent      = [-157.368, -144.36, 53.383, 61.889]
start_lalo  = (54.150, -157.017)  # fault trace start 
end_lalo    = (60.349, -142.822)   # fault trace end

plt.rcParams.update({'font.size': 18})
fig  = pygmt.Figure()
grid = pygmt.datasets.load_earth_relief(resolution="30s", region=extent)
fig.grdimage(grid=grid, projection="M15c", frame="a", cmap="geo")
fig.colorbar(
    frame    = ["x+lElevation", "y+lm"],
    position = "JMR+o1.5c/0c+w6c/0.4c",
)
fig.coast(
    resolution = 'high',
    region     = '{}/{}/{}/{}'.format(extent[0], extent[1], extent[2], extent[3]),
    projection = 'M15c',
    #land='lightgray',
    #water='lightblue',
    borders    = '1/0.5p',
    shorelines = ['1/0.5p', '2/0.2p'],
    frame      = 'a'
)
pygmt.makecpt(cmap="magma", series=[0, np.max(sdep)], reverse=True)
fig.plot(
    x     = slon,
    y     = slat,
    sizes = 0.01*(2**smag),
    color = sdep,
    cmap  = True,
    style = 'cc',
    pen   = 'black',
)
fig.colorbar(
    frame    = ["x+lDepth", "y+lkm"],
    position = "n0.06/0.2+w3c/0.3c+h",
)
fig.plot(
    y   = [start_lalo[0], end_lalo[0]],
    x   = [start_lalo[1], end_lalo[1]],
    pen = "4p,red",
)
fig.savefig('{}/alaska_dep.png'.format(picDir), transparent=True, show=True, dpi=300)

# %% Convert seismicity epicenter to projected distance on the approx fault trace
print("There are {} events".format(len(mag)))
new_lalo = pts_projline(np.array([lat,lon]).T, start_lalo, end_lalo)[0]
ev_dist = np.zeros(len(mag))
for i in range(len(mag)):
    ev_dist[i] = geodesic((start_lalo[0], start_lalo[1]), (new_lalo[i,0], new_lalo[i,1])).km

#%% Use a fixed binning
starttime = '19880101'
endtime   = '20210601'
t_id = (np.array(obDT)>=UTC(starttime)) * (np.array(obDT)<UTC(endtime)) * dep_id
print(np.sum(t_id))

bin_size = 100
fb_edge  = np.arange(0,1900,bin_size)
fb_mid   = fb_edge+np.diff(fb_edge)[0]/2
N        = len(fb_mid)

# quickly show number of events histogram
plt.figure(figsize=[14,6])
plt.hist(ev_dist[t_id], bins=fb_edge)
plt.show()

# Get events within each bin
inds = np.digitize(ev_dist, fb_edge)

# Initialize metrics
all_id = []
all_mg = []
all_dt = []
all_int = []
all_la = []
all_lo = []
all_dp = []
COVs   = []
BRs    = []
BFs    = []
LCs    = []
UCs    = []
LRs    = []
URs    = []
LFs    = []
UFs    = []
bvs    = []
Lbs    = []
Ubs    = []
M0s    = []

# Loop bins and calc seismicity metrics
for i in range(N):
    # get array_id of seismicity in each fault bin
    bid = inds==i+1

    # print progress
    print(i, fb_mid[i], np.sum(bid*sel_m*t_id))

    # Run calculation
    metrics = seis_metric(cat, bid, sel_m, t_id)

    # append info seis metrcs for this fault bin
    #     return (bevid,bmag,bdt,bint,blat,blon,bdep,cov,BR,BF,lc,uc,lr,ur,lf,uf,b,lb,ub)
    all_id.append(metrics[0])
    all_mg.append(metrics[1])
    all_dt.append(metrics[2])
    all_int.append(metrics[3])
    all_la.append(metrics[4])
    all_lo.append(metrics[5])
    all_dp.append(metrics[6])
    COVs.append(metrics[7])
    BRs.append(metrics[8])    
    BFs.append(metrics[9])
    LCs.append(metrics[10])
    UCs.append(metrics[11])
    LRs.append(metrics[12])
    URs.append(metrics[13])
    LFs.append(metrics[14])
    UFs.append(metrics[15]) 
    bvs.append(metrics[16]) 
    Lbs.append(metrics[17]) 
    Ubs.append(metrics[18]) 
    M0s.append(metrics[19]) 

print('Fixed binning calculation is completed:\nbin_size = {}'.format(bin_size))



#%% Use a moving binning
bin_size = 160
bin_step = 10
starttime = '20130101'
endtime   = '20210601'
t_id = (np.array(obDT)>=UTC(starttime)) * (np.array(obDT)<UTC(endtime)) * dep_id
N        = int(np.ceil((2 + (1900-bin_size)/bin_step)))
fb_mid   = np.zeros(N)

# Initialize metrics
all_id  = []
all_mg  = []
all_dt  = []
all_int = []
all_la  = []
all_lo  = []
all_dp  = []
COVs    = []
BRs     = []
BFs     = []
LCs     = []
UCs     = []
LRs     = []
URs     = []
LFs     = []
UFs     = []
bvs     = []
Lbs     = []
Ubs     = []
M0s     = []

# Loop bins and calc seismicity metrics
for i in range(N):
    fb_edge = np.array([0+bin_step*i, 0+bin_step*i+bin_size])
    fb_mid[i] = np.mean(fb_edge)

    # Get events within each bin
    inds = np.digitize(ev_dist, fb_edge)

    # get id of array of seismicity in this fault bin
    bid = inds==1
    print(i, fb_edge[0], fb_mid[i], fb_edge[1], np.sum(bid*sel_m*t_id))

    # Run calculation
    metrics = seis_metric(cat, bid, sel_m, t_id)

    # append info seis metrcs for this fault bin
    all_id.append(metrics[0])
    all_mg.append(metrics[1])
    all_dt.append(metrics[2])
    all_int.append(metrics[3])
    all_la.append(metrics[4])
    all_lo.append(metrics[5])
    all_dp.append(metrics[6])
    COVs.append(metrics[7])
    BRs.append(metrics[8])    
    BFs.append(metrics[9])
    LCs.append(metrics[10])
    UCs.append(metrics[11])
    LRs.append(metrics[12])
    URs.append(metrics[13])
    LFs.append(metrics[14])
    UFs.append(metrics[15]) 
    bvs.append(metrics[16])
    Lbs.append(metrics[17])
    Ubs.append(metrics[18])
    M0s.append(metrics[19])
print('Moving window calculation is completed:\nbin_size={}\nbin_step={}'.format(bin_size,bin_step))




# %%
x = np.arange(0,1900)
y = np.ones_like(x)

plot_result(fb_mid,  COVs, LCs, UCs, x, y, 'COV',                 ev_dist, ylim=[0,4],   lc='g',            fc='lightgreen', titstr='cov')
plot_result(fb_mid,  BFs,  LFs, UFs, x, y, 'Background fraction', ev_dist, ylim=[0,1],   lc='r',            fc='lightpink',  titstr='backFr')
plot_result(fb_mid,  BRs,  LRs, URs, x, y, 'Background rate',     ev_dist, ylim=[0,36],  lc='b',            fc='lightblue',  titstr='backRt')
plot_result(fb_mid,  bvs,  Lbs, Ubs, x, y, 'b-value',             ev_dist, ylim=[0.4,2.5], lc='darkorange', fc='bisque',     titstr='bv')


#%% Make plots
plot_result(fb_mid, BFs, LFs, UFs,  x, y, 'Background fraction',    ev_dist, ylim=[0,1],     curve=True, lc='r',          fc='lightpink',  titstr='backFr_mw13')
plot_result(fb_mid, bvs, Lbs, Ubs,  x, y, 'b-value',                ev_dist, ylim=[0.5,None], curve=True, lc='darkorange', fc='bisque',     titstr='bValue_mw13')
plot_result(fb_mid, BRs, LRs, URs,  x, y, 'Background rate [#/yr]', ev_dist, ylim=[0,None],    curve=True, lc='b',          fc='lightblue',  titstr='backRt_mw13')
plot_result(fb_mid, COVs, LCs, UCs, x, y, 'COV',                    ev_dist, ylim=[0,5],     curve=True, lc='g',          fc='lightgreen', titstr='cov_mw13')
plot_result(fb_mid, np.log10(M0s), np.log10(M0s), np.log10(M0s), x, y, r'$log_{10}(M_0)$', ev_dist, ylim=[None,None], curve=True, lc='r', fc='lightpink', titstr='moment13')
plot_result(fb_mid, np.array(BRs)/np.array(BFs), np.array(LRs)/np.array(LFs), np.array(URs)/np.array(UFs), x, y, 'Total rate [#/yr]', ev_dist, curve=True, lc='teal', fc='turquoise', titstr='totalRt_mw13')

# Max Magnitudes The The 
mag_max = []
for i in range(len(fb_mid)):
    mag_max.append(np.max(all_mg[i]))
plot_result(fb_mid, mag_max, mag_max, mag_max, x, y, 'Max magnitude', ev_dist, curve=True, lc='purple', fc='slateblue', titstr='maxmag_mw13')


# %%
plt.figure(figsize=[24,6])
plt.scatter(ev_dist[t_id], dep[t_id])
plt.show()

# %%
num, bin_edges = np.histogram(dep[t_id], np.arange(0,30,1))
centers = bin_edges[:-1] + np.diff(bin_edges)[0]/2
fig = plt.figure(figsize=[20,5])
gs  = fig.add_gridspec(nrows=1, ncols=8)
ax1 = fig.add_subplot(gs[:,:7])
ax2 = fig.add_subplot(gs[:, 7])
sc = ax1.scatter(ev_dist[t_id], dep[t_id], s=10, c=mag[t_id], cmap='plasma_r', vmin=2, vmax=6.5)
ax1.set_xlabel('Along-strike distance [km]')
ax1.set_ylim(30,0)
ax1.set_ylabel('Depth [km]')
ax2.barh(bin_edges[:-1], num, fc='lightgray', ec='k')
ax2.set_ylim(30,0)
ax2.set_yticks([])
ax2.set_xlabel('# events')
plt.colorbar(sc, label='Magnitude')
plt.savefig('{}/dist_depth.png'.format(picDir), bbox_inches='tight', dpi=300)
plt.show()


# %% Make 3D plot of the hypocenters

fig, ax = plt.subplots(figsize=[10,10], subplot_kw={'projection':'3d'})
plt.subplots_adjust(wspace=None, hspace=None)

X =  slon
Y =  slat
Z = -sdep
V =  smag

ax.set_box_aspect((1,1,0.2))
ax.scatter(X, Y, Z, c=-Z, cmap='jet_r', marker='o', alpha=0.6, s=0.003*V**3)

ax.set_xlabel('Lon')
ax.set_ylabel('Lat')
ax.set_zlabel('Depth')

deg=2
for ii in range(0,360,deg):
    ax.view_init(elev=22, azim=ii)
    fig.savefig("movie%03d.png" % int(ii//deg))
# %%
