#!/usr/bin/env python3
# --------------------------------------------------------------
# Post-processing for MintPy output results
#
# Yuan-Kai Liu, 2022-3-3
# --------------------------------------------------------------

# Recommended usage:
#   import sarut.tools.data as sardata

import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from tqdm import tqdm
from scipy import linalg
from scipy.interpolate import RectBivariateSpline
from matplotlib import cm


from scipy import linalg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
from mintpy.utils import (
    readfile,
    writefile
)


matplotlib.rcParams.update({'font.size': 16})


#########################################################
## Define functions (from Radar Imaging class)

def readbinary(file, nsamp, nline, dtype):
    with open(file, 'rb') as fn:
        load_arr = np.frombuffer(fn.read(), dtype=dtype)
        load_arr = load_arr.reshape((nline, nsamp))
    return np.array(load_arr)


def plot_img(data, nhdr=0, title='Data', scale=1, cmap='gray', vlim=None, orglim=None, origin='upper', aspect="equal", xlabel='Range [bins]', ylabel='Azimuth [lines]', clabel='Value [-]', interpolation='none', savetif=None, figsize=[6,6], lim=[None,None,None,None], ex=None, yticks=None):
    if scale > 1:
        clabel = '{} * {}'.format(clabel, scale)
    else:
        clabel = '{}'.format(clabel)

    # Adjust the data header part for better visualization
    val = np.array(data)
    val[:,nhdr:] = scale * val[:,nhdr:]

    if vlim is None:
        vlim = [None, None]

    # original value limit (for overlap image coloring output which varies from 0~1)
    if orglim is not None:
        if vlim[0] is not None:
            vlim[0] = (vlim[0]-orglim[0])/np.diff(orglim)[0]
        else:
            vlim[0] = 0
        if vlim[1] is not None:
            vlim[1] = (vlim[1]-orglim[0])/np.diff(orglim)[0]
        else:
            vlim[1] = 1
        cticks      = np.linspace(  vlim[0],   vlim[1], 4)
        cticklabels = np.linspace(orglim[0], orglim[1], 4)

    # plot the 2D image
    plt.figure(figsize=figsize)
    im   = plt.imshow(val, cmap=cmap, interpolation=interpolation, vmin=vlim[0], vmax=vlim[1], origin=origin, aspect=aspect, extent=ex)
    cbar = plt.colorbar(im, shrink=0.4, pad=0.06)
    cbar.set_label(clabel, rotation=270, labelpad=30)
    if orglim is not None:
        cbar.set_ticks(cticks)
        cbar.ax.set_yticklabels('{:.2f}'.format(x) for x in cticklabels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    if (yticks is not None) and (len(yticks)==3):
        plt.yticks(np.linspace(lim[2], lim[3], yticks[2]), np.linspace(yticks[0], yticks[1], yticks[2]))
    if savetif is not None:
        plt.savefig('{}'.format(savetif), dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    return


def makechirp(N, slope, tau, fs, fc=0, start=0, phi0=0):
    """ Make a reference chirp pulse
    N:     Num of points of the whole pulse   [#]
    slope: slope of the chirp                 [Hz/s]
    tau:   chirp length                       [s]
    fs:    sample rate                        [Hz]
    fc:    central carrier freq               [Hz]
    start: starting sample # of the chirp     [#]
    """
    dt    = 1/fs                                        # sampling time interval          [s]
    npts  = tau * fs                                    # num of points of the pure chirp [#]
    t     = dt * np.arange(-npts/2, npts/2)             # time axis of the pure chirp     [s]
    phase = np.pi*slope*(t**2) + 2*np.pi*fc*t + phi0    # chirp phase                     [rad]
    chirp = np.exp(1j*phase)                            # chirp                           (cmplx)
    chirp = np.pad(chirp, (start,N-len(chirp)-start))   # pad zeros at tail and beginning (cmplx)
    t_arr = dt * np.arange(0, N)                        # time axis of the whole pulse    [s]
    #print('Chirp starts from {} samples, {} mu s'.format(start, 1e6*(start/fs)))
    return chirp, t_arr


def cross_corr(sig1, sig2, axis=-1):
    sig1_fft = np.fft.fft(sig1, axis=axis)                      # transform the signal_1 to freq domain
    sig2_fft = np.fft.fft(sig2, axis=axis)                      # transform the signal_2 to freq domain
    spec    = sig1_fft * np.conjugate(sig2_fft)                 # cross-correlation gives the spectrum
    comp    = np.fft.ifft(spec, axis=axis)                      # inverse transform it back to time domain
    return comp, spec


def cross_corr2d(sig1, sig2):
    # pad sig1 if smaller than sig2
    pady = (sig2.shape[0]-sig1.shape[0]) // 2
    padx = (sig2.shape[1]-sig1.shape[1]) // 2
    sig1 = np.pad(sig1, ((pady,pady),(padx,padx)))
    sig1_fft = np.fft.fft2(sig1)                       # transform the signal_1 to freq domain
    sig2_fft = np.fft.fft2(sig2)                       # transform the signal_2 to freq domain
    spec     = sig1_fft * np.conjugate(sig2_fft)       # cross-correlation gives the spectrum
    comp     = np.fft.fftshift(np.fft.ifft2(spec))     # inverse transform it back to time domain
    if pady!=0:
        comp = comp[pady:-pady, :]
    if padx!=0:
        comp = comp[:, padx:-padx]
    return comp


def plot_freq(freq, val, title, x='Frequency', y='20*log10(|spectrum|), [dB]', xlim=[None,None], ylim=[None,None], unit='MHz', shift=False, axvl=None):
    x += ' [{}]'.format(unit)
    if unit == 'MHz':
        u = 1e-6
    elif unit == 'Hz':
        u = 1
    if shift:
        val = np.fft.fftshift(val)
    plt.figure(figsize=[14,4])
    plt.plot(freq*u, val)
    plt.title(title)
    plt.xlim(min(freq)*u, max(freq)*u)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(x)
    plt.ylabel(y)
    if axvl is not None:
        for vl in axvl:
            plt.axvline(x=vl, c='k', ls='--')
    plt.show()


def plot_time(t, val, title, x=r'Time', y='amplitude [-]', xlim=[None,None], ylim=[None,None], unit='micros', shift=False, plotcomplex=False, axvl=None):
    if unit == 'micros':
        x += r' [$\mu$ s]'
        u = 1e6
    elif unit == 's':
        x += r' [s]'
        u = 1
    if shift:
        val = np.fft.fftshift(val)
    plt.figure(figsize=[14,4])
    if plotcomplex:
        plt.plot(t*u, np.real(val), label='Real part')
        plt.plot(t*u, np.imag(val), label='Imaginary part')
        plt.legend(loc='upper right')
    else:
        plt.plot(t*u, val)
    plt.title(title)
    plt.xlim(min(t)*u, max(t)*u)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(x)
    plt.ylabel(y)
    if axvl is not None:
        for vl in axvl:
            plt.axvline(x=vl, c='k')
    plt.show()


def magdB(val):
    mag = 20 * np.log10(np.abs(val) + 1e-30)
    return mag


def doppler_phase_shift(R, PRF):
    Naz, Nr = R.shape
    psRg = np.zeros(Nr, dtype=np.complex128)
    for i in np.arange(2,Naz):
        psRg += R[i,:] * np.conjugate(R[i-1,:])
    # compute phase shift at each range bin
    phi = np.arctan(np.imag(psRg)/np.real(psRg))
    phi_avg = np.mean(phi)
    fd = PRF * phi_avg/(2*np.pi)
    return fd, phi_avg


def doppler_cent_squint(fd, v, theta, wavelength):
    sine_squint = fd * wavelength * 0.5 * v / np.sin(theta)
    squint = np.arcsin(sine_squint)
    return squint


def roll_and_pad(x, shift):
    y = np.roll(x,-shift)
    if shift <= 0:
        y[:-shift] = 0
    elif shift > 0:
        y[-shift:] = 0
    return y


def multilook(data, alk=1, rlk=1):
    bin_azimuth = int(data.shape[0]/alk)
    bin_range   = int(data.shape[1]/rlk)
    data_azlook = np.zeros([bin_azimuth, data.shape[1]], dtype=np.complex128)
    for i in range(bin_azimuth):
        data_azlook[i,:] = np.mean(data[i*alk:(i+1)*alk, :], axis=0)
    data_out = np.zeros([bin_azimuth, bin_range], dtype=np.complex128)
    for j in range(bin_range):
        data_out[:,j] = np.mean(data_azlook[:, j*rlk:(j+1)*rlk], axis=1)
    return data_out


def interpCmpl(data, y_grid, x_grid):
    # Resample/interpolate complex data
    # INPUT
    #       data:       input image
    #       y_grid:     interpolate on y_grid
    #       x_grid:     interpolate on x_grid
    # OUTPUT
    #       data_new:   resampled/interpolated data

    # decompose the complex array to mag and phase
    mag = np.abs(data)
    phi = np.angle(data)

    # Interpolate the mag and phi separately
    xmax, ymax = data.shape[1], data.shape[0]
    spline_mag = RectBivariateSpline(np.arange(0, ymax, 1), np.arange(0, xmax, 1), mag)
    spline_phi = RectBivariateSpline(np.arange(0, ymax, 1), np.arange(0, xmax, 1), phi)
    mag_new    = spline_mag(y_grid, x_grid)
    phi_new    = spline_phi(y_grid, x_grid)
    data_new   = mag_new * np.exp(1j * phi_new)
    return data_new


def overlay_img(amp, phi, cmap='RdYlBu'):
    # For visualization purpose
    # Overlay phase with the amplitude (adjusting the brightness of the cmap)
    amp_hi = np.percentile(amp, 95)
    amp_lo = np.percentile(amp,  5)
    phi_hi = np.percentile(phi, 99)
    phi_lo = np.percentile(phi,  1)

    amp       = np.clip(amp, amp_lo, amp_hi)
    amp_scale = (amp-amp_lo) / (amp_hi-amp_lo)
    phi_scale = (phi - phi_lo) / (phi_hi - phi_lo)

    # cmap_func is a function which returns (rgba) colors base on a range 0-1.
    # Therefore, it transforms your values to 0-1 to use them as input for cmap()
    nbin = 512          # bins of color
    cmap_func = cm.get_cmap(cmap, nbin)

    # initialize a overlayed image (4 channels, RGB and alpha)
    img = cmap_func((nbin*phi_scale).astype(int))

    # scale the RGB columns (brightness) with the amplitude
    img[:,:,:3] *= amp_scale.reshape(amp.shape[0] ,amp.shape[1], -1)
    return img


def compute_cc(img1, img2, alk=4, rlk=4):
    ifg  = img1 * np.conj(img2)
    ifg  = multilook(ifg, alk=alk, rlk=rlk)
    pow1 = img1 * np.conj(img1)
    pow1 = multilook(pow1, alk=alk, rlk=rlk)
    pow2 = img2 * np.conj(img2)
    pow2 = multilook(pow2, alk=alk, rlk=rlk)
    cc   = np.abs(ifg) / (pow1**0.5 * pow2**0.5 + 1e-30)
    return np.abs(cc)


def lstsq_offset(d, x, n=3):
    # INPUT
    #       d:      data y-axis
    #       x:      data x-axis
    #       n:      the threshold to exclude the outliers
    # OUTPUT
    #       avg:    data mean after excluding the outliers
    #       good:   non-outliers index
    #       m:      linear fit parameters (slope and intercept)

    # remove outliers by MAD (median absolute deviation)
    med   = np.median(d)             # compute the median of the data
    mad   = np.median(np.abs(d-med)) # compute the MAD
    dev   = np.abs(d-med)            # data's deviation from median
    thres = n*mad                    # set threshold for excluding the outliers (outliers are those values > meadian ± n*MAD)
    good  = dev<=thres               # get the good index (non-outliers)
    d     = d[good]                  # the good data
    avg   = np.mean(d)               # estimate the mean of the good data

    # do a least-squares linear model fit to the data
    x = x[good]                                              # x-axis of the good data
    A = np.concatenate((x, np.ones_like(x))).reshape(2,-1).T # get the design matrix for a line Am = d (m is the parameters for a line)
    m, e2 = linalg.lstsq(A, d)[:2]                           # compute least-squares solution to equation Am = d (m[1] is the intercept, m[0] is the slope)
    return avg, good, m


## Plot the elevation with contours
def plot_DEM(dem, level, n, dimscale=[1,1], dimunit='pixels', title='DEM', figsize=10):
    # INPUT
    #       dem:        DEM array
    #       level:      number of DEM contour lines
    #       n:          DEM contour resolution (every n pixel)
    #       dimscale:   scale the x and y axis (based on ground pixel spacing; optional, default=1)
    #       dimunit:    unit of x and y axis (default = pixels)
    #       title:      figure title
    xx, yy   = np.meshgrid(dimscale[1]*np.arange(dem.shape[1]), dimscale[0]*np.arange(dem.shape[0]))
    peak_dem = np.nanmax(dem)                                                     # find the altitude peak
    peak     = dimscale * np.array(np.where(dem==peak_dem)).flatten()   # peak location

    plt.figure(figsize=[figsize,figsize])
    im = plt.imshow(dem, cmap='terrain', origin='lower', vmin=0, extent=[0, np.max(xx), 0, np.max(yy)])
    contours = plt.contour(xx[::n,::n], yy[::n,::n], dem[::n,::n], level, colors='k', vmin=0, linewidths=0.5)
    plt.clabel(contours, inline=True, fontsize=12,  fmt='%.0f')
    cbar = plt.colorbar(im, shrink=0.4, pad=0.04)
    cbar.set_label('elevation [m]', rotation=270, labelpad=30)
    plt.scatter(peak[1], peak[0], marker='^', lw=1, c='r', s=150)
    plt.text(peak[1], peak[0], '{:.0f} m'.format(peak_dem), fontsize=14)
    plt.title(title)
    plt.xlabel('Range direction [{}]'.format(dimunit))
    plt.ylabel('Azimuth direction [{}]'.format(dimunit))
    plt.show()

    print('Highest altitude: {:.1f} m'.format(peak_dem))





###############################################################
## Define functions (from UNAVCO InSAR class)

if False:
    # directory in which the notebook resides
    if 'tutorial_home_dir' not in globals():
        tutorial_home_dir = os.getcwd()
    print("Notebook directory: ", tutorial_home_dir)

    # directory for data downloads
    slc_dir = os.path.join(tutorial_home_dir,'data', 'slcs')
    orbit_dir = os.path.join(tutorial_home_dir, 'data', 'orbits')
    insar_dir = os.path.join(tutorial_home_dir, 'insar')


    # defining backup dirs in case of download issues on the local server
    s3 = boto3.resource("s3")
    data_backup_bucket = s3.Bucket("asf-jupyter-data")
    data_backup_dir = "TOPS"

    # generate all the folders in case they do not exist yet
    os.makedirs(slc_dir, exist_ok=True)
    os.makedirs(orbit_dir, exist_ok=True)
    os.makedirs(insar_dir, exist_ok=True)

    # Always start at the notebook directory
    os.chdir(tutorial_home_dir)


    # Utility to copy data from
    def copy_from_bucket(file_in_bucket, dest_file,
                        bucket=data_backup_bucket):
        if os.path.exists(dest_file):
            print("Destination file {0} already exists. Skipping download...".format(dest_file))
        else:
            bucket.download_file(file_in_bucket, dest_file)

# Utility to plot a 2D array
def plotdata(GDALfilename, band=1,
             title=None,colormap='gray',
             aspect=1, background=None,
             datamin=None, datamax=None,
             interpolation='nearest',
             nodata = None,
             phase2disp = False,
             wrapdisp = False,
             draw_colorbar=True, colorbar_orientation="horizontal"):
    plt.rcParams.update({'font.size': 20})
    # Read the data into an array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(band).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None

    try:
        if nodata is not None:
            data[data == nodata] = np.nan
    except:
        pass

    # getting the min max of the axes
    firstx = transform[0]
    firsty = transform[3]
    deltay = transform[5]
    deltax = transform[1]
    lastx = firstx+data.shape[1]*deltax
    lasty = firsty+data.shape[0]*deltay
    ymin = np.min([lasty,firsty])
    ymax = np.max([lasty,firsty])
    xmin = np.min([lastx,firstx])
    xmax = np.max([lastx,firstx])

    # put all zero values to nan and do not plot nan
    if background is None:
        try:
            data[data==0]=np.nan
        except:
            pass


    # convert phase to displacement (if phase file is given)
    # note: Sentinel-1 wavlength = 0.05546576 meter
    if phase2disp == True:
        data = data * 0.05546576/(4*np.pi)

    if wrapdisp == True:
        data = (5.55/4/np.pi)*((data*100*4*np.pi/5.55)%(2*np.pi))


    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    cax = ax.imshow(data, vmin = datamin, vmax=datamax,
                    cmap=colormap, extent=[xmin,xmax,ymin,ymax],
                    interpolation=interpolation)
    ax.set_title(title)
    if draw_colorbar is not None:
        cbar = fig.colorbar(cax,orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show()

    # clearing the data
    data = None

# Utility to plot interferograms
def plotcomplexdata(GDALfilename,
                    title=None, aspect=1,
                    datamin=None, datamax=None,
                    interpolation='nearest',
                    draw_colorbar=None, colorbar_orientation="horizontal"):
    # Load the data into numpy array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    slc = ds.GetRasterBand(1).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None

    # getting the min max of the axes
    firstx = transform[0]
    firsty = transform[3]
    deltay = transform[5]
    deltax = transform[1]
    lastx = firstx+slc.shape[1]*deltax
    lasty = firsty+slc.shape[0]*deltay
    ymin = np.min([lasty,firsty])
    ymax = np.max([lasty,firsty])
    xmin = np.min([lastx,firstx])
    xmax = np.max([lastx,firstx])

    # put all zero values to nan and do not plot nan
    try:
        slc[slc==0]=np.nan
    except:
        pass


    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(1,2,1)
    cax1=ax.imshow(np.abs(slc), vmin = datamin, vmax=datamax,
                   cmap='gray', extent=[xmin,xmax,ymin,ymax],
                   interpolation=interpolation)
    ax.set_title(title + " (amplitude)")
    if draw_colorbar is not None:
        cbar1 = fig.colorbar(cax1,orientation=colorbar_orientation)
    ax.set_aspect(aspect)

    ax = fig.add_subplot(1,2,2)
    cax2 =ax.imshow(np.angle(slc), cmap='RdYlBu',
                    vmin=-np.pi, vmax=np.pi,
                    extent=[xmin,xmax,ymin,ymax],
                    interpolation=interpolation)
    ax.set_title(title + " (phase [rad])")
    if draw_colorbar is not None:
        cbar2 = fig.colorbar(cax2, orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show()

    # clearing the data
    #slc = None
    return slc

# Utility to plot multiple similar arrays
def plotstackdata(GDALfilename_wildcard, band=1,
                  title=None, colormap='gray',
                  aspect=1, datamin=None, datamax=None,
                  interpolation='nearest',
                  draw_colorbar=True, colorbar_orientation="horizontal"):
    # get a list of all files matching the filename wildcard criteria
    GDALfilenames = glob.glob(GDALfilename_wildcard)

    # initialize empty numpy array
    data = None
    for GDALfilename in GDALfilenames:
        ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
        data_temp = ds.GetRasterBand(band).ReadAsArray()
        ds = None

        if data is None:
            data = data_temp
        else:
            data = np.vstack((data,data_temp))

    # put all zero values to nan and do not plot nan
    try:
        data[data==0]=np.nan
    except:
        pass

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    cax = ax.imshow(data, vmin = datamin, vmax=datamax,
                    cmap=colormap, interpolation=interpolation)
    ax.set_title(title)
    if draw_colorbar is not None:
        cbar = fig.colorbar(cax,orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show()

    # clearing the data
    data = None

# Utility to plot multiple simple complex arrays
def plotstackcomplexdata(GDALfilename_wildcard,
                         title=None, aspect=1,
                         datamin=None, datamax=None,
                         interpolation='nearest',
                         draw_colorbar=True, colorbar_orientation="horizontal"):
    # get a list of all files matching the filename wildcard criteria
    GDALfilenames = glob.glob(GDALfilename_wildcard)
    print(GDALfilenames)
    # initialize empty numpy array
    data = None
    for GDALfilename in GDALfilenames:
        ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
        data_temp = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        if data is None:
            data = data_temp
        else:
            data = np.vstack((data,data_temp))

    # put all zero values to nan and do not plot nan
    try:
        data[data==0]=np.nan
    except:
        pass

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(1,2,1)
    cax1=ax.imshow(np.abs(data), vmin=datamin, vmax=datamax,
                   cmap='gray', interpolation='nearest')
    ax.set_title(title + " (amplitude)")
    if draw_colorbar is not None:
        cbar1 = fig.colorbar(cax1,orientation=colorbar_orientation)
    ax.set_aspect(aspect)

    ax = fig.add_subplot(1,2,2)
    cax2 =ax.imshow(np.angle(data), cmap='RdYlBu',
                            interpolation='nearest')
    ax.set_title(title + " (phase [rad])")
    if draw_colorbar is not None:
        cbar2 = fig.colorbar(cax2,orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show()

    # clearing the data
    data = None


#####################################
# Below are for post-processing mintpy results


############## New functions for plotting nicer velocity deramp/iono/scaling #############
def read_mask(mask_file):
    mask_data = readfile.read(mask_file)[0]
    mask = mask_data==1
    return 1*mask

def read_img(fname, mask):
    # The dataset unit is meter
    v     = readfile.read(fname, datasetName='velocity')[0]*1000     # Unit: mm/y
    #meta  = readfile.read(fname, datasetName='velocity')[1]          # metadata
    #v_std = readfile.read(fname, datasetName='velocityStd')[0] *1000  # Unit: mm/y

    # read mask and mask the dataset
    mask_file = mask   # 'waterMask.h5' or 'maskTempCoh.h5'
    mask_data = readfile.read(mask_file)[0]
    v[mask_data==0] = np.nan
    #v_std[mask_data==0] = np.nan
    #water_mask = readfile.read('../../waterMask.h5')[0]
    return v


def read_stack(fname, dset, mask):
    # metadata
    dsname4atr = None   #used to determine UNIT
    if isinstance(dset, list):
        dsname4atr = dset[0].split('-')[0]
    elif isinstance(dset, str):
        dsname4atr = dset.split('-')[0]
    meta = readfile.read_attribute(fname, datasetName=dsname4atr)

    length = int(meta['LENGTH'])
    width  = int(meta['WIDTH'])

    v      = readfile.read_hdf5_file(fname, datasetName=dset, box=[0,0,width,length])*1000

    # read mask and mask the dataset
    mask_file = mask   # 'waterMask.h5' or 'maskTempCoh.h5'
    mask_data = readfile.read(mask_file)[0]
    v[mask_data==0] = np.nan
    return v


def est_ramp(data, ramp_type='linear', mask='none'):
    width, length = data.shape
    # design matrix
    xx, yy = np.meshgrid(np.arange(0, width),
                         np.arange(0, length))
    xx = np.array(xx, dtype=np.float32).reshape(-1, 1)
    yy = np.array(yy, dtype=np.float32).reshape(-1, 1)
    ones = np.ones(xx.shape, dtype=np.float32)

    if ramp_type == 'linear':
        G = np.hstack((yy, xx, ones))
    elif ramp_type == 'quadratic':
        G = np.hstack((yy**2, xx**2, yy*xx, yy, xx, ones))
    elif ramp_type == 'linear_range':
        G = np.hstack((xx, ones))
    elif ramp_type == 'linear_azimuth':
        G = np.hstack((yy, ones))
    elif ramp_type == 'quadratic_range':
        G = np.hstack((xx**2, xx, ones))
    elif ramp_type == 'quadratic_azimuth':
        G = np.hstack((yy**2, yy, ones))
    else:
        raise ValueError('un-recognized ramp type: {}'.format(ramp_type))

    # estimate ramp
    mask = mask.flatten()
    X = np.dot(np.linalg.pinv(G[mask, :], rcond=1e-15), data[mask, :])
    ramp = np.dot(G, X)
    ramp = np.array(ramp, dtype=data.dtype)

    data_out = data - ramp
    return data_out, ramp

def linear_fit(x, y):
    # Create an instance of a linear regression model and fit it to the data with the fit() function:
    model = LinearRegression().fit(x, y)
    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('slope:', model.coef_[0])
    print('intercept:', model.intercept_)
    y_pred = model.predict(x)
    return model, y_pred

def flatten_isnotnan(x):
    x = x.flatten()[~np.isnan(x.flatten())]
    return x

def dem_shading(dem, shade_azdeg=315, shade_altdeg=45, shade_exag=0.5, shade_min=-2e3, shade_max=3e3):
    # prepare shade relief
    import warnings
    from matplotlib.colors import LightSource
    from mintpy.objects.colors import ColormapExt

    ls = LightSource(azdeg=shade_azdeg, altdeg=shade_altdeg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dem_shade = ls.shade(dem, vert_exag=shade_exag, cmap=ColormapExt('gray').colormap, vmin=shade_min, vmax=shade_max)
    dem_shade[np.isnan(dem_shade[:, :, 0])] = np.nan
    return dem_shade


def plot_imgs(v, meta, dem=None, vlims=[[None, None]], bbox=[None]*4, unit='mm/yr', cmap='RdYlBu_r', picdir='./pic', outf='img001'):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    font_size=16
    plt.rcParams.update({'font.size': font_size})

    ## get attributes
    length    = int(meta['LENGTH'])
    width     = int(meta['WIDTH'])
    x_min     = float(meta['X_FIRST'])
    x_step    = float(meta['X_STEP'])
    y_min     = float(meta['Y_FIRST'])
    y_step    = float(meta['Y_STEP'])
    lats      = np.arange(y_min,length*y_step+y_min, y_step)
    lons      = np.arange(x_min, width*x_step+x_min, x_step)
    ref_lat   = float(meta['REF_LAT'])
    ref_lon   = float(meta['REF_LON'])
    geo_extent= [lons[0],lons[-1],lats[-1],lats[0]]

    nrows, ncols = 1, len(v)
    keys = v.keys()

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*8,12], sharey=True, gridspec_kw={'wspace':0.02})
    if len(v) == 1:
        axs = [axs]
    if len(vlims) != len(axs):
        vlims.append([None,None]*(len(axs)-len(vlims)))

    for i, (ax, k, vlim) in enumerate(zip(axs, keys, vlims)):
        # plot DEM and the image
        if dem is not None:
            ie = ax.imshow(dem, extent=geo_extent, vmin=-500, vmax=2000)
        im = ax.imshow(v[k],    extent=geo_extent, cmap=cmap, vmin=vlim[0], vmax=vlim[1], alpha=0.6)

        # colorbar
        high_val = np.nanpercentile(v[k], 99.8)
        low_val  = np.nanpercentile(v[k],  0.2)
        cbck = inset_axes(ax, width="60%", height="7.5%", loc='lower left',
                bbox_transform=ax.transAxes, bbox_to_anchor=(-0.02,-0.015,1,1))    # colorbar background
        cbck.set_facecolor('w')
        cbck.patch.set_alpha(0.7)
        cbck.get_xaxis().set_visible(False)
        cbck.get_yaxis().set_visible(False)
        cbar = inset_axes(cbck, width="90%", height="45%",loc='upper center',
                bbox_transform=cbck.transAxes, bbox_to_anchor=(0, 0.1, 1, 1))
        fig.colorbar(im, cax=cbar, orientation='horizontal')
        cbar.text(0.5, 0.5, unit, ha='center', va='center', fontsize=16, transform=cbar.transAxes)
        cbar.text(0.02, 0.5, '({:.1f})'.format(low_val),   ha='left', va='center', fontsize=12, transform=cbar.transAxes)
        cbar.text(0.98, 0.5, '({:.1f})'.format(high_val), ha='right', va='center', fontsize=12, transform=cbar.transAxes)

        # scale bar & xlabel
        if i == 0:
            if not all(x is None for x in bbox):
                cent_lat = np.mean(bbox[2:])
                span_lon = bbox[1]-bbox[0]
            else:
                cent_lat = np.mean(lats)
                span_lon = np.max(lons)-np.min(lons)
            r_earth    = 6378.1370
            km_per_lon = np.pi/180 * r_earth * np.cos(np.pi*cent_lat/180)
            span_km    = span_lon * km_per_lon
            scal_km    = round(span_km/3/10)*10
            scal_lon   = scal_km / km_per_lon
            scax  = inset_axes(ax, width=scal_lon, height="1%", loc='upper left',
                    bbox_transform=ax.transAxes, bbox_to_anchor=(0.05, 0.05, 1, 1))
            scax.set_facecolor('k')
            scax.axes.xaxis.set_ticks([])
            scax.axes.yaxis.set_ticks([])
            scax.set_xlabel('{:d} km'.format(scal_km), fontsize=16, labelpad=2)
            ax.set_ylabel('Latitude', fontsize=font_size+4)

        # reference point if available
        if ref_lon and ref_lat:
            ax.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')

        # others
        ax.set_title(k)
        ax.set_xlabel('Longitude', fontsize=font_size+4)
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])

    # output
    if not os.path.exists(picdir):
        os.makedirs(picdir)
    out_file = f'{picdir}/{outf}.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=300)
    print('save to file: '+out_file)
    plt.show()


def scatter_fields(data1, data2, labels=['data1','data2'], vlim=[-20,20], title='', savedir=False):
    ## linear fit to the trend
    x = flatten_isnotnan(data1)
    y = flatten_isnotnan(data2)
    model, y_pred = linear_fit(x.reshape(-1, 1), y)

    # plot
    plt.figure(figsize=[6,6])
    plt.scatter(x, y)
    plt.scatter(x, y_pred, s=0.3, label='y=ax+b \n a={:.3f}, b={:.3f}'.format(model.coef_[0], model.intercept_))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(vlim[0], vlim[1])
    plt.legend(loc='upper left')
    plt.title(title)
    if savedir is not False:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # output
        out_file = f'{savedir}/{title}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=300)
        print('save to file: '+out_file)
        plt.close()
    else:
        plt.show()
