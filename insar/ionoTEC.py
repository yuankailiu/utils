#!/usr/bin/env python3
# Author: Zhang Yunjun, Heresh Fattahi, Feb 2020
# Useful links:
#   IGS (NASA): https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html
#   IMPC (DLR): https://impc.dlr.de/products/total-electron-content/near-real-time-tec/nrt-tec-global/
# Contents
#   Utils
#   IGS TEC I/O
#   Test
# Recommend usage:
#   from *module*.simulation import iono


import os
import sys
import glob
import re
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from scipy import interpolate

from mintpy.utils import readfile, utils as ut


# global variables
SPEED_OF_LIGHT = 299792458 # m/s
EARTH_RADIUS = 6371e3      # Earth radius in meters
K = 40.31                  # m^3/s^2, constant

SAR_BAND = {
    'L' : 1.270e9,  #NISAR-L, ALOS-2
    'S' : 3.2e9,    #NISAR-S
    'C' : 5.405e9,  #Sentinel-1
    'X' : 9.65e9,   #TerraSAR-X
}


######################################## Utilities ###################################

def vtec2range_delay(vtec, inc_angle, freq, obs_type='phase'):
    """Calculate/predict the range delay in SAR from TEC in zenith direction.

    Parameters: vtec      - float, zenith TEC in TECU
                inc_angle - float/np.ndarray, incidence angle at the ionospheric shell in deg
                freq      - float, radar carrier frequency in Hz.
                
    Returns:    rg_delay  - float/np.ndarray, predicted range delay in meters
    """
    # ignore no-data value in inc_angle
    if type(inc_angle) is np.ndarray:
        inc_angle[inc_angle == 0] = np.nan

    # convert to TEC in LOS based on equation (3) in Chen and Zebker (2012)
    tec = vtec / np.cos(inc_angle * np.pi / 180.0)

    # calculate range delay based on equation (1) in Chen and Zebker (2012)
    range_delay = (tec * 1e16 * K / (freq**2)).astype(np.float32)

    # group delay = phase advance * -1
    if obs_type != 'phase':
        range_delay *= -1.

    return range_delay


def prep_geometry_iono_shell_along_los(geom_file, box=None, iono_height=450e3, print_msg=True):
    """Prepare geometry info of LOS vector at thin-shell ionosphere.
    Parameters: geom_file      - str, path to the geometry file in HDF5/MintPy format
                box            - tuple of 4 int, box of interest in (x0, y0, x1, y1)
                iono_height    - float, height of the assume effective thin-shell ionosphere in m
    Returns:    iono_inc_angle - 2D np.ndarray / float, incidence angle in degree
                iono_lat/lon   - float, latitude/longitude of LOS vector at the thin-shell in deg
                iono_height    - float, height of the assume effective thin-shell ionosphere in m
    """
    # inc_angle on the ground
    inc_angle, meta = readfile.read(geom_file, datasetName='incidenceAngle', box=box)
    inc_angle = np.squeeze(inc_angle)
    inc_angle[inc_angle == 0] = np.nan
    inc_angle_center = np.nanmean(inc_angle)
    if print_msg:
        print('incidence angle on the ground     min/max: {:.1f}/{:.1f} deg'.format(np.nanmin(inc_angle),
                                                                                    np.nanmax(inc_angle)))

    # inc_angle on the thin-shell ionosphere
    iono_inc_angle = incidence_angle_ground2iono_shell_along_los(inc_angle)
    if print_msg:
        print('incidence angle on the ionosphere min/max: {:.1f}/{:.1f} deg'.format(np.nanmin(iono_inc_angle),
                                                                                    np.nanmax(iono_inc_angle)))

    # center lat/lon on the ground & thin-shell ionosphere
    lat, lon = ut.get_center_lat_lon(geom_file, box=box)
    iono_lat, iono_lon = lalo_ground2iono_shell_along_los(lat, lon,
                                                          inc_angle=inc_angle_center,
                                                          head_angle=float(meta['HEADING']))

    if print_msg:
        print('center lat/lon  on the ground    : {:.4f}/{:.4f} deg'.format(lat, lon))
        print('center lat/lon  on the ionosphere: {:.4f}/{:.4f} deg'.format(iono_lat, iono_lon))

    return iono_inc_angle, iono_lat, iono_lon, iono_height


def incidence_angle_ground2iono_shell_along_los(inc_angle, iono_height=450e3):
    """Calibrate the incidence angle of LOS vector on the ground surface to the ionosphere shell
    based on equation (6) in Chen ang Zebker (2012, TGRS)

    Reference: Jingyi, C., and H. A. Zebker (2012), Ionospheric Artifacts in Simultaneous
        L-Band InSAR and GPS Observations, Geoscience and Remote Sensing, IEEE Transactions on,
        50(4), 1227-1239, doi:10.1109/TGRS.2011.2164805.

    Parameters: inc_angle      - float/np.ndarray, incidence angle on the ground in degrees
                iono_height    - float, effective ionosphere height in meters
                                 under the thin-shell assumption
    Returns:    inc_angle_iono - float/np.ndarray, incidence angle on the iono shell in degrees
    """
    # ignore nodata in inc_angle
    if type(inc_angle) is np.ndarray:
        inc_angle[inc_angle == 0] = np.nan

    # deg -> rad & copy to avoid changing input variable
    inc_angle = np.array(inc_angle) * np.pi / 180

    # calculation
    cos_inc_angle_iono = np.sqrt(1 - (EARTH_RADIUS * np.sin(inc_angle) / (EARTH_RADIUS + iono_height))**2)
    inc_angle_iono = np.arccos(cos_inc_angle_iono)
    inc_angle_iono *= 180. / np.pi

    return inc_angle_iono


def lalo_ground2iono_shell_along_los(lat, lon, inc_angle=30, head_angle=-168, iono_height=450e3):
    """Convert the lat/lon of a point on the ground to the ionosphere thin-shell
    along the line-of-sight (LOS) direction.

    Reference: Jingyi, C., and H. A. Zebker (2012), Ionospheric Artifacts in Simultaneous
        L-Band InSAR and GPS Observations, Geoscience and Remote Sensing, IEEE Transactions on,
        50(4), 1227-1239, doi:10.1109/TGRS.2011.2164805.

    Parameters: lat/lon     - float, latitude/longitude of the point on the ground in degrees
                inc_angle   - float, incidence angle of the line-of-sight on the ground in degrees
                head_angle  - float, heading angle of the satellite orbit in degrees
                              from the north direction with positive in clockwise direction
                iono_height - float, height of the ionosphere thin-shell in meters
    """
    # deg -> rad & copy to avoid changing input variable
    inc_angle = np.array(inc_angle) * np.pi / 180.
    head_angle = np.array(head_angle) * np.pi / 180.

    # offset angle from equation (25) in Chen and Zebker (2012)
    off_iono = inc_angle - np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + iono_height) * np.sin(np.pi - inc_angle))

    # update lat/lon
    lat += off_iono * np.cos(head_angle) * 180. / np.pi
    lon += off_iono * np.sin(head_angle) * 180. / np.pi
    return lat, lon



######################################## IGS TEC ########################################
# https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html
# Ionospheric TEC products from IGS GNSS data
# Spatial resolution in latitude / longitude [deg]: 2.5 / 5.0
# Temporal resolution [hour]: 2.0

def calc_igs_iono_ramp(tec_dir, date_str, geom_file, box=None, print_msg=True):
    """Get 2D ionospheric delay from IGS TEC data for one acquisition.
    due to the variation of the incidence angle along LOS.

    Parameters: tec_dir        - str, path of the local TEC directory, i.e. ~/data/aux/IGS_TEC
                date_str       - str, date of interest in YYYYMMDD format
                geom_file      - str, path of the geometry file including incidenceAngle data
                box            - tuple of 4 int, subset in (x0, y0, x1, y1)
    Returns:    range_delay    - 2D np.ndarray for the range delay in meters
                vtec           - float, TEC value in zenith direction in TECU
                iono_lat/lon   - float, latitude/longitude in degrees of LOS vector in ionosphere shell
                iono_height    - float, height             in meter   of LOS vector in ionosphere shell
                iono_inc_angle - float, incidence angle    in degrees of LOS vector in ionosphere shell
    """
    # geometry
    tec_file = dload_igs_tec(date_str, tec_dir, print_msg=print_msg)
    iono_height = grab_ionex_height(tec_file)
    (iono_inc_angle,
     iono_lat,
     iono_lon) = prep_geometry_iono_shell_along_los(geom_file, box=box,
                                                    iono_height=iono_height)[:3]

    # time
    meta = readfile.read_attribute(geom_file)
    utc_hour = float(meta['CENTER_LINE_UTC']) / 3600.
    if print_msg:
        print('UTC hour: {:.2f}'.format(utc_hour))

    # extract zenith TEC
    freq = SPEED_OF_LIGHT / float(meta['WAVELENGTH'])
    vtec = get_igs_tec_value(tec_file, utc_hour, iono_lon, iono_lat)
    rang_delay = vtec2range_delay(vtec, iono_inc_angle, freq)

    return rang_delay, vtec, iono_lat, iono_lon, iono_height, iono_inc_angle


def get_igs_tec_value(tec_file, utc_hour, lat, lon):
    """Get the TEC value based on input lat/lon/datetime
    Parameters: tec_file - str, path of local TEC file
                utc_hour - float, UTC hour of the day
                lat/lon  - float, latitude / longitude in degrees
    Returns:    tecValue - float, TEC value in TECU
    """
    # read TEC file
    lons, lats, times, tecs = read_ionex_tec(tec_file)

    # find the TEC value nearest to the input location/time
    lonIdx = np.abs(lons - lon).argmin()
    latIdx = np.abs(lats - lat).argmin()
    timeIdx = int(np.round(utc_hour/2))
    tecValue = tecs[0,lonIdx,latIdx,timeIdx]

    return tecValue


def get_igs_tec_filename(tec_dir, date_str, datefmt='%Y%m%d'):
    """Get the local file name of downloaded IGS TEC product."""
    dd = dt.datetime.strptime(date_str, datefmt)
    doy = '{:03d}'.format(dd.timetuple().tm_yday)
    fbase = "jplg{0}0.{1}i.Z".format(doy, str(dd.year)[2:4])
    fbase = fbase[:-2]
    tec_file = os.path.join(tec_dir, fbase)
    return tec_file


def dload_igs_tec(d, out_dir, datefmt='%Y%m%d', print_msg=False):
    """Download IGS vertical TEC files computed by JPL
    Link: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html
    """
    # date info
    dd = dt.datetime.strptime(d, datefmt)
    doy = '{:03d}'.format(dd.timetuple().tm_yday)

    fbase = "jplg{0}0.{1}i.Z".format(doy, str(dd.year)[2:4])
    src_dir = "https://cddis.nasa.gov/archive/gnss/products/ionex/{0}/{1}".format(dd.year, doy)

    # input/output filename
    fname_src = os.path.join(src_dir, fbase)
    fname_dst = os.path.join(out_dir, fbase)
    fname_dst_uncomp = fname_dst[:-2]

    # download
    cmd = 'wget --continue --auth-no-challenge "{}"'.format(fname_src)
    #import pdb; pdb.set_trace()
    if os.path.isfile(fname_dst) and os.path.getsize(fname_dst) > 1000:
        cmd += ' --timestamping '
    if not print_msg:
        cmd += ' --quiet '
    else:
        print(cmd)

    # run cmd in output dir
    pwd = os.getcwd()
    os.chdir(out_dir)
    os.system(cmd)
    os.chdir(pwd)

    # uncompress
    # if output file 1) does not exist or 2) smaller than 400k in size or 3) older
    if (not os.path.isfile(fname_dst_uncomp)
            or os.path.getsize(fname_dst_uncomp) < 600e3
            or os.path.getmtime(fname_dst_uncomp) < os.path.getmtime(fname_dst)):
        cmd = "gzip --force --keep --decompress {}".format(fname_dst)
        if print_msg:
            print(cmd)
        os.system(cmd)

    return fname_dst_uncomp


def grab_ionex_height(tec_file):
    """Grab the height of the thin-shell ionosphere from IONEX file"""
    # read ionex file into list of lines
    with open(tec_file, 'r') as f:
        lines = f.readlines()

    # search for height - DHGT
    iono_height = None
    for line in lines:
        c = [i.strip() for i in line.strip().replace('\n', '').split()]
        if c[-1] == 'DHGT':
            iono_height = float(c[0]) * 1e3
            break
    return iono_height


def read_ionex_tec(igs_file):
    """Read IGS TEC file in IONEX format
    Parameters: igs_file - str, path of the TEC file
    Returns:    lon - 1D array for the longitude in size of (num_lon,) in degrees
                lat - 1D array for the latitude  in size of (num_lat,) in degrees
                map_times - 1D array for the time of the day in size of (13,) in minutes
                tec_array - 4D array for the vertical TEC value in size of (2, num_lon, num_lat, 13) in TECU
                    1 TECU = 10^16 electrons / m^2
    """

    #print(igs_file)
    if igs_file.startswith('igs'):
        tec_type = 'IGS_Final_Product'
    elif igs_file.startswith('igr'):
        tec_type = 'IGS_Rapid_Product'
    elif igs_file.startswith('jpr'):
        tec_type = 'JPL_Rapid_Product'
    elif igs_file.startswith('jpl'):
        tec_type = "JPL"
    else:
        tec_type = None

    #print(tec_type)
    ## =========================================================================
    ##
    ## The following section reads the lines of the ionex file for 1 day
    ## (13 maps total) into an array a[]. It also retrieves the thin-shell
    ## ionosphere height used by IGS, the lat./long. spacing, etc. for use
    ## later in this script.
    ##
    ## =========================================================================
    #print 'Transfering IONEX data format to a TEC/DTEC array for ',ymd_date

    ## Opening and reading the IONEX file into memory as a list
    linestring = open(igs_file, 'r').read()
    LongList = linestring.split('\n')

    ## Create two lists without the header and only with the TEC and DTEC maps (based on code from ionFR.py)
    AddToList = 0
    TECLongList = []
    DTECLongList = []
    for i in range(len(LongList)-1):
        ## Once LongList[i] gets to the DTEC maps, append DTECLongList
        if LongList[i].split()[-1] == 'MAP':
            if LongList[i].split()[-2] == 'RMS':
                AddToList = 2
        if AddToList == 1:
            TECLongList.append(LongList[i])
        if AddToList == 2:
            DTECLongList.append(LongList[i])
        ## Determine the number of TEC/DTEC maps
        if LongList[i].split()[-1] == 'FILE':
            if LongList[i].split()[-3:-1] == ['MAPS','IN']:
                num_maps = float(LongList[i].split()[0])
        ## Determine the shell ionosphere height (usually 450 km for IGS IONEX files)
        if LongList[i].split()[-1] == 'DHGT':
            ion_H = float(LongList[i].split()[0])
        ## Determine the range in lat. and long. in the ionex file
        if LongList[i].split()[-1] == 'DLAT':
            start_lat = float(LongList[i].split()[0])
            end_lat = float(LongList[i].split()[1])
            incr_lat = float(LongList[i].split()[2])
        if LongList[i].split()[-1] == 'DLON':
            start_long = float(LongList[i].split()[0])
            end_long = float(LongList[i].split()[1])
            incr_long = float(LongList[i].split()[2])
        ## Find the end of the header so TECLongList can be appended
        if LongList[i].split()[0] == 'END':
            if LongList[i].split()[2] == 'HEADER':
                AddToList = 1

    #print(ion_H)
    ## Variables that indicate the number of points in Lat. and Lon.
    points_long = ((end_long - start_long)/incr_long) + 1
    points_lat = ((end_lat - start_lat)/incr_lat) + 1   ## Note that incr_lat is defined as '-' here
    #points_long = int(((end_long - start_long)/incr_long) + 1)
    #points_lat = int(((end_lat - start_lat)/incr_lat) + 1)    ## Note that incr_lat is defined as '-' here
    number_of_rows = int(np.ceil(points_long/16))    ## Note there are 16 columns of data in IONEX format

    ## 4-D array that will contain TEC & DTEC (a[0] and a[1], respectively) values
    #print(points_long,points_lat,num_maps)
    a = np.zeros((2,int(points_long),int(points_lat),int(num_maps)))

    ## Selecting only the TEC/DTEC values to store in the 4-D array.
    for Titer in range(2):
        counterMaps = 1
        UseList = []
        if Titer == 0:
            UseList = TECLongList
        elif Titer == 1:
            UseList = DTECLongList
        for i in range(len(UseList)):
            ## Pointing to first map (out of 13 maps) then by changing 'counterMaps' the other maps are selected
            if UseList[i].split()[0] == ''+str(counterMaps)+'':
                if UseList[i].split()[-4] == 'START':
                    ## Pointing to the starting Latitude then by changing 'counterLat' we select TEC data
                    ## at other latitudes within the selected map
                    counterLat = 0
                    newstartLat = float(str(start_lat))
                    for iLat in range(int(points_lat)):
                        if UseList[i+2+counterLat].split()[0].split('-')[0] == ''+str(newstartLat)+'':
                            ## Adding to array a[] a line of Latitude TEC data
                            counterLon = 0
                            for row_iter in range(number_of_rows):
                                for item in range(len(UseList[i+3+row_iter+counterLat].split())):
                                    a[Titer,counterLon,iLat,counterMaps-1] = UseList[i+3+row_iter+counterLat].split()[item]
                                    counterLon = counterLon + 1
                        if '-'+UseList[i+2+counterLat].split()[0].split('-')[1] == ''+str(newstartLat)+'':
                            ## Adding to array a[] a line of Latitude TEC data. Same chunk as above but
                            ## in this case we account for the TEC values at negative latitudes
                            counterLon = 0
                            for row_iter in range(number_of_rows):
                                for item in range(len(UseList[i+3+row_iter+counterLat].split())):
                                    a[Titer,counterLon,iLat,counterMaps-1] = UseList[i+3+row_iter+counterLat].split()[item]
                                    counterLon = counterLon + 1
                        counterLat = counterLat + row_iter + 2
                        newstartLat = newstartLat + incr_lat
                    counterMaps = counterMaps + 1

    ## =========================================================================
    ##
    ## The section creates a new array that is a copy of a[], but with the lower
    ## left-hand corner defined as the initial element (whereas a[] has the
    ## upper left-hand corner defined as the initial element).  This also
    ## accounts for the fact that IONEX data is in 0.1*TECU.
    ##
    ## =========================================================================

    ## The native sampling of the IGS maps minutes
    incr_time = 24*60/int(num_maps-1)
    tec_array = np.zeros((2,int(points_long),int(points_lat),int(num_maps)))

    for Titer in range(2):
        incr = 0
        for ilat in range(int(points_lat)):
            tec_array[Titer,:,ilat,:] = 0.1*a[Titer,:,int(points_lat-1-ilat),:]

    #return points_long,points_lat,start_long,end_lat,incr_long,np.absolute(incr_lat),incr_time,num_maps,tec_array,tec_type

    lat = np.arange(start_lat, start_lat + points_lat*incr_lat, incr_lat)
    lon = np.arange(start_long, start_long + points_long*incr_long, incr_long)
    map_times = np.arange(0,incr_time*num_maps,incr_time)

    return lon, lat, map_times, tec_array


def plot_tec_animation(tec_file, save=False):
    """Plot the input tec file as animation"""
    from cartopy import crs as ccrs
    from matplotlib.animation import FuncAnimation
    from mintpy.utils import plot as pp

    def grab_date(tec_file, datefmt='%Y-%m-%d'):
        """Grab the date in YYYYMMDD format from the TEC filename"""
        tec_file = os.path.basename(tec_file)
        # year
        year = tec_file.split('.')[1][:2]
        if year[0] == '9':
            year = '19' + year
        else:
            year = '20' + year
        year = int(year)

        # month and day
        doy = int(tec_file.split('.')[0].split('g')[1][:3])
        dt_obj = dt.datetime(year, 1, 1) + dt.timedelta(doy - 1)
        date_str = dt_obj.strftime(datefmt)
        return date_str

    date_str = grab_date(tec_file)
    lon, lat, mins, tec_array = read_ionex_tec(tec_file)
    geo_box = (np.min(lon), np.max(lat), np.max(lon), np.min(lat))   # (W, N, E, S)
    extent = (geo_box[0], geo_box[2], geo_box[3], geo_box[1])

    # init figure
    fig, ax = plt.subplots(figsize=[10, 4], subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()
    pp.draw_lalo_label(geo_box, ax, projection=ccrs.PlateCarree(), print_msg=False)
    # init image
    data = tec_array[0,:,:,0].T
    im = ax.imshow(data, vmin=0, vmax=50, extent=extent, origin='upper', animated=True)
    # colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('TECU')

    # update image
    global idx
    idx = 0
    def animate(*args):
        global idx
        idx += 1
        if idx >= 12:
            idx -= 12
        # update image
        data = tec_array[0,:,:,idx].T
        im.set_array(data)
        # update title
        hour = idx * 2
        fig_title = '{} UTC {:02d}:00:00'.format(date_str, hour)
        ax.set_title(fig_title)
        return im,

    # play animation
    ani = FuncAnimation(fig, animate, interval=300, blit=True)

    # output
    if save:
        outfig = '{}.gif'.format(os.path.abspath(igs_file))
        print('saving animation to {}'.format(outfig))
        save_kwargs = dict()
        save_kwargs['transparent'] = True
        #ani.save(outfig, writer='ffmpeg', dpi=150, **save_kwargs)
        ani.save(outfig, writer='imagemagick', dpi=300, **save_kwargs)
    print('showing animation ...')
    plt.show()
    return



######################################## test ########################################

def main(iargs=None):
    tec_dir = os.path.expanduser('~/data/aux/TEC')
    tec_file = dload_igs_tec('20190409', tec_dir)
    plot_tec_animation(tec_file)
    return
if __name__ == '__main__':
    main(sys.argv[1:])
