#!/usr/bin/env python3
# Author: Yuan-Kai Liu, Zhang Yunjun, Oct 2022
"""Test mintpy.objects.euler module for the Euler pole and velocity computation."""


import collections
import math

import numpy as np
import pandas as pd
# https://github.com/insarlab/MintPy/blob/main/src/mintpy/objects/euler_pole.py
from mintpy.objects.euler_pole import MASY2DMY, EulerPole, sph2cart
from mintpy.plate_motion import ITRF2014_PMM
from mintpy.utils.utils0 import calc_azimuth_from_east_north_obs

# validation against the UNAVCO Plate Motion Calculator
# https://www.unavco.org/software/geodetic-utilities/plate-motion-calculator/plate-motion-calculator.html
# accessed on Oct 29, 2022.
# config: height=0, model=ITRF2014, reference=NNR no-net-rotation
# note: azimuth here is measured from the north with positive for clock-wise direction
# unit:                              deg deg mm/yr   deg   mm/yr mm/yr
Tag = collections.namedtuple('Tag', 'lat lon speed azimuth vel_n vel_e')
POINT_PM = {
    'Australia' : Tag(-24,  132,  67.56,  28.94,  59.12,  32.69),
    'Eurasia'   : Tag( 27,   62,  28.85,  79.21,   5.40,  28.34),
    'Arabia'    : Tag( 18,   48,  46.51,  50.92,  29.32,  36.11),
}
PLATE_NAMES = POINT_PM.keys()


def get_lalo(lat0, lat1, lon0, lon1, N):
    """
    Parameters: lat0 - starting lat
                lat1 -   ending lat
                lon0 - starting lon
                lon1 -   ending lon
                N    - number of grids in both directions
    Returns:    lats - gridded lat
                lons - gridded lon
    """
    lats = np.linspace(lat0, lat1, N)
    lons = np.linspace(lon0,lon1, N)
    lats, lons = np.meshgrid(lats, lons)
    return lats, lons


def test_euler_pole_initiation():
    print('Test 1: EulerPole object initiation and vector2pole conversion.')

    for plate_name in PLATE_NAMES:
        print(f'Plate name: ITRF2014-PMM {plate_name}')

        # get PMM info from Table 1 in Altamimi et al. (2017) as reference
        plate_pmm = ITRF2014_PMM[plate_name]

        # build EulerPole obj
        pole_obj = EulerPole(wx=plate_pmm.omega_x, wy=plate_pmm.omega_y, wz=plate_pmm.omega_z)

        # compare rotation rate: ITRF2014_PMM vs. Euler vector to pole conversion
        print(f'Reference  rotation rate from Altamimi et al. (2017): {plate_pmm.omega:.4f} deg/Ma')
        print(f'Calculated rotation rate from pole2vector conversion: {plate_pmm.omega:.4f} deg/Ma')
        assert math.isclose(plate_pmm.omega, pole_obj.rotRate*MASY2DMY, abs_tol=5e-4)
        print('Pass.')


def test_plate_motion_calc():
    print('Test 2: Plate motion calculation and validation against UNAVCO website.')

    for plate_name in PLATE_NAMES:
        # get UNAVCO result as reference
        point_pm = POINT_PM[plate_name]
        print(f'Plate = ITRF2014-PMM {plate_name}, point lat/lon = {point_pm.lat}/{point_pm.lon}')

        # calculate using EulerPole in m/yr
        plate_pmm = ITRF2014_PMM[plate_name]
        pole_obj = EulerPole(wx=plate_pmm.omega_x, wy=plate_pmm.omega_y, wz=plate_pmm.omega_z)
        ve, vn = pole_obj.get_velocity_enu(point_pm.lat, point_pm.lon, print_msg=False)[:2]

        print(f'Reference   (UNAVCO): vel_e={point_pm.vel_e:.2f}, vel_n={point_pm.vel_n:.2f} mm/yr')
        print(f'Calculation (MintPy): vel_e={ve*1e3:.2f}, vel_n={vn*1e3:.2f} mm/yr')
        assert math.isclose(point_pm.vel_e, ve*1e3, abs_tol=0.05)
        assert math.isclose(point_pm.vel_n, vn*1e3, abs_tol=0.05)
        print('Pass.')


if __name__ == '__main__':

    print('-'*50)
    #print(f'Testing {__file__}')

    ########################################################
    print('Example: Simple testing')

    test_euler_pole_initiation()

    test_plate_motion_calc()


    ########################################################
    print('\nExample: computation on grid')

    lat0, lat1 = 30, 40
    lon0, lon1 = 90, 110
    lats, lons = get_lalo(lat0, lat1, lon0, lon1, 1000)

    P_eura = EulerPole(name='Eurasia', wx=ITRF2014_PMM['Eurasia'].omega_x, wy=ITRF2014_PMM['Eurasia'].omega_y, wz=ITRF2014_PMM['Eurasia'].omega_z)

    v = 1e3 * np.array(P_eura.get_velocity_enu(lats, lons))


    ########################################################
    ## Example: add/computate relative Euler pole
    print('\nExample: add/computate relative Euler pole')

    name  = 'Sinai'
    dname = 'Sinai'

    # Castro-Perdomo et al., GJI, 2022
    # (https://doi.org/10.1093/gji/ggab353)
    # the Euler pole of the Sinai subplate with respect to ITRF2014 using five reliable stations
    # located on the Sinai subplate at distances greater than 40 km from the DST (CSAR, ALON, YRCM, BSHM and RAMO)
    # The SINAI-ITRF2014 Euler pole derived here is 54.7±0.7° N, 347.8±4.0° E, ω = 0.417± 0.021° Ma−1
    P_sinai = EulerPole(pole_lat=54.7, pole_lon=347.8, rot_rate=0.417, unit='deg/Ma', name='Sinai')
    print('New plate: Sinai (Castro-Perdomo et al., GJI, 2022)\n', P_sinai)


    # Augment ITRF2014 PMM
    TagPMM = collections.namedtuple('Tag', 'name num_site omega_x omega_y omega_z omega wrms_e wrms_n')
    ITRF2014_PMM[dname] = TagPMM(name, None, P_sinai.wx, P_sinai.wy, P_sinai.wz, P_sinai.rotRate, None, None)


    # build EulerPole obj
    P_arab = EulerPole(name='Araiba' , wx=ITRF2014_PMM[ 'Arabia'].omega_x, wy=ITRF2014_PMM[ 'Arabia'].omega_y, wz=ITRF2014_PMM[ 'Arabia'].omega_z)
    P_eura = EulerPole(name='Eurasia', wx=ITRF2014_PMM['Eurasia'].omega_x, wy=ITRF2014_PMM['Eurasia'].omega_y, wz=ITRF2014_PMM['Eurasia'].omega_z)
    P_nubi = EulerPole(name='Nubia'  , wx=ITRF2014_PMM[  'Nubia'].omega_x, wy=ITRF2014_PMM[  'Nubia'].omega_y, wz=ITRF2014_PMM[  'Nubia'].omega_z)


    # relative motion
    P_nubi_arab = P_nubi - P_arab
    P_sini_arab = P_sinai - P_arab


    # Sinai-Arabia pole (Viltres et. al., 2021 Table 1)
    P_sini_arab_2 = EulerPole(wx=-0.1055, wy=-0.0282, wz=-0.0803, unit='deg/Ma')


    print('\nRelative Euler pole: Nubia-Arabia (based on Altamimi et al., 2017)\n', P_nubi_arab)
    print('\nRelative Euler pole: Sinai-Arabia (based on Castro-Perdomo et al., 2022)\n', P_sini_arab)
    print('\nRelative Euler pole: Sinai-Arabia (based on Viltres et al., 2021)\n', P_sini_arab_2)

    print(pd.DataFrame.from_dict(ITRF2014_PMM, orient='index'))

    print('-'*50)
    print('End of the test and Example. Good!')
