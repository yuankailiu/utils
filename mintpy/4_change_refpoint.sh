#! /bin/bash

# In lat, lon coordinates
rlat=30.2923
rlon=34.2466

# In y,x coordinates
# ry=384
# rx=82    

reference_point.py timeseries.h5              --lat ${rlat} --lon ${rlon}
reference_point.py timeseries_ERA5.h5         --lat ${rlat} --lon ${rlon}
reference_point.py timeseries_ERA5_demErr.h5  --lat ${rlat} --lon ${rlon}
reference_point.py timeseriesResidual.h5      --lat ${rlat} --lon ${rlon}
reference_point.py timeseriesResidual_ramp.h5 --lat ${rlat} --lon ${rlon}
diff.py timeseries.h5 timeseries_ERA5.h5 -o inputs/ERA5_ref.h5

rm -rf rms_timeseriesResidua* reference_date.txt
bash ./1_run_SBApp.sh


