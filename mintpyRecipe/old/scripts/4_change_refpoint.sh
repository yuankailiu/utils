#! /bin/bash


# =============== Read defined variables from json file ==================
my_json="./params.json"
declare -A dic
while IFS="=" read -r key value
do
    dic[$key]="$value"
done < <(jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' $my_json)
# =============== ===================================== ==================
# Get parameters
proc_home="${dic['proc_home']}"
config="${proc_home}/${dic['config']}"
refla=${dic['ref_lat']}
reflo=${dic['ref_lon']}

## Reference point
find . -name 'timeseries*.h5' -exec   reference_point.py {}  -t ${config} \;
reference_point.py inputs/ERA5.h5             -t ${config}
reference_point.py inputs/SET.h5              -t ${config}
reference_point.py inputs/timeseriesIon.h5    -t ${config}

## Reference date
find . -name 'timeseries*.h5' -exec   reference_date.py  {}  -t ${config} \;
reference_date.py inputs/ERA5.h5              -t ${config}
reference_date.py inputs/SET.h5               -t ${config}
reference_date.py inputs/timeseriesIon.h5     -t ${config}

## Others (forget what they do...)
#diff.py timeseries.h5 timeseries_ERA5.h5 -o inputs/ERA5_ref.h5
#rm -rf rms_timeseriesResidua* reference_date.txt

echo " ============================= "
echo "Finish referencing!!"
