#! /bin/bash

## I forgot what this script does
##
##  ykliu 2022.01.28

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
mask_file="${proc_home}/${dic['tcoh_mask']}"
tCoh_file="${proc_home}/${dic['tcoh_file']}"
geom_file="${proc_home}/${dic['geom_file']}"
cmap=${dic['tcoh_cmap']}


## Plotting function
velo_file="$velo_dir/velocity_RlP.h5"
view=" view.py $velo_file velocity --nodisplay --dpi 300 "
roi=" --sub-lon $xmin $xmax --sub-lat $ymin $ymax --ref-lalo ${refla} ${reflo} "
opt=" --dem $dem_file --alpha $alpha --dem-nocontour --shade-exag $shade_exag --mask $tmCoh_mask -u mm $roi -v -3 3 "

## Command
cmd=" $view $roi $opt "
plot_coherence_matrix.py ${ifgStack} -c ${cmap} --view-cmd $cmd
