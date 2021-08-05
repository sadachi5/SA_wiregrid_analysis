#!/bin/bash

njob=0

:<<'#_COMMENT_'
for i in {1..9}; do
    echo '';
    echo '';
    echo "Execute i=${i}";
    boloname="PB20.13.13_Comb01Ch0${i}"
    #python3 grid_rotation_analysis.py -b ${boloname} | tee tmp/gridana_ch0${i}.out
    #python3 fitDemodResult.py -b $boloname  | tee tmp/fit_ch0${i}.out &
    #python3 fitDemodResult.py -b $boloname  > tmp/fit_ch0${i}.out &
    njob=`expr $njob + 1`;

    njob=`ps uwwxx | grep -c python`;
    njob=`expr $njob - 1 `
    echo "njob = ${njob}"
    while [ $njob -gt 4 ]; do
        njob=`ps uwwxx | grep -c python`;
        njob=`expr $njob - 1 `
        echo "njob = ${njob}"
        sleep 5;
    done
done
#_COMMENT_

#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch02 -L &

#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch02 -L &
#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch01 | tee ch01.out
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch02

#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 -v 1 | tee plot_ver1_Ch01.out
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch02 -v 1
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch08 -v 1
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch03 -v 1


#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0alpha"    --init-alpha "0."   &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.2alpha"  --init-alpha "0.2"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.4alpha"  --init-alpha "0.4"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.5alpha"  --init-alpha "0.5"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.2alpha" --init-alpha "-0.2" &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.4alpha" --init-alpha "-0.4" &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.5alpha" --init-alpha "-0.5" &


wafer='PB20.13.13';
bolonames=(
'PB20.13.13_Comb01Ch01'
);
#'PB20.13.13_Comb01Ch01'
#'PB20.13.13_Comb01Ch02'
#'PB20.13.13_Comb01Ch03'
#'PB20.13.13_Comb01Ch07'

outdir='output_ver3';
#outdir='output_ver3_anglecalib';
#filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
filename='/group/cmb/polarbear/usr/sadachi/SparseWireCalibration/PB2a/g3compressed/Run22300609/';
loadpickledir="output_ver2/pkl/${wafer}";
pickledir="${outdir}/pkl/${wafer}";
mkdir -vp  ${outdir}/txt/${wafer}/gridana
mkdir -vp  ${outdir}/txt/${wafer}/fit


for boloname in ${bolonames[@]}; do
    echo $boloname;
    #python3 grid_rotation_analysis.py -b ${boloname} -o 'gridana_' -f ${filename} -d ${outdir}/plot/${wafer}/${boloname} -p ${pickledir} 2>&1>& ${outdir}/txt/${wafer}/gridana_${boloname}.out
    python3 fitDemodResult.py -b ${boloname} -p ${pickledir} --pickleprefix 'gridana_' --picklesuffix '' -d ${outdir} --outprefix 'Fit_' --outsuffix '' --notbatch -v 1 2>&1>& ${outdir}/txt/${wafer}/fit_${boloname}.out

    # Wt angle calibration
    #python3 grid_rotation_analysis.py -b ${boloname} -o 'gridana_' -f ${filename} -d ${outdir}/plot/${wafer}/${boloname} -l ${loadpickledir} -p ${pickledir} --anglecalib './output_ver3/db/all.db,wiregrid,readout_name' -L -v 2   2>&1 | tee ${outdir}/txt/${wafer}/gridana/gridana_${boloname}.out
    #python3 fitDemodResult.py -b ${boloname} -p ${pickledir} --pickleprefix 'gridana_' --picklesuffix '' -d ${outdir} --outprefix 'Fit_' --outsuffix '' -v 1 2>&1 | tee ${outdir}/txt/${wafer}/fit/fit_${boloname}.out


    #python3 grid_rotation_analysis.py -b '' -o 'gridana_' -f ${filename} -d ${outdir}/plot -p ${outdir}/pkl/ 2>&1>& ${outdir}/txt/gridana_all.out

    done
