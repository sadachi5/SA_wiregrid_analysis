#!/bin/bash

njob=0

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


#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch02 -L &
#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch01 | tee ch01.out
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch02

#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch02
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch08
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch03


#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0alpha"    --init-alpha "0."   &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.2alpha"  --init-alpha "0.2"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.4alpha"  --init-alpha "0.4"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_0.5alpha"  --init-alpha "0.5"  &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.2alpha" --init-alpha "-0.2" &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.4alpha" --init-alpha "-0.4" &
#python3 fitDemodResult.py -b PB20.13.13_Comb01Ch01 --outsuffix "_-0.5alpha" --init-alpha "-0.5" &


wafer='PB20.13.13';
boloname='PB20.13.13_Comb01Ch01';
outdir='output_ver2';
filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
pickledir="${outdir}/pkl/${wafer}";
mkdir -vp  ${outdir}/txt/${wafer}/
python3 grid_rotation_analysis.py -b ${boloname} -o 'gridana_' -f ${filename} -d ${outdir}/plot/${wafer}/${boloname} -p ${pickledir} -L 2>&1>& ${outdir}/txt/${wafer}/gridana_${boloname}.out
python3 fitDemodResult.py -b ${boloname} -p ${pickledir} --pickleprefix 'gridana_' --picklesuffix '' -o ${outdir} --outprefix \"Fit_\" --outsuffix \"\" -v 1 2>&1>& ${outdir}/txt/${wafer}/fit_${boloname}.out
