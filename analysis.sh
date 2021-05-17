#!/bin/bash
for i in {1..9}; do
    echo '';
    echo '';
    boloname="PB20.13.13_Comb01Ch0${i}"
    #python3 grid_rotation_analysis.py -b ${boloname} | tee tmp/gridana_ch0${i}.out
    #python3 fitDemodResult.py $boloname  | tee tmp/fit_ch0${i}.out &
done


#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch02 -L &
#python3 grid_rotation_analysis.py -b PB20.13.13_Comb01Ch01 | tee ch01.out
#python3 fitDemodResult.py PB20.13.13_Comb01Ch02

python3 fitDemodResult.py PB20.13.13_Comb01Ch02
python3 fitDemodResult.py PB20.13.13_Comb01Ch08
