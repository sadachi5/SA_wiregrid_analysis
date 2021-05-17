#!/bin/bash
datadir='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/';
#L='';
L='-L';

# Stimulator
#python3 plot.py --filename $datadir/Run22300607 --outdir 'plot_ver1' --outname StimulatorBefore --noHWP --start '20210205_171500' --end '20210205_173800' -L &
#python3 plot.py --filename $datadir/Run22300610 --outdir 'plot_ver1' --outname StimulatorAfter  --noHWP --start '20210205_182800' --end '20210205_184600' -L &
#python3 OneAngleData.py --filename $datadir/Run22300607 --outdir 'plot_ver1' --outname StimulatorBefore --noHWP --start '20210205_171500' --end '20210205_173800' -L &
python3 OneAngleData.py --filename $datadir/Run22300609 --outdir 'plot_ver1' --outname WireCalibration  --noHWP --start '20210205_180200' --end '20210205_181900' -L &

# Long term
#python3 plot.py --outdir 'plot_ver1' --outname 1738-1845 --start '20210205_173800' --end '20210205_184500' $L &
#python3 OneAngleData.py --filename $datadir/Run22300609 --outdir 'plot_ver1' --outname 1738-1845 --start '20210205_173800' --end '20210205_184500' $L &
#python3 OneAngleData.py --filename $datadir/Run22300609 --outdir 'plot_ver1' --outname 1738-1845 --start '20210205_173800' --end '20210205_184500' $L &

# 10 min term
<<"#__COMMENT__"
python3 OneAngleData.py --outdir 'plot_ver1' --outname 1739-1749 --start '20210205_173900' --end '20210205_174900' $L &
python3 OneAngleData.py --outdir 'plot_ver1' --outname 1749-1759 --start '20210205_174900' --end '20210205_175900' $L &
python3 OneAngleData.py --outdir 'plot_ver1' --outname 1759-1809 --start '20210205_175900' --end '20210205_180900' $L &
python3 OneAngleData.py --outdir 'plot_ver1' --outname 1809-1819 --start '20210205_180900' --end '20210205_181900' $L &
python3 OneAngleData.py --outdir 'plot_ver1' --outname 1819-1829 --start '20210205_181900' --end '20210205_182900' $L &
#__COMMENT__
#python3 plot.py --outdir 'plot_ver1' --outname 1739-1749 --start '20210205_173900' --end '20210205_174900' &
#python3 plot.py --outdir 'plot_ver1' --outname 1749-1759 --start '20210205_174900' --end '20210205_175900' &
#python3 plot.py --outdir 'plot_ver1' --outname 1759-1809 --start '20210205_175900' --end '20210205_180900' &
#python3 plot.py --outdir 'plot_ver1' --outname 1809-1819 --start '20210205_180900' --end '20210205_181900' &
#python3 plot.py --outdir 'plot_ver1' --outname 1819-1829 --start '20210205_181900' --end '20210205_182900' &

# A-1~10 (0deg~180deg) ver1
<<"#__COMMENT__"
python3 OneAngleData.py --outdir 'plot_ver1' --outname A1_0deg     --start '20210205_180230' --end '20210205_180400' $L & # 1.5min
python3 OneAngleData.py --outdir 'plot_ver1' --outname A2_22.5deg  --start '20210205_180510' --end '20210205_180530' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A3_45deg    --start '20210205_180640' --end '20210205_180700' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A4_67.5deg  --start '20210205_180800' --end '20210205_180820' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A5_90deg    --start '20210205_181000' --end '20210205_181130' $L & # 1.5min
python3 OneAngleData.py --outdir 'plot_ver1' --outname A6_112.5deg --start '20210205_181320' --end '20210205_181340' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A7_135deg   --start '20210205_181557' --end '20210205_181617' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A8_157.5deg --start '20210205_181750' --end '20210205_181810' $L & # 20sec
python3 OneAngleData.py --outdir 'plot_ver1' --outname A9_180deg   --start '20210205_182000' --end '20210205_182400' $L & # 4min
#__COMMENT__

# A-0~11 ver0
# A-1
#python3 plot.py --outname A-1L_-22.5deg --start '20210205_173915' --end '20210205_174500'
#python3 plot.py --outname A-1_-22.5deg --start '20210205_173930' --end '20210205_174430' # 5min
# A0
#python3 plot.py --outname A0L_0deg --start '20210205_174900' --end '20210205_175800'
#python3 plot.py --outname A0_0deg --start '20210205_174930' --end '20210205_175600' # 5.5min
# A1
#python3 plot.py --outname A1L_22.5deg --start '20210205_180220' --end '20210205_180420'
#python3 plot.py --outname A1_22.5deg --start '20210205_180230' --end '20210205_180400' # 1.5min
# A2
#python3 plot.py --outname A2L_45deg --start '20210205_180500' --end '20210205_180540'
#python3 plot.py --outname A2_45deg --start '20210205_180510' --end '20210205_180530' # 20sec
# A3
#python3 plot.py --outname A3L_67.5deg --start '20210205_180630' --end '20210205_180710'
#python3 plot.py --outname A3_67.5deg --start '20210205_180640' --end '20210205_180700' # 20sec
# A4
#python3 plot.py --outname A4L_90deg --start '20210205_180750' --end '20210205_180830'
#python3 plot.py --outname A4_90deg --start '20210205_180800' --end '20210205_180820' # 20sec
# A5
#python3 plot.py --outname A5L_112.5deg --start '20210205_180930' --end '20210205_181150'
#python3 plot.py --outname A5_112.5deg --start '20210205_181000' --end '20210205_181130' # 1.5min
# A6
#python3 plot.py --outname A6L_135deg --start '20210205_181300' --end '20210205_181350'
#python3 plot.py --outname A6_135deg --start '20210205_181320' --end '20210205_181340' # 20sec
# A7
#python3 plot.py --outname A7L_157.5deg --start '20210205_181540' --end '20210205_181630'
#python3 plot.py --outname A7_157.5deg --start '20210205_181557' --end '20210205_181617' # 20sec
# A8
#python3 plot.py --outname A8L_180deg --start '20210205_181740' --end '20210205_181820'
#python3 plot.py --outname A8_180deg --start '20210205_181750' --end '20210205_181810' # 20sec
# A9
#python3 plot.py --outname A9L_202.5deg --start '20210205_181940' --end '20210205_182800'
#python3 plot.py --outname A9_202.5deg --start '20210205_182000' --end '20210205_182400' # 4min
