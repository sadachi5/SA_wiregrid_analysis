SA wiregrid analysis
====================

## Setting
Environmental setting:

    . env-shell.sh

## Requirements of python libraries
There are several python libraries aside from the libraries installed by env-shell.sh.
Please intall them by

    pip3 install -r requirements.txt

**Do NOT forget to use pip3 instead of pip!**

## Versions
output\_ver2
------------
  - version description: first version having theta\_det (DB + each bolometer demod fit/TODs)
  - input data: raw data
  - data : 2021/02/05 Run22300609
  - output\_ver2/db/all.db
    - CREATE TABLE wiregrid(id INTEGER PRIMARY KEY,boloname STRING,lmfit_x0 REAL,lmfit_y0 REAL,lmfit_a REAL,lmfit_b REAL,lmfit_alpha REAL,lmfit_chisqr REAL,x0 REAL,y0 REAL,theta0 REAL,a REAL,b REAL,r REAL,alpha REAL,x0_err REAL,y0_err REAL,theta0_err REAL,a_err REAL,b_err REAL,r_err REAL,alpha_err REAL,chisqr REAL,wireangle0 REAL,wireangle0_err REAL,theta_wire0 REAL,theta_wire0_err REAL); 
    - Maybe this DB has bug or wrong data.

output\_ver3
------------
  - version description: first version having theta\_det (DB + each bolometer demod fit)
  - input data: output\_ver2 each bolometer TOD
  - output\_ver3/db/all.db
      - theta_det : detector angle (NO time-constant correction)
      - theta_det_err : detector angle error (but it is NOT correct now.)

output\_ver4
-------------
  - version description: theta\_det has time-constant correction (only DB)
  - input data: output\_ver3 DB
  - output\_ver4/db/all\_pandas\_correct\_label.db
    - theta\_det : detector angle (Wt time-constant correction)
        - If there is no stimulator data, theta\_det is not corrected. (2710/4324 bolos are corrected.)
    - theta\_det\_err : detector angle error (but it is NOT correct now.)
    - Stimulator data used for the correction is also included.

output\_ver5
------------
  - version description: DB with each wire angle demod data [(x,y) in demode complex plane]
    - reproduce from demod fit in ver3

output\_ver6
------------
  - version description: add wire\_angle=180 deg. data
    - change the time period of 180 deg. data (select only first 20 sec. This is similar length of the other data.)
    - reproduce from g3 data

output\_ver7/plot\_ver7
------------------------
  - version description: *change the libg3py3.py & loadbolo.py*
    - Use libg3py3\_test.py and loadbolo\_test.py
    - Change to *import loadbolo\_test.py instead of loadbolo.py in OneAngleData.py*
    - Modify reference offset in HWP angle calculation (using getOffsets() func. in libg3py3\_test.py
    - Add implementation of stimulator template DB to retrieve measured stimulator temperature in grid\_rotation\_analysis.py
        - Use averaged temperature measured by Jupiter
    - output\_ver7 is for bsub jobs. / plot\_ver7 is for test.

output\_ver8/plot\_ver8
------------------------
  - version description: *restore the change of the libg3py3.py & loadbolo.py*
    - Use libg3py3.py and loadbolo.py
    - Change to *import loadbolo.py instead of loadbolo_test.py in OneAngleData.py*
    - Main change from ver6: Add implementation of stimulator template DB to retrieve measured stimulator temperature in grid\_rotation\_analysis.py
        - Use averaged temperature measured by Jupiter
    - demod wire\_angle=180 deg data, but do not use it in fitDemodResult.py.
    - reproduce from TOD pickle file in ver6, which (output\_ver6/pkl/<wafer>/<boloname>/A..\_..deg.pkl were copied to output\_ver8/pkl/<wafer>/<boloname>/

## run scripts
 - (./plot.sh: make plot of TODs)

 - ./analysis.sh
    - From demod to fit circle

 - python3 run\_batch.py
    - Run demod & fit for many detectors by using batch job


## Scripts for each analysis steps
- (plot.py: simple TOD plotter)

- grid\_rotation\_analysis.py
    - get TOD & demod each wire angle data 
    - Using scripts:
        - OneAngleData.py

- fitDemodResult.py
    - fit circle data of demod datas
    - Using scripts:
        - minuitfit.py
        - LMfit.py (not used now)
        - plotFitResult.py


## DB modification

### merge & modify DB
- mergeDB.py: 
    - merge output of run\_batch.sh (fitDemodResult.py) & modify SQL database 
        - Tau (timing constant) calibration
        - Add hardware map DB
    - input : output\_verX/db/"wafer name"/\*.db
    - output:
        - output\_verX/db/all.db : sqlite DB without modification
        - output\_verX/db/all\_mod.db : sqlite DB with modification
        - output\_verX/db/all\_pandas.pkl : pandas DB with modification 
        - output\_verX/db/all\_pandas.db  : sqlite DB converted from pandas DB with modification 

### Compare DB
- compare\_db.py
    - merge two sqlite DBs with 'readout\_name' column & compare them
    - This script is used in the following scripts:
        - compare\_DB\_for\_labelcorrection.py
        - compare\_and\_make\_DB\_for\_labelcorrection.py
    - input :
        - original hardware map          : data/pb2a-20210205/pb2a_mapping.db
        - kyohei's corrected hardware map: data/ykyohei/mapping/pb2a_mapping_postv2.db
        - my wiregrid DB                 : output_verX/db/all_pandas.db
    - output : 
        - aho.png : 2D plot between varname1 v.s. varname2
        - aho.csv : converted from pandas of bolometers with varname!=varname2

- compare\_DB\_for\_labelcorrection.py
    - compare wiregrid DB and Kyohei's DB to find mislabel bolometers.
    - make a new sqlite DB with corrected pol\_angle,pixel\_type,bolo\_type on found mislabeled bolometers by wiregrid DB
        - base DB : my wire grid DB
    - add Kyohei's det\_offset\_x/y datas 
    - input : 
        - my wiregrid DB                 : output_verX/db/all_pandas.db
        - kyohei's corrected hardware map: data/ykyohei/mapping/pb2a_mapping_postv2.db
    - output:
        - output_verX/db/all_pandas_correct_label.db

- compare\_and\_make\_DB\_for\_labelcorrection.py
    - compare wiregrid DB and kyohei's DB to find mislabel bolometers.
    - make a new sqlite DB for SA official DB with corrected pol\_angle,pixel\_type,pixel\_handedness,bolo\_type on found mislabeled bolometers by wiregrid DB
        - base DB : original hardware map DB
    - If tau is Nan or 0 (stimulator data is not good), theta\_det is set to nan.
    - input : 
        - original hardware map          : data/pb2a-20210205/pb2a_mapping.db
        - kyohei's corrected hardware map: data/ykyohei/mapping/pb2a_mapping_postv2.db
        - my wiregrid DB                 : output_verX/db/all_pandas.db
    - output:
        - output_verX/db/pb2a_mapping_corrected_label_v2.db



## Plotting scripts

### Check TOD
- checkspeed.py
    - print average frequency [Hz] of HWP
    - input : output\_ver2/pkl/PB20.13.13/PB20.13.13\_Comb01Ch01/\*.pkl
        - TOD for only one bolometer
    - output: No (only printing)

### Check DB
- printDB.py :
    - Print contents of pandas pickle file
    - input : output\_verX/db/all\_pandas.pkl : pandas DB with modification 
    - output: No (only printing)

- checkDB.py
    - Make plots of pandas pickle file
    - input : output\_verX/db/all\_pandas.pkl : pandas DB with modification 
    - output: output\_verX/check\_db/\*.pdf : figures

### Check diff between wiregrid measured angle and design detector angle
- check\_absolute.py
    - make angle plots of DB with wiregrid corrected labels
        - 2D plot: theta_det_angle (wiregrid measured angle) v.s. pol_angle (design value) for good data (tau!=nan, theta_det_err<0.5deg, pol_angle!=nan) 
        - 2D plot: theta_det_angle (wiregrid measured angle) v.s. pol_angle (design value) for correct labels
        - 1D plot: diff. between measured angle(theta_det_angle) and design angle(pol_angle)
    - make focal plane plot
    - input : output\_ver4/db/all\_pandas\_correct\_label.db
    - output: out\_check\_absolute/check\_absolute.png

- check\_absolute\_labelcorrecteddb.py
    - make angle plots of DB with wiregrid corrected labels
    - input :
        - output\_ver4/db/all\_pandas.db # No correction data
        - output\_ver4/db/pb2a\_mapping\_corrected_label\_v2.db.db # Corrected data
    - output: out\_check\_absolute/check\_absolute\_labelcorrectedDB.png

- check\_absolute\_nocorr.py
    - make angle plots of DB without wiregrid label correction
    - input : 
        - output\_ver4/db/all\_pandas.db 
        - data/ykyohei/mapping/pb2a\_mapping\_postv2.db
    - output: out\_check\_absolute/check\_absolute\_nocorr.png

### Others
- makehist.py
    - make histogram of wiregrid measured angels for each bolometer groups
    - input : output\_ver4/db/all\_pandas.pkl
    - output: output\_ver4/summary/\*.png

- check\_jobs.py
    - check if all the jobs run by bsub is finished successfully.
        - check if error log is empty.
        - check if 'error' word exists in the output log of fitDemodResult.py.
        - check if 'error' word exists in the output log of grid_rotation_analysis_.py.
    - output information :
        0. # of all failed checks
        1. # of failed checks except the error of "Function minimum is not valid." (Fmin error) 
            - This number should be 0.
            - "Function..." error is caused by failure of fit. It can be occurred by bad data and can be ignored.
        2. # of memory limit error (TERM_MEMLIMIT)
    - results :
        - output_ver5 (only fitDemodResult.py)
            0. # of all failed checks               = 88
            1. # of failed checks except Fmin error = 1 
                - bsub log error in ./output_ver5/txt/PB20.13.12/bsub/bsub_PB20.13.12_Comb15Ch26.out
                - memory limit error (TERM_MEMLIMIT)
                - It has been rerun successfully.
            2. # of memory limit error = 1
        - output_ver6 (all analysis from g3 files)
            ### Summary for check_job for output_ver6 ################                                                 
            bsub error log (5461) : ./output_ver6/txt/PB20.13.12/bsub/bsub_PB20.13.12_Comb11Ch17.err...
            bsub log       (5461) : ./output_ver6/txt/PB20.13.12/bsub/bsub_PB20.13.12_Comb09Ch22.out...
            fitDemodResult.py         log (5461) : ./output_ver6/txt/PB20.13.12/fit/PB20.13.12_Comb03Ch20.out...
            grid_rotation_analysis.py log (5461) : ./output_ver6/txt/PB20.13.12/gridana/PB20.13.12_Comb03Ch20.out...
            There are errors in some jobs for output_ver6
            # of all failed checks = 203
            # of failed checks except the error of "Function minimum is not valid." = 0
            # of memory limit error = 0
            #################################################
        - output_ver8 (analysis from TOD obtained in output_ver6)
            ### Summary for check_job for output_ver8 ################
            bsub error log (5461) : ./output_ver8/txt/PB20.13.12/bsub/bsub_PB20.13.12_Comb11Ch17.err...
            bsub log       (5461) : ./output_ver8/txt/PB20.13.12/bsub/bsub_PB20.13.12_Comb09Ch22.out...
            fitDemodResult.py         log (5461) : ./output_ver8/txt/PB20.13.12/fit/PB20.13.12_Comb03Ch20.out...
            grid_rotation_analysis.py log (5461) : ./output_ver8/txt/PB20.13.12/gridana/PB20.13.12_Comb03Ch20.out...
            There are errors in some jobs for output_ver8
            # of all failed checks = 186
            # of failed checks except the error of "Function minimum is not valid." = 0
            # of "Function minimum is not valid." error = 186
            # of memory limit error = 0
            #################################################
