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

output\_ver7, plot\_ver7
------------------------
  - version description: *change the libg3py3.py & loadbolo.py*
    - Use libg3py3\_v2.py and loadbolo\_v2.py (Old name: libg3py3\_test.py, loadbolo\_test.py)
    - Change to *import loadbolo\_v2.py instead of loadbolo.py in OneAngleData.py*
    - Modify reference offset in HWP angle calculation (using getOffsets() func. in libg3py3\_v2.py
    - Add implementation of stimulator template DB to retrieve measured stimulator temperature in grid\_rotation\_analysis.py
        - Use averaged temperature measured by Jupiter
    - output\_ver7 is for bsub jobs. / plot\_ver7 is for test.

output\_ver8, plot\_ver8
------------------------
  - version description: *restore the change of the libg3py3.py & loadbolo.py*
    - Use libg3py3.py and loadbolo.py
    - Change to *import loadbolo.py instead of loadbolo\_test.py in OneAngleData.py*
    - Main change from ver6: Add implementation of stimulator template DB to retrieve measured stimulator temperature in grid\_rotation\_analysis.py
        - Use averaged temperature measured by Jupiter
    - demod wire\_angle=180 deg data, but do not use it in fitDemodResult.py.
    - reproduce from TOD pickle file in ver6, which (output\_ver6/pkl/<wafer>/<boloname>/A..\_..deg.pkl were copied to output\_ver8/pkl/<wafer>/<boloname>/

output\_ver9, plot\_ver9
------------------------
  - version description: Use libg3py3\_v2.py & loadbolo\_v2.py
    - Use libg3py3.py and loadbolo.py
    - Change to *import loadbolo.py --> loadbolo\_v2.py in OneAngleData.py*
        - HWP encoder 0 point should be corrected.
    - demod wire\_angle=180 deg data, but do not use it in fitDemodResult.py.

ver9.2 (tag, branch) 
-----------------------
The last version using libg3py3.py


ver10
----------
- version description: *Use sa_pipeline_software to retrieve TOD & HWP encoder angle data*

From this version, sa pipeline is used to read TOD.
To use *simons_array_offline_software* at kekcc, there are several modifications.
1. Make a *sa_config.py* file for myself (in library/mychange)
2. Comment-out *from toast import qarray as qa*
    - Importing toast has an issue at kekcc.
3. Comment-out *from .sa_cuts import ...* to avoid toast importing
4. Add paths on *PYTHONPATH* in *env-shell.sh*

Other modification on *simons_array_offline_software*:
1. To use time clip operation on the TOD, I modified *sa_pipline_filters.py/OperatorClipBeginEnd()*
2. To fix bug in *sa_pipline_filters.py/OperatorClipBeginEnd()*, 
   add items on the list that would go with bolometer (L195).

- Update database to 20211004 (database/pb2a-20211004) after fitDemod
    - *Amplitude calibration* of ADC counts: NOT updated
    - *Tau correction*:
        - updated in merge.py: pb2a-20211004/pb2a\_stim.db
            - It has duplicate bolometers. (Use DISTINCT to retrieve sqlite3 database.)
            - To avoid the duplicated bolos, add new selection run_subid=='[1, 4, 7, 10, 13, 16, 19, 22]'
    - *pb2a_mapping* (offset\_det\_x,y):
        - NOTE: No updating in merge.py because newer database than p22a-20210205 has calibrated pol\_angles but it needs design values in pol\_angle.
    - *Kyohei's pb2a_focalplane DB* :
        - updated in compare_DB_for_labelcorrection.py: ykyohei/mapping/pb2a\_focalplane\_postv4.db
            - (The HWM hash is not changed) 6f306f8261c2be68bc167e2375ddefdec1b247a2

- Add errors, tau correction in wiregrid DB
    - Add systematics from tau\_err in *merge.py*
        - *theta_det_err* : (Same as before) Statistical error of theta\_det calibration
        - *thete_det_err_tau* : Error from tauerr in tau calibration
        - *thete_det_err_total* : Root of squared sum of the above errors
    - Add tau correction on theta\_det in *merge.py*
        - *theta_det_taucorr* : - 2 * tau * hwp_speed * 2pi
        - *theta_det* : theta_det original + theta_det_taucorr

- Fill 0 in NULL in tau, tauerr in wiregrid in *merge.py*
- Update *compare\_DB\_for\_labelcorrection.py* for ver10
    - Wiregrid DB is merged with an old pb2a\_focalplane DB which has original detector labels and design pol\_angle.
    - It is compared with Kyohei's DB (postv4) to get corrected pixel labels.
    - Drop NaN bolometers in readout\_name or theta\_det (wiregrid data) for label corrected DB
    - Add a column of 'isCorrectLabel'
        - True if mislabel is corrected or not mislabel.
    - Fix bugs in bolo\_name or pixel\_number

- Update *check\_absolute\_nocorr.py* for ver10
    - Remove old check\_absolute\_nocorr.py & create from check\_absolute.py
    - No plots with *det\_offset\_x/y* or *mislabel* 
        because the DB before label correction does not have them
    - Outliers cut: >45deg --> >15deg.

- Update *check\_absolute.py* for ver10
    - Outliers cut: >45deg --> >15deg.

- Add *modifyDB.py*
    - Make a *pb2a_wiregrid_ver10.db* from *all_pandas_correct_label.db*.
    - Modify & check the DB 
    - theta\_det
        - Demod[TOD * exp(-i(4*HWP\_angle-2*(theta\_det+tau*hwp\_speed*2.*pi)))

## run scripts
 - (./plot.sh: make plot of TODs)

 - ./test\_pipeline.sh
    - From demod to fit circle
    - run each stage of analysis by using loadbolo\_pipeline.py

 - ./analysis\_misc.sh
    - For test or check run
    - From demod to fit circle

 - python3 run\_batch.py
    - Run demod & fit for many detectors by using batch job

 - ./analysis\_DB.sh
    - Run DB modification or check after the fit circle


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
### Summary of DB modification steps
1. mergeDB.py
2. compare\_DB\_for\_labelcorrection.py
3. check\_absolute.py
4. modifyDB.py

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
          or a new pb2a_focalplane DB in pb2a_mapping.db
    - output:
        - output_verX/db/all_pandas_correct_label.db/pkl

- compare\_and\_make\_DB\_for\_labelcorrection.py (Old)
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

### Create wiregrid DB for public from all\_pandas\_correct\_label.db
- modifyDB.py
    - Make a new sqlite DB for SA official DB with corrected pol\_angle,pixel\_type,pixel\_handedness,bolo\_type by wiregrid DB
    - Modifications:
        - Keep the following columns:
            - bolo_name
            - pixel_name
            - wafer_number
            - band
            - pixel_name
            - pixel_number
            - pixel_type
            - pixel_handedness
            - bolo_type
            - mislabel
            - pol_angle
            - theta_det
            - theta_det_err
        - theta\_det is shifted by 90 deg. (0~pi) to match the definition as in pol\_angle.
        - Drop bolometers:
            - if tau is Nan or 0 (stimulator data is not good)
            - if pol\_angle is Nan
            - if theta\_det\_err is >=0.5deg
        - If bolo_name is Nan and labels are corrected, bolo_name is corrected.
            -  There are still mis-label in bolo_name related to the band. (534 bolos in ver10)
        - Rename theta\_det\_err\_total --> theta\_det\_err


## Plotting scripts

### Check TOD
- checkspeed.py
    - print average frequency [Hz] of HWP
    - input : output\_verX/pkl/PB20.13.13/PB20.13.13\_Comb01Ch01/\*.pkl
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
    - make angle plots of DB after wiregrid corrected labels
        - 2D plot: theta_det_angle (wiregrid measured angle) v.s. pol_angle (design value) for good data (tau!=nan, theta_det_err<0.5deg, pol_angle!=nan) 
        - 2D plot: theta_det_angle (wiregrid measured angle) v.s. pol_angle (design value) for correct labels
        - 1D plot: diff. between measured angle(theta_det_angle) and design angle(pol_angle)
    - make focal plane plot
    - input : output\_verX/db/all\_pandas\_correct\_label.db
    - output: output\_verX/check\_absolute/\*.png

- check\_absolute\_labelcorrecteddb.py
    - make angle plots of DB with wiregrid corrected labels
    - input :
        - output\_verX/db/all\_pandas.db # No correction data
        - output\_verX/db/pb2a\_mapping\_corrected_label\_v2.db.db # Corrected data
    - output: out\_check\_absolute/check\_absolute\_labelcorrectedDB.png

- check\_absolute\_nocorr.py
    - make angle plots of DB before wiregrid label correction
    - input : 
        - output\_verX/db/all\_pandas.db 
    - output: output\_verX/check\_absolute\_nocorr/\*.png

### Check diff between 2 DBs for the same readout\_name
- diff\_db.py
    - make diff or same histograms for each columns for the same *readout\_name*
    - input : 2 DBs (any DB is OK if it has the readout\_name column.)
    - output: output\_verX/db/diff\_db.png, diff\_db2.png, diff\_db.csv (different bolometer info)

### Others
- makehist.py
    - make histogram of wiregrid measured angels for each bolometer groups
    - input : output\_verX/db/all\_pandas.pkl
    - output: output\_verX/summary/\*.png

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
