import os, sys
import time
import subprocess
import numpy as np
import copy

import libg3py3 as libg3

doRun = True;
doGridAna = True;
doFit     = True;
# Number of bolometers in one job
#Nbolobunch= 50; # to execute only fit-script
Nbolobunch= 1; 
# Arguments of multiple blometers are implemented only in Fit but not in GridAna.
# So please doGridAna=False if Nbolobunch>1.

# Max # of jobs running at the same time
maxNjob=100;

# filename of boloname list to be run
#bolofile='' # In this case, bololist is used.
bolofile='output_ver8/job_check/3rd/fails_bololist2.out'
# boloname list. NOTE: This is valid only when bolofile is empty
bololist=[];
#blolist=['PB20.13.13_Comb01Ch02','PB20.13.13_Comb01Ch03'];

ignoreFileExist = False;

# All
wafers=['PB20.13.13', 'PB20.13.15', 'PB20.13.28', 'PB20.13.11', 'PB20.13.12', 'PB20.13.10', 'PB20.13.31'];
#wafers=['PB20.13.10'];

#outdir1='output_ver2';
#outdir2='output_ver5';

outdir1='output_ver8';
outdir2='output_ver8';

#filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
filename='/group/cmb/polarbear/usr/sadachi/SparseWireCalibration/PB2a/g3compressed/Run22300609/';

#optgrid = '' # read TOD from g3 files
optgrid = '-L' # read TOD from pickle files
optfit = '--excludeAngle 180'

# get bolonames
g3c=libg3.G3Compressed(filename,loadtime=False)
bolonames = np.array(g3c.bolonames_all) ;
print('get {} bolos from {}'.format(len(bolonames), filename));
# retrieve wafername
wafer_bolos = {};
for boloname in bolonames :
    wafer = boloname.split('_')[0];
    if wafer in wafer_bolos.keys() : wafer_bolos[wafer].append(boloname);
    else                           : wafer_bolos[wafer]=[boloname];
    pass;


def runJob() :

    if len(bolofile)>0 :
        f = open(bolofile);
        bololist = [ s.strip() for s in f.readlines() ];
        pass;
    print('bololist = {}'.format(bololist));
    
    # loop over wafers
    for wafer in wafers :
        bolos = wafer_bolos[wafer];
        print('Wafer = {} : {} bolos'.format(wafer, len(bolos)));
    
        # make script
        scriptdir = '{}/scripts/{}'.format(outdir2,wafer);
        if not os.path.isdir(scriptdir) : os.makedirs(scriptdir);
    
        # setting
        pickledir='{}/pkl/{}'.format(outdir1, wafer);
        prefix   ='gridana_';
        plotdir  ='{}/plot/{}'.format(outdir2, wafer);
        txtbsubdir   ='{}/txt/{}/bsub'.format(outdir2, wafer);
        txtgridanadir='{}/txt/{}/gridana'.format(outdir2, wafer);
        txtfitdir    ='{}/txt/{}/fit'.format(outdir2, wafer);
        if not os.path.isdir(txtbsubdir)    : os.makedirs(txtbsubdir);
        if not os.path.isdir(txtgridanadir) : os.makedirs(txtgridanadir);
        if not os.path.isdir(txtfitdir)     : os.makedirs(txtfitdir);
    
        # loop over bolos :
        bolobunches = [];
        bolobunch   = [];
        for bolo in bolos :
            print(bolo);
            if (not bolo in bololist) and len(bololist)>0 : continue;
            bolobunch.append(bolo);
            if len(bolobunch)==Nbolobunch :
                bolobunches.append(copy.copy(bolobunch));
                bolobunch.clear();
                pass;
            pass;
        print(bolobunches);

        # loop over bolobunches :
        Njob = 0;
        for bolobunch in bolobunches :
            bolo0 = bolobunch[0];
            scriptfilename = '{}/{}.sh'.format(scriptdir, bolo0);
            txtgridana     = '{}/{}.out'.format(txtgridanadir, bolo0);
            txtfit         = '{}/{}.out'.format(txtfitdir, bolo0);
            #print('Creating script file: {}'.format(scriptfilename));
            scriptfile     = open(scriptfilename, 'w');
            commands='#!/bin/bash\ncd {};\n. ./env-shell.sh;\n'.format(os.environ['PWD']);

            bolonames = ','.join(bolobunch);
            plotdirs  = ','.join([ plotdir+'/'+bolo for bolo in bolobunch ]);
            if doGridAna :
                commands += \
                'python3 grid_rotation_analysis.py \
                -b \"{bolonames}\" -o \"{prefix}\" -f \"{filename}\" \
                -d \"{plotdirs}\" -p \"{pickledir}\" --loadpickledir \"{pickledir}\" {opt} \
                2>&1>& {txtout};\n'.format(\
                bolonames=bolonames, filename=filename, \
                plotdirs=plotdirs, pickledir=pickledir, prefix=prefix, \
                txtout=txtgridana, opt=optgrid);
                pass;
            if doFit :
                commands += \
                'python3 fitDemodResult.py \
                -b \"{bolonames}\" -p \"{pickledir}\" --pickleprefix \"{prefix}\" --picklesuffix \"\" \
                -d \"{outdir}\" --outprefix \"Fit_\" --outsuffix \"\" {opt} -v 1 \
                2>&1>& {txtout};\n'.format(\
                bolonames=bolonames, pickledir=pickledir, outdir=outdir2, prefix=prefix, txtout=txtfit, opt=optfit)
                pass;
    
            scriptfile.write(commands);
    
            # -q : queue (cmb_px : long time & large memory)
            # -oo: output file for stdout (overwrite the file)
            # -eo: output file for stderr (overwrite the file)
            command = 'bsub -q cmb_px -oo {txtdir}/bsub_{boloname}.out -eo {txtdir}/bsub_{boloname}.err  "source {script} 2>&1> {txtdir}/bsub_{boloname}.log"'.format(script=scriptfilename, txtdir=txtbsubdir, boloname=bolo0);
            print(command);
            if doRun:
                if os.path.isfile(txtfit) and ignoreFileExist :
                    print('WARNING! {} is ignored because {} exists.'.format(bolonames, txtfit));
                    continue;
                Njob = len(subprocess.getoutput('bjobs').split('\n')) - 1;
                while Njob>maxNjob :
                    Njob = len(subprocess.getoutput('bjobs').split('\n')) - 1;
                    time.sleep(10);
                    pass;
                os.system(command);
                print('-->> Submitted Job');
                pass;
            pass; # End of loop over bolos

        pass; # End of loop over wafers
            
    return 0;
    

if __name__=='__main__' :
    runJob();
    pass;
