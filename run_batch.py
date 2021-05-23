import os, sys
import time
import subprocess
import numpy as np

import libg3py3 as libg3

doRun = True;
ignoreFileExist = True;

#wafers=['PB20.13.13'];
#wafers=['PB20.13.15'];
#wafers=['PB20.13.13', 'PB20.13.15', 'PB20.13.28', 'PB20.13.11', 'PB20.13.12', 'PB20.13.10', 'PB20.13.31'];
#wafers=['PB20.13.15', 'PB20.13.28', 'PB20.13.11', 'PB20.13.12', 'PB20.13.10', 'PB20.13.31'];
wafers=['PB20.13.28', 'PB20.13.11', 'PB20.13.12', 'PB20.13.10', 'PB20.13.31'];
outdir='output_ver2';
#filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
filename='/group/cmb/polarbear/usr/sadachi/SparseWireCalibration/PB2a/g3compressed/Run22300609/';
selections=[];
#selections=['PB20.13.13_Comb01Ch02','PB20.13.13_Comb01Ch03'];


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


def runJob(maxNjob=30) :
    
    # loop over wafers
    for wafer in wafers :
        bolos = wafer_bolos[wafer];
        print('Wafer = {} : {} bolos'.format(wafer, len(bolos)));
    
        # make script
        scriptdir = '{}/scripts/{}'.format(outdir,wafer);
        if not os.path.isdir(scriptdir) : os.makedirs(scriptdir);
    
        # setting
        pickledir='{}/pkl/{}'.format(outdir, wafer);
        prefix   ='gridana_';
        plotdir  ='{}/plot/{}'.format(outdir, wafer);
        txtbsubdir   ='{}/txt/{}/bsub'.format(outdir, wafer);
        txtgridanadir='{}/txt/{}/gridana'.format(outdir, wafer);
        txtfitdir    ='{}/txt/{}/fit'.format(outdir, wafer);
        if not os.path.isdir(txtbsubdir)    : os.makedirs(txtbsubdir);
        if not os.path.isdir(txtgridanadir) : os.makedirs(txtgridanadir);
        if not os.path.isdir(txtfitdir)     : os.makedirs(txtfitdir);
    
        # loop over bolos :
        Njob = 0;
        for bolo in bolos :
            scriptfilename = '{}/{}.sh'.format(scriptdir, bolo);
            txtgridana     = '{}/{}.out'.format(txtgridanadir, bolo);
            txtfit         = '{}/{}.out'.format(txtfitdir, bolo);
            #print('Creating script file: {}'.format(scriptfilename));
            scriptfile     = open(scriptfilename, 'w');
            commands='#!/bin/bash\ncd {};\n. ./env-shell.sh;\n'.format(os.environ['PWD'])+\
            'python3 grid_rotation_analysis.py \
            -b \"{boloname}\" -o \"{prefix}\" -f \"{filename}\" \
            -d \"{plotdir}/{boloname}\" -p \"{pickledir}\" --loadpickledir \"{pickledir}\" \
            2>&1>& {txtout};\n'.format(\
            boloname=bolo, filename=filename, \
            plotdir=plotdir, pickledir=pickledir, prefix=prefix, \
            txtout=txtgridana)+\
            'python3 fitDemodResult.py \
            -b \"{boloname}\" -p \"{pickledir}\" --pickleprefix \"{prefix}\" --picklesuffix \"\" \
            -d \"{outdir}\" --outprefix \"Fit_\" --outsuffix \"\" -v 1 \
            2>&1>& {txtout};\n'.format(\
            boloname=bolo, pickledir=pickledir, outdir=outdir, prefix=prefix, txtout=txtfit)
    
            scriptfile.write(commands);
    
            command = 'bsub -q cmb_px "source {script} > {txtdir}/bsub_{boloname}.log 2>&1"'.format(script=scriptfilename, txtdir=txtbsubdir, boloname=bolo);
            print(command);
            if doRun:
                if (not bolo in selections) and len(selections)>0 : continue;
                if os.path.isfile(txtfit) and ignoreFileExist :
                    print('WARNING! {} is ignored because {} exists.'.format(bolo, txtfit));
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
