import os, sys
import numpy as np
import libg3py3 as libg3

doRun = True;

wafers=['PB20.13.13'];
outdir='output_ver2';
filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';


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
    txtdir   ='{}/txt/{}'.format(outdir, wafer);
    if not os.path.isdir(txtdir) : os.makedirs(txtdir);

    # loop over bolos :
    for bolo in bolos :
        scriptfilename = '{}/{}.sh'.format(scriptdir, bolo);
        #print('Creating script file: {}'.format(scriptfilename));
        scriptfile     = open(scriptfilename, 'w');
        commands='#!/bin/bash\ncd {};\n. ./env-shell.sh;\n'.format(os.environ['PWD'])+\
        'python3 grid_rotation_analysis.py -b \"{boloname}\" -o \"{prefix}\" -f \"{filename}\" -d \"{plotdir}/{boloname}\" -p \"{pickledir}\" -L 2>&1>& {txtdir}/{prefix}{boloname}.out;\n'.format(boloname=bolo, filename=filename, plotdir=plotdir, txtdir=txtdir, wafer=wafer, pickledir=pickledir, prefix=prefix)+\
        'python3 fitDemodResult.py -b \"{boloname}\" -p \"{pickledir}\" --pickleprefix \"{prefix}\" --picklesuffix \"\" -o \"{outdir}\" --outprefix \"Fit_\" --outsuffix \"\" -v 1 2>&1>& {txtdir}/fit_{boloname}.out;\n'.format(boloname=bolo, pickledir=pickledir, wafer=wafer, outdir=outdir, prefix=prefix, txtdir=txtdir)

        #'python3 fitDemodResult.py -b \"{boloname}\" -p \"{pickledir}\" --pickleprefix \"{prefix}\" --picklesuffix \"\" -o \"{outdir}\" --outprefix \"Fit_\" --outsuffix \"\" -v 1 2>&1>& {txtdir}/fit_{boloname}.out;\n'.format(boloname=bolo, pickledir=pickledir, wafer=wafer, outdir=outdir, prefix=prefix, txtdir=txtdir)

        scriptfile.write(commands);

        command = 'bsub -q sx "source {script} > {txtdir}/bsub_{boloname}.log 2>&1"'.format(script=scriptfilename, txtdir=txtdir, boloname=bolo);
        print(command);
        if doRun:
            os.system(command);
            print('-->> Submitted Job');
            pass;
        pass;
    pass;
        

