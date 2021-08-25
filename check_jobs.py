import os, sys
import glob, re
import subprocess
import numpy as np

bsubsuccessline = 'Successfully completed.';
Fmin_errorline  = 'Function minimum is not valid.';
memory_errorline= 'TERM_MEMLIMIT';

def check_jobs(outdir='output_ver5') :
    # 0: all failures,
    # 1: except error of Fmin_errorline = "Function minimum is not valid."
    # 2: Fmin_errorline
    # 3: memory limit error
    Nfail = np.array([0,0,0,0]); 

    #RED   = '\033[31m'; # print color: red
    #RESET = '\033[0m' ; # print color: reset to default
    RED   = ''; # No color
    RESET = '' ; # No color

    # Get all log file names
    bsub_errlog = glob.glob('./{}/txt/PB*/bsub/*.err'.format(outdir));
    bsub_outlog = glob.glob('./{}/txt/PB*/bsub/*.out'.format(outdir));
    fit_outlog  = glob.glob('./{}/txt/PB*/fit/*.out'.format(outdir));
    grid_outlog = glob.glob('./{}/txt/PB*/gridana/*.out'.format(outdir));

    Nbsuberr = len(bsub_errlog);
    Nbsubout = len(bsub_outlog);
    Nfitout  = len(fit_outlog );
    Ngridout = len(grid_outlog);

    # Check bsub error log
    print();
    for log in bsub_errlog :
        if not os.path.isfile(log):
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! There is no log file: {}'.format(log)+RESET);
            continue;
        size = os.path.getsize(log);
        #print('error log file size = {}'.format(size));
        if size>0 :
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! An error log file is not empty: {}'.format(log)+RESET);
            print('File size = {}'.format(size));
            print('--- Please check it by the following command --------');
            print(RED+'cat {}'.format(log)+RESET);
            print('-----------------------------------------------------');
            pass;
        pass;

    # Check bsub out log
    print();
    for log in bsub_outlog :
        if not os.path.isfile(log):
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! There is no log file: {}'.format(log)+RESET);
            continue;
        grep_outputs=[];
        grep_outputs.append(subprocess.run(['grep', bsubsuccessline, log], encoding='utf-8', stdout=subprocess.PIPE).stdout);
        grep_outputs.append(subprocess.run(['grep', memory_errorline, log], encoding='utf-8', stdout=subprocess.PIPE).stdout);
        Ngrep_outputs = np.array([len(out) for out in grep_outputs]);
        #print('grep_output for log = {}'.format(grep_outputs));
        if Ngrep_outputs[0]==0 :
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! A bsub log file does not have the success sentence.: {}'.format(log)+RESET);
            if Ngrep_outputs[1]>0 :
                Nfail[3] += 1;
                print('Memory limit error sentence : {}'.format(grep_outputs[1]));
                pass;
            print('--- Please check it by the following command --------');
            print(RED+'cat {}'.format(log)+RESET);
            print('-----------------------------------------------------');
            pass;
        pass;

    # Check fit out log
    print();
    for log in fit_outlog :
        if not os.path.isfile(log):
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! There is no log file: {}'.format(log)+RESET);
            continue;
        grep_outputs=[];
        # Search for "error" words (ignore difference between lower-/upper- cases)
        grep_outputs.append(subprocess.run('grep -i error {} | grep -v ErrorView'.format(log), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout);
        # Search for "warning" words (ignore difference between lower-/upper- cases)
        #grep_outputs.append(subprocess.run('grep -i warning {} '.format(log), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout);
        Ngrep_outputs = np.array([len(out) for out in grep_outputs]);
        if not (np.all(Ngrep_outputs==0)) :
            #print('Ngrep_outputs :',Ngrep_outputs);
            #print('grep_outputs  :',grep_outputs);
            Nfail[0] += 1;
            errorout = grep_outputs[0] if Ngrep_outputs[0]>0 else 'None'
            # Search for "Function minimum is not valid."
            isFmin_error = False;
            if Fmin_errorline in grep_outputs[0] :
                isFmin_error = True;
                errorout = 'Fmin error'
                Nfail[2] += 1;
            else :
                Nfail[1] += 1;
                pass;
            print(RED+'WARNING! A fit log file have error sentence.: {}'.format(log)+RESET);
            print('Error   sentence : {}'.format(errorout));
            #print('Warning sentence : {}'.format( grep_outputs[1] if Ngrep_outputs[1]>0 else 'None' ));
            if not isFmin_error :
                print('--- Please check it by the following command --------');
                print(RED+'cat {}'.format(log)+RESET);
                print('-----------------------------------------------------');
                pass;
            pass;
        pass;

    # Check grid out log
    print();
    for log in grid_outlog :
        if not os.path.isfile(log):
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! There is no log file: {}'.format(log)+RESET);
            continue;
        grep_outputs=[];
        # Search for "error" words (ignore difference between lower-/upper- cases)
        grep_outputs.append(subprocess.run('grep -i error {} | grep -v ErrorView'.format(log), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout);
        # Search for "warning" words (ignore difference between lower-/upper- cases)
        #grep_outputs.append(subprocess.run('grep -i warning {} '.format(log), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout);
        Ngrep_outputs = np.array([len(out) for out in grep_outputs]);
        if not (np.all(Ngrep_outputs==0)) :
            Nfail[0] += 1;
            Nfail[1] += 1;
            print(RED+'WARNING! A gridana log file have error/warning sentence.: {}'.format(log)+RESET);
            print('Error   sentence : {}'.format( grep_outputs[0] if Ngrep_outputs[0]>0 else 'None' ));
            #print('Warning sentence : {}'.format( grep_outputs[1] if Ngrep_outputs[1]>0 else 'None' ));
            print('--- Please check it by the following command --------');
            print(RED+'cat {}'.format(log)+RESET);
            print('-----------------------------------------------------');
            pass;
        pass;


    print();
    print();
    print('### Summary for check_job for {} ################'.format(outdir));
    print('bsub error log ({}) : {}...'.format(Nbsuberr,bsub_errlog[0] if Nbsuberr>0 else 'None'));
    print('bsub log       ({}) : {}...'.format(Nbsubout,bsub_outlog[0]) if Nbsubout>0 else 'None');
    print('fitDemodResult.py         log ({}) : {}...'.format(Nfitout , fit_outlog[0]  if Nfitout >0 else 'None'));
    print('grid_rotation_analysis.py log ({}) : {}...'.format(Ngridout, grid_outlog[0] if Ngridout>0 else 'None'));
    if  Nfail[0]==0:
        print('Successfully finished all jobs for {}'.format(outdir));
    else :
        print('There are errors in some jobs for {}'.format(outdir));
        print('# of all failed checks = {}'.format(Nfail[0]));
        print('# of failed checks except the error of "{}" = {}'.format(Fmin_errorline, Nfail[1]));
        print('# of "{}" error = {}'.format(Fmin_errorline, Nfail[2]));
        print('# of memory limit error = {}'.format(Nfail[3]));
    pass;
    print('#################################################'.format(outdir));
    print();
    print();

    return Nfail;


if __name__=='__main__' :
    outdir = 'output_ver8';
    check_jobs(outdir);


