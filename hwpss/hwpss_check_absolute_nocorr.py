#!/bin/python
import sys, os;

from check_absolute_nocorr import check_absolute


if __name__=='__main__' :
    ver='ver3';
    isCorrectHWPenc=True;
    suffix='';
    if len(sys.argv)>1:
        ver = sys.argv[1];
        pass;
    if len(sys.argv)>2:
        isCorrectHWPenc = (bool)((int)(sys.argv[2]));
        pass;
    if len(sys.argv)>3:
        suffix = sys.argv[3]; 
        pass;
    outdir = f'hwpss/output_{ver}/check_absolute_nocorr{suffix}';
    if not os.path.isdir(outdir):
        os.mkdir(outdir);
        pass;
    outfile = f'{outdir}/';
    check_absolute(
        ver=ver, outfile=outfile,
        isCorrectHWPenc=isCorrectHWPenc, isHWPSS=True);
