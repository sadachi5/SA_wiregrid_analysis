#!/bin/python
import sys, os;
import numpy as np;
import sqlite3, pandas;
import copy;
from utils import deg0to180, theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta;

from check_absolute import check_absolute

if __name__=='__main__' :
    ver='ver3';
    isCorrectHWPenc=True;
    suffix='';
    opts = [];
    if len(sys.argv)>1:
        ver = sys.argv[1];
        pass;
    if len(sys.argv)>2:
        isCorrectHWPenc = (bool)((int)(sys.argv[2]));
        pass;
    if len(sys.argv)>3:
        suffix = sys.argv[3]; 
        pass;
    if len(sys.argv)>4:
        opts = sys.argv[4].split(','); 
        pass;
    outdir = f'hwpss/output_{ver}/check_absolute{suffix}';
    if not os.path.isdir(outdir):
        os.mkdir(outdir);
        pass;
    outfile = f'{outdir}/';
    check_absolute(
        ver=ver, outfile=outfile,
        isCorrectHWPenc=isCorrectHWPenc, isHWPSS=True,
        opts=opts);
