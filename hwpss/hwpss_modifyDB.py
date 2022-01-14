import sys, os;
from modifyDB import modify


if __name__=='__main__' :
    ver='ver3';
    suffix='';
    opts = [];
    if len(sys.argv)>1:
        ver = sys.argv[1];
        pass;
    if len(sys.argv)>2:
        suffix = sys.argv[2]; 
        pass;
    if len(sys.argv)>3:
        opts = sys.argv[3].split(','); 
        pass;
    plotdir = f'hwpss/output_{ver}/db/modifyDB/'
    infile  = f'hwpss/output_{ver}/db/all_pandas_correct_label.db'
    outfile = f'hwpss/output_{ver}/db/pb2a_hwpss_{ver}{suffix}'
    modify(infile=infile, outfile=outfile, plotdir=plotdir, opts=opts, isHWPSS=True);
    pass;


