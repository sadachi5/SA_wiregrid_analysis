#!/bin/env python

import pandas as pd;
import numpy as np;
from utils import colors, getPandasPickle, plottmphist;

database = 'output_ver10/db/all_pandas.pkl';
outdir = 'output_ver10/check_db';

df = getPandasPickle(database);
print(df.keys());

# No cut
plottmphist(df['r'],i='',outname='all_r',xlabel=r'$r$',nbins=500,xrange=[0,5000],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df['r'],i='',outname='all_r_zoom',xlabel=r'$r$',nbins=50,xrange=[0,50],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df['chisqr'],i='',outname='all_chisqr',xlabel=r'$\chi ^2$',nbins=500,xrange=[0,5000],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df['chisqr'],i='',outname='all_chisqr_zoom',xlabel=r'$\chi ^2$',nbins=50,xrange=[0,50],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df['r'],y=df['chisqr'],i='',outname='all_r-chisqr',xlabel=r'$r$',ylabel=r'$\chi ^2$',nbins=50,xrange=[0,500],yrange=[0,500],log=True,show=False,outdir=outdir);
plottmphist(df['r'],y=df['chisqr'],i='',outname='all_r-chisqr_zoom',xlabel=r'$r$',ylabel=r'$\chi ^2$',nbins=50,xrange=[0,50],yrange=[0,50],log=True,show=False,outdir=outdir);

# theta_det_err cut
basecut0 = 'theta_det_err<0.5*{:e}/180.'.format(np.pi);
df_sel0 = df.query(basecut0);
print('# of "{}" = {}/{}'.format(basecut0, len(df_sel0), len(df)));
plottmphist(df_sel0['r'],i='',outname='thetaless1_r',xlabel=r'$r$',nbins=500,xrange=[0,500],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df_sel0['chisqr'],i='',outname='thetaless1_chisqr',xlabel=r'$\chi ^2$',nbins=500,xrange=[0,500],log=True,show=False,drawflow=True,outdir=outdir);

# check with theta_det_err cut 
datas = [];
cuts = [0,0.5,1.0,1.5,2.,10.];
cuts_strs = [];
for i in range(len(cuts)) :
    if i<len(cuts)-1 :
        selection = 'theta_det_err*180./{pi}>={cut1} & theta_det_err*180./{pi}<{cut2}'.format(pi=np.pi,cut1=cuts[i],cut2=cuts[i+1]);
        cut_str   = '%.1f'%cuts[i]+r' deg <= $\Delta \theta_{\mathrm{det}}$ < '+'%.1f'%cuts[i+1]+' deg';
    else :
        selection = 'theta_det_err*180./{pi}>={cut}'.format(pi=np.pi,cut=cuts[i]);
        cut_str   = '%.1f'%cuts[i]+r' deg <= $\Delta \theta_{\mathrm{det}}$';
        pass;
    datas.append(df.query(selection)['r']);
    cuts_strs.append(cut_str);
    pass;
plottmphist(df['theta_det_err']*180./np.pi,i='',outname='all_theta_det_err',xlabel=r'$\Delta \theta_{det} [deg.]$',nbins=360,xrange=[0,90.],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(df['theta_det_err']*180./np.pi,i='',outname='all_theta_det_err_zoom',xlabel=r'$\Delta \theta_{det} [deg.]$',nbins=100,xrange=[0,10.],log=True,show=False,drawflow=True,outdir=outdir);
plottmphist(datas,i='',outname='r_wt_different_theta_det_err',xlabel=r'$r$',ylabel=r'Counts of bolometers',nbins=100,xrange=[0,1000.],log=True,show=False,drawflow=True,stacked=True,nHist=len(datas),label=cuts_strs,outdir=outdir);

