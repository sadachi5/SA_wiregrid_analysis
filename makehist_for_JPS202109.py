import os, sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd;

from utils import colors, getPandasPickle, theta0topi;

def checknumber(selection_bool, description, n_all) :
    n_sel = selection_bool.sum();
    print('# of {:30s} : {:4d}/{:4d}'.format(description, n_sel, n_all));
    return n_sel, selection_bool;

def main(database, baseselect=[''],outfile='aho.png',verbose=0,
        nbins=36*5, stacked=True,
        hwp_speed = 2.0, # Hz
        calib_tau = False,
        draw_all  = True,
        ) :
    # make output dir
    outdir = './' if not '/' in outfile else '/'.join(outfile.split('/')[:-1]);
    print(outdir);
    if not os.path.isdir(outdir) : 
        print('making outdir = {}...'.format(outdir));
        os.makedirs(outdir);
        pass;
    # get pandas from pickle file
    df = getPandasPickle(database);
    print('pandas column names = {}'.format(df.columns.values));
    if verbose>0 :
        pd.set_option('display.max_columns', 20)
        print('---- Pandas head in {} -----------'.format(database));
        print(df.head());
        print('----------------------------------');
        pd.set_option('display.max_columns', 5)
        pass;

    # base selection
    baselabel = 'All';
    if len(baseselect)>1 :  baselabel = baseselect[1];
    elif len(baseselect[0])>0 : baselabel = baseselect[0];
    baselabel = baseselect[1]+' All' if len(baseselect)>1 else 'All';
    df_select = df.query(baseselect[0]);


    # check numbers
    n_all   = len(df_select);
    n_readout   , b_readout  = checknumber((df_select['readout_name'].str.len())>0, 'readout_name(size>0)'  , n_all);
    n_wafer     , b_wafer    = checknumber((df_select['wafer_number'].str.len())>0, 'wafer_number(size>0)'  , n_all);
    n_UQ        , b_UQ       = checknumber((df_select['pixel_type'      ]=='U') | (df_select['pixel_type'      ]=='Q'), 'U or Q', n_all);
    n_TB        , b_TB       = checknumber((df_select['bolo_type'       ]=='T') | (df_select['bolo_type'       ]=='B'), 'T or B', n_all);
    n_AB        , b_AB       = checknumber((df_select['pixel_handedness']=='A') | (df_select['pixel_handedness']=='B'), 'A or B', n_all);
    n_tau       , b_tau      = checknumber( df_select['tau']>0., 'tau>0' , n_all);

    # check incorrect data
    n_waferNan  , b_waferNan = checknumber( df_select['wafer_number'].isnull()     , 'wafer=NaN'   , n_all);
    n_UQNan     , b_UQNan    = checknumber( df_select['pixel_type'].isnull()       , 'UQ=NaN'      , n_all);
    n_TBNan     , b_TBNan    = checknumber( df_select['bolo_type'].isnull()        , 'TB=NaN'      , n_all);
    n_ABNan     , b_ABNan    = checknumber( df_select['pixel_handedness'].isnull() , 'AB=NaN'      , n_all);
    n_tauNan    , b_tauNan   = checknumber( df_select['tau'].isnull()              , 'tau=NaN'     , n_all);
    n_TBNone    , b_TBNone   = checknumber( (df_select['bolo_type']==None) , 'TB=None'     , n_all);
    n_tauNone   , b_tauNone  = checknumber( (df_select['tau']==None)       , 'tau=None'    , n_all);
    n_TBD       , b_TBD      = checknumber( df_select['bolo_type']=='D'    , 'TB=D'        , n_all);
    print('total of wafer   = n_wafer + n_waferNan          = {}'.format(n_wafer + n_waferNan));
    print('total of UQ      = n_UQ + n_UQNan                = {}'.format(n_UQ + n_UQNan));
    print('total of TB      = n_TB + n_TBNan +TBD + TBNone  = {}'.format(n_TB + n_TBNan + n_TBD + n_TBNone));
    print('total of AB      = n_AB + n_ABNan                = {}'.format(n_AB + n_ABNan));
    print('total of tau     = n_tau + n_tauNan + n_tauNone  = {}'.format(n_tau + n_tauNan + n_tauNone));

    #dataname = 'wireagnle0';
    #dataname = 'theta_wire0';
    dataname = 'theta_det';
    # convert factor
    #convertF = 180./np.pi*0.5; # convert from rad --> deg & 2*theta_det --> theta_det
    convertF = 180./np.pi; # convert from rad --> deg
 
    fig, axs = plt.subplots(1,1);
    fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=1, hspace=1, left=0.15, right=0.95,bottom=0.15, top=0.95)
 
    ihist = 0;
 
    y = df_select[dataname];
    if calib_tau : y -= 2. * df_select['tau'] * (hwp_speed*2.*np.pi);
    if draw_all : axs.hist(theta0topi(y)*convertF, bins=nbins, range=(0.,180.), histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ihist], alpha=0.4, label=baselabel, stacked=stacked);
    ihist +=1;
 
    selections = [\
        #["bolo_type=='T' & pixel_type=='U'", 'UT'],\
        ["bolo_type=='T' & pixel_type=='U' & pixel_handedness=='A'", 'UT/A'],\
        ["bolo_type=='T' & pixel_type=='U' & pixel_handedness=='B'", 'UT/B'],\
        #["bolo_type=='B' & pixel_type=='U'", 'UB'],\
        ["bolo_type=='B' & pixel_type=='U' & pixel_handedness=='A'", 'UB/A'],\
        ["bolo_type=='B' & pixel_type=='U' & pixel_handedness=='B'", 'UB/B'],\
        #["bolo_type=='T' & pixel_type=='Q'", 'QT'],\
        ["bolo_type=='T' & pixel_type=='Q' & pixel_handedness=='A'", 'QT/A'],\
        ["bolo_type=='T' & pixel_type=='Q' & pixel_handedness=='B'", 'QT/B'],\
        #["bolo_type=='B' & pixel_type=='Q'", 'QB'],\
        ["bolo_type=='B' & pixel_type=='Q' & pixel_handedness=='A'", 'QB/A'],\
        ["bolo_type=='B' & pixel_type=='Q' & pixel_handedness=='B'", 'QB/B'],\
        ];
    data_selects = [];
    labels = [];
    ndata = len(selections);

    n_sels = [];
    for i, selectinfo in enumerate(selections) :
        selection   = selectinfo[0] + ('' if len(baseselect[0])==0 else ('&' + baseselect[0]));
        selectlabel = selectinfo[1] if len(selectinfo)>1 else selection.strip().replace("'",'').replace('==','=').replace('_type','');
        labels.append(selectlabel);
        df_select = df.query(selection);
        n_sel     = len(df_select);
        print('selection = {}'.format(selection));
        print('    # of bolos = {}'.format(n_sel));
        y = df_select[dataname];
        if calib_tau : y -= 2. * df_select['tau'] * (hwp_speed*2.*np.pi);
        data_selects.append(theta0topi(y)*convertF);
        n_sels.append(n_sel);
        pass;
    print('Sum of selected bolos = {}'.format(sum(n_sels)));


    axs.hist(data_selects, bins=nbins, range=(0.,180.), histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ihist:ihist+ndata], alpha=0.4, label=labels, stacked=stacked);
 
    axs.set_title(baseselect[1] if len(baseselect)>1 else '');
    #axs.set_xlabel(r'$\theta(Wire angle=0)/2 = \theta_{\mathrm{det}}$ [deg.]',fontsize=16);
    axs.set_xlabel(r'$\theta_{\mathrm{det}}$ [deg.]',fontsize=16);
    axs.set_ylabel(r'# of bolometers',fontsize=16);
    axs.set_xticks(np.arange(0,180,22.5));
    axs.set_xlim(0,180);
    #axs.set_ylim(-5000,5000);
    axs.tick_params(labelsize=12);
    axs.grid(True);
    axs.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
 
    fig.savefig(outfile);
 
    return 0;

if __name__=='__main__' :
    #database = 'output_ver2/db/all_pandas.pkl';
    #outdir   = 'output_ver2/summary';
    #database = 'output_ver3/db/all_pandas.pkl';
    #outdir0  = 'output_ver3/summary';
    #database = 'output_ver4/db/all_pandas.pkl';
    #outdir0  = 'output_ver4/summary';
    #database = 'output_ver5/db/all_pandas_correct_label.pkl';
    database = 'output_ver8/db/all_pandas_correct_label.pkl';
    outdir0  = 'plot_for_JPS202109/makehist';
    wafers=['13.13', '13.15', '13.28', '13.11', '13.12', '13.10', '13.31'];
    nbins = (int)(180/2.5);
    ext = 'png';
    for stacked, stacklabel in (False,'nostack'),(True,'stack') :
        draw_all = stacked; # stacked histograms has a total histogram.
        outdir = outdir0+'/'+stacklabel;
        main(database, baseselect=["readout_name==readout_name",''], outfile=outdir+'/all_nocut.'+ext,stacked=stacked,calib_tau=False,draw_all=draw_all,nbins=36*5,verbose=1);
        #for calib_tau, tausuffix, taulabel in (False,'',r' (No $\tau$-corr.)'),(True,'_taucorr',r' (Wt $\tau$-corr.)') :
        for calib_tau, tausuffix, taulabel in [(False,'',r'')] :
            # 0.017453292519943295 rad. = 1 deg.
            wg_qualitycut = 'theta_det_err<0.017453292519943295*0.5';
            main(database, baseselect=["tau>0.",'Wt stimulator / No wiregrid quality cut'], outfile=outdir+'/all'+tausuffix+'_noWGQcut.'+ext,calib_tau=calib_tau,draw_all=draw_all,stacked=stacked,nbins=nbins,verbose=1);
            main(database, baseselect=["tau>0.&"+wg_qualitycut,'Wt stimulator / Wt wiregrid quality cut'], outfile=outdir+'/all'+tausuffix+'.'+ext,calib_tau=calib_tau,draw_all=draw_all,stacked=stacked,nbins=nbins,verbose=1);
            for wafer in wafers :
                print();
                main(database, baseselect=["wafer_number=='{}'&".format(wafer)+wg_qualitycut,wafer+' All'+taulabel], 
                        calib_tau=calib_tau,draw_all=draw_all,stacked=stacked,nbins=nbins,verbose=1,outfile=outdir+'/'+wafer+'_all'+tausuffix+'.'+ext);
                #main(database, baseselect=["band==90&wafer_number=='{}'&".format(wafer) +wg_qualitycut,wafer+' 90GHz'+taulabel],
                #        calib_tau=calib_tau,draw_all=draw_all,stacked=stacked,nbins=nbins,verbose=1,outfile=outdir+'/'+wafer+'_90GHz'+tausuffix+'.'+ext);
                #main(database, baseselect=["band==150&wafer_number=='{}'&".format(wafer)+wg_qualitycut,wafer+' 150GHz'+taulabel],
                #        calib_tau=calib_tau,draw_all=draw_all,stacked=stacked,nbins=nbins,verbose=1,outfile=outdir+'/'+wafer+'_150GHz'+tausuffix+'.'+ext);
                pass;
            pass;
    pass;
