#!/bin/python

import numpy as np;
import sqlite3, pandas;
import copy;
from utils import deg0to180, theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta;
from matplotlib import pyplot as plt;
from matplotlib import cm as cm;
from lmfit.models import GaussianModel

ver='_ver9';
isCorrectHWPenc=True;


def plotEachWire(dfs, var_set, var_mean_set, var_std_set, slabels, 
        wire_angles, alabels,
        histbins=100, histxrange=(0.9,1.1),
        title=r'Signal power ($r$) ratio', 
        vartitle=r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$',
        mean_ref = 1., mean_yrange = (0.97,1.03),
        outfilename = 'aho.png',
        fp_width = 3.0, colors = colors,
        ) :

    nsel = len(dfs);

    i_figs = 4;
    j_figs = 4;
    fig2, axs2 = plt.subplots(i_figs,j_figs);
    fig2.set_size_inches(6*j_figs,6*i_figs);
    #fig2.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)
 
    stacked=False;
    for s in range(nsel) :
        axs2[0][s].hist(var_set[s], bins=histbins, range=histxrange, histtype='stepfilled',
            align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
            color=colors[0:len(var_set[s])], alpha=0.4, label=alabels, stacked=stacked);
        axs2[0][s].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
        axs2[0][s].grid(True);
        axs2[0][s].set_title(title+'for {}'.format(slabels[s])+ ( '(Stacked)' if stacked else '(Not stacked)'));
        axs2[0][s].set_xlabel(vartitle,fontsize=16);
        axs2[0][s].set_ylabel(r'# of bolometers',fontsize=16);
        pass;
      
    # Plots of wire angle v.s. var mean
    for s in range(nsel) :
        axs2[0][nsel].errorbar(wire_angles, var_mean_set[s], yerr=var_std_set[s], c=colors[s], marker='o', markersize=3.,capsize=2.,linestyle='',label=slabels[s]);
        pass;
    axs2[0][nsel].plot([-1000,1000],[mean_ref,mean_ref],marker='', linestyle='-', linewidth=1, c='k');
    axs2[0][nsel].grid(True);
    axs2[0][nsel].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
    axs2[0][nsel].set_title(title+' for {}'.format(slabels[s]));
    axs2[0][nsel].set_xlabel(r'Wire angle [deg.]',fontsize=16);
    axs2[0][nsel].set_ylabel(vartitle+r' $\pm$ std.',fontsize=16);
    axs2[0][nsel].set_xlim(0.,180.);
    axs2[0][nsel].set_ylim(mean_yrange);
 
 
    # Each histogram for each wire angles
    var_set = np.array(var_set, dtype=object);
    for n, angle in enumerate(wire_angles) :
        i = (int)(n/j_figs) +1;
        j = n%j_figs;

        __var_set_nangle = var_set[:,n];
        __colors = colors[n*nsel:n*nsel+nsel];
        __labels = slabels;
        # if only one dataset
        if len(__var_set_nangle)==1:
            __datas    = __var_set_nangle[0];
            __colors   = __colors[0];
            __labels   = __labels[0];
        else :
            __datas  = __var_set_nangle;
            pass;
        #print(__var_set_nangle);
        #print('colors =', __colors);
 
        __axs2 = axs2[i][j];
        __axs2.hist(__datas, bins=histbins, range=histxrange, histtype='stepfilled',
            align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
            color=__colors, alpha=0.4, label=__labels, stacked=False);
        __axs2.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
        __axs2.grid(True);
        __axs2.set_title(title+': {} deg.'.format(angle));
        __axs2.set_xlabel(vartitle,fontsize=16);
        __axs2.set_ylabel(r'# of bolometers',fontsize=16);
        __xlim = __axs2.get_xlim();
        __ylim = __axs2.get_ylim();
        for s, slabel in enumerate(slabels) :
            __nbolo = len(__var_set_nangle[s]);
            #print('# of bolos for {} = {}'.format(slabel, __nbolo));
            __axs2.text(__xlim[0]+(__xlim[1]-__xlim[0])*0.05,__ylim[1]*(0.25-0.05*s), 
                    'Mean for {:15s}({:4d} bolos) = {:.2f} +- {:.2f} (std.)'.format(slabel, __nbolo, var_mean_set[s][n], var_std_set[s][n]), 
                    fontsize=8, color='tab:blue');
            pass;
        pass;

    # Plot FP position for each selections
    for s in range(nsel) :
        __df = dfs[s];
        axs2[3][0].scatter(__df['det_offset_x'], __df['det_offset_y'], marker='o', s=10, c=colors[s], label=slabels[s]);
        axs2[3][0].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
        axs2[3][0].grid(True);
        axs2[3][0].set_title(r'FP position for each selections');
        axs2[3][0].set_xlabel('x offset', fontsize=16);
        axs2[3][0].set_ylabel('y offset', fontsize=16);
        axs2[3][0].set_xlim([-fp_width,fp_width]);
        axs2[3][0].set_ylim([-fp_width,fp_width]);
        pass;
 
 
    print('savefig to '+outfilename);
    fig2.savefig(outfilename);

    return 0;


def drawAngleHist(ax, iselections, selections, fit_models, fit_results, nzerobins, xbinrange, baseselect, data_selects, labels, nbins, showText=True, zeroCenter=False) :
    # Get shift value
    if zeroCenter :
        shift = [ res.params['center'].value if res is not None else 0. for res in fit_results ];
    else :
        shift = [ 0. for res in fit_results ];
        pass;
    # Draw data
    __datas = [ data_selects[i] for i in iselections ];
    __labels= [ labels[i]       for i in iselections ];
    __colors= [ colors[i]       for i in iselections ];
    ax.hist(__datas, bins=nbins, range=xbinrange, histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=__colors, alpha=0.4, label=__labels, stacked=False);
    # Draw fitted func.
    centers = [];
    xlim = ax.get_xlim();
    for i in iselections :
        selection = selections[i];
        func   = fit_models[i].func;
        result = fit_results[i];
        if result is None: 
            continue;
        center = result.params['center'];
        sigma  = result.params['sigma'];
        chi    = result.chisqr;
        nfree  = result.nfree - nzerobins[i];
        redchi = chi / nfree;
        if center.value  is None : center.value  = 0.
        if center.stderr is None : center.stderr = 0.
        if sigma.value  is None : sigma.value  = 0.
        if sigma.stderr is None : sigma.stderr = 0.
        if redchi is None : redchi = 0.;
        if chi    is None : chi = 0.;
        if nfree  is None : nfree = 0.;
        x = np.linspace(xlim[0],xlim[1],1000);
        y = func(x+shift[i], result.params['amplitude'].value, result.params['center'].value, result.params['sigma'].value );
        ax.plot(x, y, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$\n  $\chi^2/n={:.1f}/{:.1f}={:.1f}$'.format(selection[1],0 if zeroCenter else center.value,center.stderr,sigma.value,sigma.stderr,chi,nfree,redchi));
        centers.append(center);
        pass;
    if len(centers)==0: return -1; # No fit result
    ax.set_title(baseselect[1] if len(baseselect)>1 else '');
    ax.set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=16);
    ax.set_ylabel(r'# of bolometers',fontsize=16);
    ax.set_xticks(np.arange(xbinrange[0],xbinrange[1],5));
    ax.tick_params(labelsize=12);
    ax.grid(True);
    if showText: ax.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.,labelspacing=1.2);
    center_ave = sum([ center.value for center in centers])/float(len(centers));
    center_ave_err = np.sqrt(sum([ center.stderr**2. for center in centers]))/float(len(centers));
    xlim = ax.get_xlim();
    ylim = ax.get_ylim();
    if showText and not zeroCenter: ax.text(xlim[0]+(xlim[1]-xlim[0])*0.05,ylim[0]+(ylim[1]-ylim[0])*0.3,
            'Averaged center:\n {:.2f} $\pm$ {:.2f}'.format(center_ave,center_ave_err), fontsize=10, color='tab:blue');
    return 0;




def check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute'+ver, additional_cut=''):

    # Configure for base selections
    stim_quality_cut = 'tau>0.';
    wg_quality_cut   = 'theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi);
    #base_selection_name = additional_cut.replace('.','').replace('=','');

    base_cuts      = [stim_quality_cut, wg_quality_cut, additional_cut];
    base_cuts      = [ cut for cut in base_cuts if cut!='']; # remove empty cut
    base_selection = '&'.join(base_cuts);

    #outfile =outfile+ ('_' if base_selection_name!='' else '') + base_selection_name;


    # Configure for inputs
    database_wiregrid = 'output{}/db/all_pandas_correct_label.db'.format(ver);
    tablename_wiregrid = 'wiregrid';
    #columns_wiregrid   = 'readout_name,theta_det,theta_det_err,tau,tauerr';
    columns_wiregrid   = '*';
    #database_mapping = 'data/ykyohei/mapping/pb2a_mapping_postv2.db';
    #database_mapping = 'data/ykyohei/mapping/pb2a_mapping_postv1.db';
    #database_mapping = 'data/ykyohei/mapping/pb2a_mapping_210510.db';
    database_mapping = None;
    tablename_mapping = 'pb2a_focalplane';
    selection_mapping = "hardware_map_commit_hash='6f306f8261c2be68bc167e2375ddefdec1b247a2'";
    columns_mapping   = 'readout_name,pol_angle,pixel_type';

    # Plot cosmetics
    fp_width = 3.0;


    # Get pandas
    conn_w = sqlite3.connect(database_wiregrid);
    df_w=pandas.read_sql_query('SELECT {} FROM {}'.format(columns_wiregrid, tablename_wiregrid), conn_w);
    df_m=None;
    if not database_mapping is None :
        conn_m = sqlite3.connect(database_mapping);
        df_m=pandas.read_sql_query('SELECT {} FROM {} {}'.format(
            columns_mapping,tablename_mapping, ('where '+selection_mapping) if not selection_mapping=='' else ''  )
            , conn_m);
        df_m = df_m.rename(columns={'pol_angle':'pol_angle_calibed'});
        df_m = df_m.rename(columns={'pixel_type':'pixel_type_correct'});
        pass;


    # Get merged DB
    if df_m is not None : df_all = pandas.merge(df_w, df_m, how='left', on='readout_name');
    else                : df_all = df_w;
    bools = df_all['pixel_type'].isnull() | df_all['pixel_type']=='';
    print('Total number of df_all = {}'.format(len(df_all)))
    print("booleans for pixel_type.isnull() or '' in df_all = {}".format(bools))
    print('# of null in df_all = {}'.format(sum(bools)))
    #df_all = df_all[~bools]
    #print('Total number of df_all after removing null = {}'.format(len(df_all)))


    # Add variables
    # Add theta_det_angle column for comparison with pol_angle
    convertF = 180./np.pi; # convert from rad --> deg
    if isCorrectHWPenc :
        # -16.71 is obtained from HWP offset angle in out_check_HWPzeroangle/check_HWPzeroangle_ver9.out 
        # -16.708 is obtained from HWP offset angle in out_check_HWPzeroangle/check_HWPzeroangle_ver9_includeMislabel_afterLabelCorr.out 
        # Use -16.5 as default.
        df_all['theta_det_angle'] = theta0topi(df_all['theta_det'] + 2.*deg_to_rad(-16.5), upper=180.*np.pi/180.)*convertF; 
    else :
        df_all['theta_det_angle'] = theta0topi(df_all['theta_det'], upper=180.*np.pi/180.)*convertF;
        pass;
    # Convert pol_angle to the coordinates of theta_det_angle;
    df_all['theta_design_angle'] = deg0to180(df_all['pol_angle']-90., upper=180.);
    # Add diff. between theta_det_angle and pol_angle [-90., 90]
    df_all['diff_angle']      = deg90to90(df_all['theta_det_angle'] - df_all['theta_design_angle']);


    # Create DB after selections

    # Base DB
    print('base selection = {}'.format(base_selection));
    if base_selection!='' : df_base = df_all.query(base_selection);

    # DB with correct label originally (before label correction)
    df_notmislabel = df_base.query('mislabel==False');

    # DB of outliers in angles (possible mis-label) (|diff.| > 45 deg.)
    bools_angle_outlier = np.abs(df_base['diff_angle']) >= 45.;
    '''
    print( '*** booleans for angle outliers (|diff.| > 45 deg.) ***');
    print( bools_angle_outlier );
    print( '*******************************************************');
    '''
    df_angle_outlier = df_base[bools_angle_outlier];
    df_angle_outlier.to_csv(outfile+'.csv');

    # DB with offset (x,y)
    #print('df_base offset(x,y)', df_base[['det_offset_x','det_offset_y']]);
    df_offset = df_base.dropna(subset=['det_offset_x','det_offset_y']) # drop Nan in det_offset_x/y
    # Additional quality cut: remove outliers in angles (possible mis-label)
    df_offset = df_offset[abs(df_offset['diff_angle'])<20.] 

    '''
    pandas.set_option('display.max_columns', 50)
    print('************df_base*********************');
    #print(df_base.head());
    print(df_base['diff_angle']);
    print('************df_angle_outlier*************');
    print(df_angle_outlier.head());
    print('****************************************');
    pandas.set_option('display.max_columns', 5)
    #'''
    print('Total number of bolos after base selection = {}'.format(len(df_base)));
    print('Total number of bolos after base selection2 (mislabel==False) = {}'.format(len(df_notmislabel)));
    print('Total number of outliers after base selection = {}'.format(len(df_angle_outlier)));

    Nrow    = 3;
    Ncolumn = 4;
    abs_fig, abs_axs = plt.subplots(Nrow,Ncolumn);
    abs_fig.set_size_inches(6*Ncolumn,6*Nrow);

    ax = abs_axs[0,0]
    _df = df_base.query("band==90&pixel_handedness=='A'&pol_angle==15&wafer_number=='13.15'");
    ax.hist(_df['diff_angle'])

    
    # Save fig
    print('savefig to '+outfile+'.png');
    abs_fig.savefig(outfile+'.png');

    if additional_plots:
        pass; # End of additional_plots

    return 0;


if __name__=='__main__' :
    check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_tmp'+ver, additional_cut='');
