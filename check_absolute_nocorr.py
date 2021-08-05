#!/bin/python

import numpy as np;
import sqlite3, pandas;
import copy;
from utils import theta0topi, colors, printVar, rad_to_deg, deg_to_rad, diff_angle;
from matplotlib import pyplot as plt;
from lmfit.models import GaussianModel

def check_absolute(outfile='out_check_absolute/check_absolute_nocorr'):
    database_wiregrid = 'output_ver4/db/all_pandas.db';
    #database_wiregrid = 'output_ver4/db/all_pandas_correct_label.db';
    tablename_wiregrid = 'wiregrid';
    #columns_wiregrid   = 'readout_name,theta_det,theta_det_err,tau,tauerr';
    columns_wiregrid   = '*';
    database_mapping = 'data/ykyohei/mapping/pb2a_mapping_postv2.db';
    #database_mapping = 'data/ykyohei/mapping/pb2a_mapping_postv1.db';
    #database_mapping = 'data/ykyohei/mapping/pb2a_mapping_210510.db';
    #database_mapping = None;
    tablename_mapping = 'pb2a_focalplane';
    selection_mapping = "hardware_map_commit_hash='6f306f8261c2be68bc167e2375ddefdec1b247a2'";
    columns_mapping   = 'readout_name,pol_angle,pixel_type';

    # get pandas
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


    if df_m is not None : dfmerge = pandas.merge(df_w, df_m, how='left', on='readout_name');
    else                : dfmerge = df_w;

    # add theta_det_angle column for comparison with pol_angle
    convertF = 180./np.pi; # convert from rad --> deg
    #dfmerge['theta_det_angle'] = theta0topi(dfmerge['theta_det']-np.pi/2., upper=150.*np.pi/180.)*convertF;
    dfmerge['theta_det_angle'] = theta0topi(dfmerge['theta_det']-np.pi/2., upper=180.*np.pi/180.)*convertF;

    bools = dfmerge['pixel_type_correct'].isnull()
    print('Total number of dfmerge = {}'.format(len(dfmerge)))
    #print("booleans for pixel_type_correct.isnull() in dfmerge = {}".format(bools))
    print('# of null in pixel_type_correct of dfmerge = {}'.format(sum(bools)))
    df_base0 = dfmerge[~bools]
    print('Total number of dfmerge after removing null = {}'.format(len(df_base0)))

    #dfmerge  = df_base0.query('pol_angle>=0.');
    df_base  = df_base0.query('tau>0.&theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi));
    #df_base = df_base0.query('tau>0.&theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi));
    df_base2 = df_base0;
    dfmerge2 = dfmerge.query('tau>0.&theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi));

    df_base_outlier = df_base[diff_angle(deg_to_rad(df_base['pol_angle']),deg_to_rad(df_base['theta_det_angle']),upper90deg=True)>=np.pi/4.];
    df_base_outlier.to_csv(outfile+'.csv');

    pandas.set_option('display.max_columns', 50)
    print('************df_base*********************');
    print(df_base.head());
    print('************df_base_outlier*************');
    print(df_base_outlier);
    print('****************************************');
    pandas.set_option('display.max_columns', 5)
    print('Total number of bolos after base selection = {}'.format(len(df_base)));
    print('Total number of bolos before base selection= {}'.format(len(df_base2)));
    print('Total number of outliers after base selection = {}'.format(len(df_base_outlier)));
    print('Total number of dfmerge2 = {}'.format(len(dfmerge2)));


    selections = [\
        #["bolo_type=='T' & pixel_type=='U' & pixel_handedness=='A'", 'UT/A'],\
        #["bolo_type=='T' & pixel_type=='U' & pixel_handedness=='B'", 'UT/B'],\
        #["bolo_type=='B' & pixel_type=='U' & pixel_handedness=='A'", 'UB/A'],\
        #["bolo_type=='B' & pixel_type=='U' & pixel_handedness=='B'", 'UB/B'],\
        #["bolo_type=='T' & pixel_type=='Q' & pixel_handedness=='A'", 'QT/A'],\
        #["bolo_type=='T' & pixel_type=='Q' & pixel_handedness=='B'", 'QT/B'],\
        #["bolo_type=='B' & pixel_type=='Q' & pixel_handedness=='A'", 'QB/A'],\
        #["bolo_type=='B' & pixel_type=='Q' & pixel_handedness=='B'", 'QB/B'],\
        ["band==90 & pixel_handedness=='A'", '90 GHz A'],\
        ["band==90 & pixel_handedness=='B'", '90 GHz B'],\
        ["band==150 & pixel_handedness=='A'", '150 GHz A'],\
        ["band==150 & pixel_handedness=='B'", '150 GHz B'],\
        ];
    data_selects = [];
    labels = [];
    ndata = len(selections);
    baseselect = [[]];
    dataname = 'theta_det_angle';
    dataname2= 'pol_angle';

    # histogram setting
    binwidth = 0.5;
    xbinrange = [-20,20];
    nbins = int((xbinrange[1]-xbinrange[0])/binwidth);
    #nbins = 360;
    #xbinrange = [-180,180];
    
    #pandas.set_option('display.max_rows', None)
    #print(df_base[dataname]);
    #print(df_base[dataname2]);
    #pandas.set_option('display.max_rows', 10)

    n_sels = [];
    fit_results = [];
    fit_bins    = [];
    for i, selectinfo in enumerate(selections) :
        print('*********** i={}th selection ***************'.format(i));
        selection   = selectinfo[0] + ('' if len(baseselect[0])==0 else ('&' + baseselect[0]));
        selectlabel = selectinfo[1] if len(selectinfo)>1 else selection.strip().replace("'",'').replace('==','=').replace('_type','');
        selections[i][1] = selectlabel;
        labels.append(selectlabel);
        df_select = df_base.query(selection);
        n_sel     = len(df_select);
        print('selection = {}'.format(selection));
        print('    # of bolos = {}'.format(n_sel));

        y = df_select[dataname] - df_select[dataname2];
        data_selects.append(y);
        #print(y);
        n_sels.append(n_sel);
        histo,bins = np.histogram(y,range=xbinrange,bins=nbins, density=False);
        model = GaussianModel()
        params = model.guess(data=histo, x=bins[1:]);
        bools  = (y>xbinrange[0]) & (y<xbinrange[1]);
        #params['center'].set(value = np.median(y[bools])*2.);
        #params['sigma'].set(value = 2.);
        print('init for center of gauusian in {}th selection = {}'.format(i, params['center'].value));
        print('init for sigma  of gauusian in {}th selection = {}'.format(i, params['sigma'].value));
        printVar(histo);
        printVar(bins[1:]);
        result = model.fit(data=histo, x=bins[1:], params=params)
        #newparams = result.params;
        #result = model.fit(data=histo, x=bins[1:], params=newparams)
        print(result.fit_report());
        fit_results.append(copy.deepcopy(result));
        fit_bins   .append(copy.deepcopy(bins));
        #del result, bins, histo, model, params;
        pass;
    print('Sum of selected bolos = {}'.format(sum(n_sels)));


    fig, axs = plt.subplots(2,2);
    fig.set_size_inches(12,12);
    #fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    # 2D plot
    axs[0][0].plot(df_base2[dataname2], df_base2[dataname],marker='o', markersize=1.0, linestyle='');
    axs[0][0].plot(df_base[dataname2], df_base[dataname],marker='o', markersize=1.0, linestyle='', color='r');
    axs[0][0].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].grid(True);
    axs[0][0].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle" in focalplane database)',fontsize=16);
    axs[0][0].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ - 90 [deg.]',fontsize=16);
    axs[0][0].set_xticks(np.arange(-360,360,45));
    axs[0][0].set_yticks(np.arange(-360,360,45));
    axs[0][0].set_xlim(-22.5,180);
    axs[0][0].set_ylim(-22.5,180);
    axs[0][0].tick_params(labelsize=12);

    # 2D plot
    axs[0][1].plot(dfmerge[dataname2], dfmerge[dataname],marker='o', markersize=1.0, linestyle='');
    axs[0][1].plot(dfmerge2[dataname2], dfmerge2[dataname],marker='o', markersize=1.0, linestyle='',color='r');
    axs[0][1].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].grid(True);
    axs[0][1].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle" in focalplane database)',fontsize=16);
    axs[0][1].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ - 90 [deg.]',fontsize=16);
    axs[0][1].set_xticks(np.arange(-360,360,45));
    axs[0][1].set_yticks(np.arange(-360,360,45));
    axs[0][1].set_xlim(-22.5,180);
    axs[0][1].set_ylim(-22.5,180);
    axs[0][1].tick_params(labelsize=12);

    # diff. of angles
    stacked=False;
    logy = False;
    ishows=[0,2,4];
 
    centers = [];
    for i,selection in enumerate(selections[ishows[0]:ishows[1]]) :
        i = i+ishows[0];
        result = fit_results[i];
        center = result.params['center'];
        sigma  = result.params['sigma'];
        axs[1][0].plot(fit_bins[i][1:], result.best_fit, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$'.format(selection[1],center.value,center.stderr,sigma.value,sigma.stderr));
        centers.append(center);
        pass;
    axs[1][0].hist(data_selects[ishows[0]:ishows[1]], bins=nbins, range=xbinrange, histtype='stepfilled',
             align='mid', orientation='vertical', log=logy, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ishows[0]:ishows[1]], alpha=0.4, label=labels[ishows[0]:ishows[1]], stacked=stacked);
    axs[1][0].set_title(baseselect[1] if len(baseselect)>1 else '');
    axs[1][0].set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=16);
    axs[1][0].set_ylabel(r'# of bolometers',fontsize=16);
    axs[1][0].set_xticks(np.arange(xbinrange[0],xbinrange[1],5));
    axs[1][0].tick_params(labelsize=12);
    axs[1][0].grid(True);
    axs[1][0].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.,labelspacing=1.2);
    center_ave = sum([ center.value for center in centers])/float(len(centers));
    center_ave_err = np.sqrt(sum([ center.stderr**2. for center in centers]))/float(len(centers));
    axs[1][0].text(-20,20, 'Averaged center:\n {:.2f} $\pm$ {:.2f}'.format(center_ave,center_ave_err), fontsize=10, color='tab:blue');

    centers2 = [];
    for i,selection in enumerate(selections[ishows[1]:ishows[2]]) :
        i = i+ishows[1];
        result = fit_results[i];
        center = result.params['center'];
        sigma  = result.params['sigma'];
        axs[1][1].plot(fit_bins[i][1:], result.best_fit, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$'.format(selection[1],center.value,center.stderr,sigma.value,sigma.stderr));
        centers2.append(center);
        pass;
    axs[1][1].hist(data_selects[ishows[1]:ishows[2]], bins=nbins, range=xbinrange, histtype='stepfilled',
             align='mid', orientation='vertical', log=logy, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ishows[1]:ishows[2]], alpha=0.4, label=labels[ishows[1]:ishows[2]], stacked=stacked);
    axs[1][1].set_title(baseselect[1] if len(baseselect)>1 else '');
    axs[1][1].set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=16);
    axs[1][1].set_ylabel(r'# of bolometers',fontsize=16);
    axs[1][1].set_xticks(np.arange(xbinrange[0],xbinrange[1],5));
    axs[1][1].tick_params(labelsize=12);
    axs[1][1].grid(True);
    axs[1][1].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.,labelspacing=1.2);
    center2_ave = sum([ center.value for center in centers2])/float(len(centers2));
    center2_ave_err = np.sqrt(sum([ center.stderr**2. for center in centers2]))/float(len(centers2));
    axs[1][1].text(-20,20, 'Averaged center\n {:.2f} $\pm$ {:.2f}'.format(center2_ave,center2_ave_err), fontsize=10, color='tab:blue');
 
 
    fig.savefig(outfile+'.png');
 

    return 0;


if __name__=='__main__' :
    check_absolute();
