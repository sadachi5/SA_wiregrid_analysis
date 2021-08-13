#!/bin/python

import numpy as np;
import sqlite3, pandas;
import copy;
from utils import theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, deg_to_rad, diff_angle;
from matplotlib import pyplot as plt;
from lmfit.models import GaussianModel

ver='_ver5';

def check_absolute(outfile='out_check_absolute/check_absolute'+ver):
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

    #bools = dfmerge['pixel_type_correct'].isnull()
    bools = dfmerge['pixel_type'].isnull() | dfmerge['pixel_type']=='';
    print('Total number of dfmerge = {}'.format(len(dfmerge)))
    print("booleans for pixel_type.isnull() or '' in dfmerge = {}".format(bools))
    print('# of null in dfmerge = {}'.format(sum(bools)))
    dfmerge = dfmerge[~bools]
    print('Total number of dfmerge after removing null = {}'.format(len(dfmerge)))

    # add theta_det_angle column for comparison with pol_angle
    convertF = 180./np.pi; # convert from rad --> deg
    dfmerge['theta_det_angle'] = theta0topi(dfmerge['theta_det']-np.pi/2., upper=150.*np.pi/180.)*convertF;
    #dfmerge['theta_det_angle'] = theta0topi(dfmerge['theta_det']-np.pi/2., upper=180.*np.pi/180.)*convertF;

    #dfmerge  = dfmerge.query('pol_angle>=0.');
    df_base  = dfmerge.query('tau>0.&theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi));
    df_base2 = dfmerge.query('mislabel==False');
    print('df_base', df_base[['det_offset_x','det_offset_y']]);
    df_base3 = df_base.dropna(subset=['det_offset_x','det_offset_y']) # drop Nan in det_offset_x/y
    df_base3 = df_base3[abs(df_base3['theta_det_angle']-df_base3['pol_angle'])<20.] # remove outliers
    print('df_base3', df_base3);

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
    print('Total number of bolos after base selection2 (mislabel==False) = {}'.format(len(df_base2)));
    print('Total number of outliers after base selection = {}'.format(len(df_base_outlier)));


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


    fig, axs = plt.subplots(3,3);
    fig.set_size_inches(18,18);
    #fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    # 2D plot
    #axs[0][0].plot(dfmerge[dataname2], dfmerge[dataname],marker='o', markersize=1.0, linestyle='');
    axs[0][0].plot(df_base2[dataname2], df_base2[dataname],marker='o', markersize=1.0, linestyle='');
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
    axs[0][1].plot(df_base[dataname2], df_base[dataname],marker='o', markersize=1.0, linestyle='');
    axs[0][1].plot(df_base_outlier[dataname2], df_base_outlier[dataname],marker='o', markersize=1.0, linestyle='',color='r');
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
 
    # polarization offset r, theta
    # select good bolos
    df_base['r0'] = np.sqrt(df_base['x0']**2.+df_base['y0']**2.);
    df_base['r0_err'] = np.sqrt(np.power(df_base['x0']*df_base['x0_err'],2.)+np.power(df_base['y0']*df_base['y0_err'],2.))/df_base['r0'];
    df_base_center = df_base.query('theta0_err/theta0<0.1&r0_err/r0<0.1')
    print('before r0_err, theta0_err selection: # of bolos = {}'.format(len(df_base)));
    print('after  r0_err, theta0_err selection: # of bolos = {}'.format(len(df_base_center)));
    # plot: polarization offset theta
    axs[0][2].hist(df_base_center['r0'], bins=50, range=[0.,500.], histtype='stepfilled',
    #axs[0][2].hist(df_base_center['r0_err']/df_base_center['r0'], bins=100, range=[0.,0.1], histtype='stepfilled', # r0_err check
             align='mid', orientation='vertical', log=logy, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[0], alpha=1.0, label='r0', stacked=stacked);
    axs[0][2].set_title('r0 of wire grid calibration');
    axs[0][2].grid(True);
    # plot: polarization offset theta
    # theta_wire0 : 0-degree wire's theta in the demod complex plane [0,2pi]
    # theta0      : theta of center of circle in the demod complex plane [0,2pi]
    theta = (df_base_center['theta_wire0']-df_base_center['theta0'])/2.; 
    #theta = rad_to_deg_pitopi(theta) # range [-pi,pi]
    theta = rad_to_deg(theta0topi(theta)); # range [0,pi]
    axs[1][2].hist(theta, bins=90, range=[0.,180.], histtype='stepfilled',
    #axs[1][2].hist(df_base_center['theta0_err']/df_base_center['theta0'], bins=100, range=[0.,0.1], histtype='stepfilled', # theta0_err check
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[0], alpha=1.0, label='theta0', stacked=stacked);
    axs[1][2].set_title('Polarization offset angles in wire angle of wire grid calibration');
    axs[1][2].grid(True);


    # focal plane plot
    im0= axs[2][0].scatter(df_base3['det_offset_x'], df_base3['det_offset_y'], c=df_base3[dataname], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
    axs[2][0].set_title(dataname);
    axs[2][0].grid(True);
    axs[2][0].set_xlabel('x offset', fontsize=16);
    axs[2][0].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im0, ax=axs[2][0]);

    im1= axs[2][1].scatter(df_base3['det_offset_x'], df_base3['det_offset_y'], c=df_base3[dataname2], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
    axs[2][1].set_title(dataname2);
    axs[2][1].grid(True);
    axs[2][1].set_xlabel('x offset', fontsize=16);
    axs[2][1].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im1, ax=axs[2][1]);
 
    im2= axs[2][2].scatter(df_base3['det_offset_x'], df_base3['det_offset_y'], c=df_base3[dataname] - df_base3[dataname2], marker='o', s=10, cmap='coolwarm', vmin=-10., vmax=10.);
    axs[2][2].set_title(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
    axs[2][2].grid(True);
    axs[2][2].set_xlabel('x offset', fontsize=16);
    axs[2][2].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im2, ax=axs[2][2]);


    center_offset_x = np.mean(df_base3['det_offset_x']);
    center_offset_y = np.mean(df_base3['det_offset_y']);
    df_base3_R  = df_base3.query('det_offset_x-{0[x0]}>=det_offset_y-{0[y0]}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    df_base3_L  = df_base3.query('det_offset_x-{0[x0]}<det_offset_y-{0[y0]}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    diffangle_mean   = np.mean(df_base3[dataname] - df_base3[dataname2]);
    diffangle_mean_R = np.mean(df_base3_R[dataname] - df_base3_R[dataname2]);
    diffangle_mean_L = np.mean(df_base3_L[dataname] - df_base3_L[dataname2]);
    print('Right bolo selection : det_offset_x-{0[x0]:.2f} >= det_offset_y-{0[y0]:.2f}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    print('Left  bolo selection : det_offset_x-{0[x0]:.2f} <  det_offset_y-{0[y0]:.2f}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    print('diff. angle mean in all   bolos = {}'.format(diffangle_mean));
    print('diff. angle mean in right bolos = {}'.format(diffangle_mean_R));
    print('diff. angle mean in left  bolos = {}'.format(diffangle_mean_L));
 
    # add points/lines
    xlim = axs[2][2].get_xlim();
    ylim = [xlim[0]-center_offset_x+center_offset_y, xlim[1]-center_offset_x+center_offset_y];
    axs[2][2].plot([xlim[0],xlim[1]],[ylim[0],ylim[1]],c='k',marker='',linestyle='--',linewidth=1);
    axs[2][2].scatter([center_offset_x],[center_offset_y],c='k',marker='*',s=200);
    axs[2][2].scatter([center_offset_x],[center_offset_y],c='yellow',marker='*',s=50);
    axs[2][2].set_xlim(xlim);
    axs[2][2].set_ylim(ylim);

 
    fig.savefig(outfile+'.png');


    ########################
    # each wire angle info #
    ########################

    # Define columns names
    wire_angles = [0,22.5,45.,67.5,90.,112.5,135.,157.5];
    columns_x     = [];
    columns_x_err = [];
    columns_y     = [];
    columns_y_err = [];
    for angle in wire_angles :
        columns_x    .append('x_{}deg'    .format(angle));
        columns_x_err.append('x_err_{}deg'.format(angle));
        columns_y    .append('y_{}deg'    .format(angle));
        columns_y_err.append('y_err_{}deg'.format(angle));
        pass;

    # Get columns names in database
    columns = dfmerge.columns.values;

    # Check columns names
    column_ok = True;
    for name in columns_x + columns_x_err + columns_y + columns_y_err :
        if not name in columns : 
            print('Error! There is no {}'.format(name));
            column_ok = False;
        pass;

    # Making plots...
    if column_ok :
        print('Found all the columns for each wire angles');

        # calculate r for each x,y
        r_set = [];
        r_ratio_set = [];
        r_err_set = [];
        r_ratio_mean = [];
        r_ratio_std  = [];
        angle_labels = [];
        for n, angle in enumerate(wire_angles) :
            x     = df_base[columns_x[n]    ];
            x_err = df_base[columns_x_err[n]];
            y     = df_base[columns_y[n]    ];
            y_err = df_base[columns_y_err[n]];
            r     = np.sqrt( x**2. + y**2. );
            r_err = np.sqrt(np.power(x*x_err,2.)+np.power(y*y_err,2.))/r;
            df_base['r_{}deg'.format(angle)] = r;
            df_base['r_err_{}deg'.format(angle)] = r_err;
            r_set.append(r);
            r_ratio_set.append(r/df_base['r']);
            r_err_set.append(r_err);
            angle_labels.append('wire={} deg'.format(angle));
            r_ratio_mean.append(np.mean(r/df_base['r']));
            r_ratio_std .append(np.std(r/df_base['r']));
            pass;

        selection_sets = [
                {'outname':'_pixel-handed', 'sel1':"pixel_handedness=='A'", 'sel2':"pixel_handedness=='B'", 'label1':'pixel A' , 'label2':'pixel B'    },
                {'outname':'_band'        , 'sel1':"band==90"             , 'sel2':"band==150"            , 'label1':'90 GHz'  , 'label2':'150 GHz'    },
                {'outname':'_bolo-TB'     , 'sel1':"bolo_type=='T'"       , 'sel2':"bolo_type=='B'"       , 'label1':'Top bolo', 'label2':'Bottom bolo'},
                {'outname':'_pixel-UQ'    , 'sel1':"pixel_type=='U'"      , 'sel2':"pixel_type=='Q'"      , 'label1':'U pixel' , 'label2':'Q pixel'    },
                ];
        
        for selection_set in selection_sets :

            i_figs = 3;
            j_figs = 4;
            fig2, axs2 = plt.subplots(i_figs,j_figs);
            fig2.set_size_inches(6*j_figs,6*i_figs);
            #fig2.tight_layout(rect=[0,0,1,1]);
            plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)
         
            axs2[0][0].hist(r_ratio_set, bins=100, range=(0.8,1.2), histtype='stepfilled',
                align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                color=colors[1:1+len(r_ratio_set)], alpha=0.4, label=angle_labels, stacked=True);
            axs2[0][0].legend();
            axs2[0][0].grid(True);
            axs2[0][0].set_title(r'Signal power ($r$) ratio');
            axs2[0][0].set_xlabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$ for each angle',fontsize=16);
            axs2[0][0].set_ylabel(r'# of bolometers',fontsize=16);
         
            axs2[0][1].hist(r_ratio_set, bins=100, range=(0.8,1.2), histtype='stepfilled',
                align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                color=colors[1:1+len(r_ratio_set)], alpha=0.4, label=angle_labels, stacked=False);
            axs2[0][1].legend();
            axs2[0][1].grid(True);
            axs2[0][1].set_title(r'Signal power ($r$) ratio');
            axs2[0][1].set_xlabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$ for each angle',fontsize=16);
            axs2[0][1].set_ylabel(r'# of bolometers',fontsize=16);
         
            axs2[0][2].errorbar(wire_angles, r_ratio_mean, yerr=r_ratio_std, c='tab:blue', marker='o', markersize=3.,capsize=2.,linestyle='',label=r'');
            axs2[0][2].legend();
            axs2[0][2].grid(True);
            axs2[0][2].set_title(r'Signal power ($r$) ratio for each wire angles');
            axs2[0][2].set_xlabel(r'\theta _{\mathrm{wire}}',fontsize=16);
            axs2[0][2].set_ylabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}} \pm $ std.',fontsize=16);
         
         
            sel1 = selection_set['sel1'];
            sel2 = selection_set['sel2'];
            label1 = selection_set['label1'];
            label2 = selection_set['label2'];
            for n, angle in enumerate(wire_angles) :
                i = (int)(n/j_figs) +1;
                j = n%j_figs;
                print('n={}: angle={}, axs({},{})'.format(n,angle,i,j));
                __r_name = 'r_{}deg'.format(angle);
                __df1 = df_base.query(sel1);
                __df2 = df_base.query(sel2);
                __r_ratio1 = __df1[__r_name]/__df1['r'];
                __r_ratio2 = __df2[__r_name]/__df2['r'];
                #print('# of bolos for {} = {}'.format(sel1, len(__r_ratio1)));
                #print('# of bolos for {} = {}'.format(sel2, len(__r_ratio2)));
         
                __axs2 = axs2[i][j];
                __axs2.hist([__r_ratio1,__r_ratio2], bins=100, range=(0.8,1.2), histtype='stepfilled',
                    align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                    color=[colors[1+n],colors[1+n+1]], alpha=0.4, label=[label1,label2], stacked=False);
                __axs2.legend();
                __axs2.grid(True);
                __axs2.set_title(r'Signal power ($r$) ratio: {} deg.'.format(angle));
                __axs2.set_xlabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$ for $\theta_{\mathrm{wire}}=$'+'{} deg.'.format(angle),fontsize=16);
                __axs2.set_ylabel(r'# of bolometers',fontsize=16);
                __axs2.text
                xlim = __axs2.get_xlim();
                ylim = __axs2.get_ylim();
                __axs2.text(xlim[0]+0.05,ylim[1]*0.30, 'Total # of bolos. = {}'.format(len(r_ratio_set[n])), fontsize=10, color='tab:blue');
                __axs2.text(xlim[0]+0.05,ylim[1]*0.25, 'Mean for all = {:.3f} +- {:.3f} (std.)'.format(r_ratio_mean[n], r_ratio_std[n]), fontsize=10, color='tab:blue');
                __axs2.text(xlim[0]+0.05,ylim[1]*0.20, 'Mean for A   = {:.3f} +- {:.3f} (std.)'.format(np.mean(__r_ratio1),np.std(__r_ratio1)), fontsize=10, color='tab:blue');
                __axs2.text(xlim[0]+0.05,ylim[1]*0.15, 'Mean for B   = {:.3f} +- {:.3f} (std.)'.format(np.mean(__r_ratio2),np.std(__r_ratio2)), fontsize=10, color='tab:blue');
                pass;
         
         
            fig2.savefig(outfile+'_eachwire{}.png'.format(selection_set['outname']));
            pass;
        pass;

    return 0;


if __name__=='__main__' :
    check_absolute();
