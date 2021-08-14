#!/bin/python

import numpy as np;
import sqlite3, pandas;
import copy;
from utils import theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta;
from matplotlib import pyplot as plt;
from lmfit.models import GaussianModel

ver='_ver5';

def check_absolute(outfile='out_check_absolute/check_absolute'+ver):

    # Configure for base selections
    stim_quality_cut = 'tau>0.';
    wg_quality_cut   = 'theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi);
    additional_cut = '';
    #additional_cut = 'band==90';
    #additional_cut = 'band==150';
    base_selection_name = additional_cut.replace('.','').replace('=','');

    base_cuts      = [stim_quality_cut, wg_quality_cut, additional_cut];
    base_cuts      = [ cut for cut in base_cuts if cut!='']; # remove empty cut
    base_selection = '&'.join(base_cuts);

    outfile =outfile+ ('_' if base_selection_name!='' else '') + base_selection_name;


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
    df_all['theta_det_angle'] = theta0topi(df_all['theta_det']-np.pi/2., upper=180.*np.pi/180.)*convertF;
    # Add diff. between theta_det_angle and pol_angle [-90., 90]
    df_all['diff_angle']      = deg90to90(df_all['theta_det_angle'] - df_all['pol_angle']);


    # Create DB after selections

    # Base DB
    print('base selection = {}'.format(base_selection));
    if base_selection!='' : df_base = df_all.query(base_selection);

    # DB with correct label originally (before label correction)
    df_notmislabel = df_base.query('mislabel==False');

    # DB of outliers in angles (possible mis-label) (|diff.| > 45 deg.)
    #df_angle_outlier = df_base[diff_angle(deg_to_rad(df_base['pol_angle']),deg_to_rad(df_base['theta_det_angle']),upper90deg=True)>=np.pi/4.];
    bools_angle_outlier = np.abs(df_base['diff_angle']) >= 45.;
    print( '*** booleans for angle outliers (|diff.| > 45 deg.) ***');
    print( bools_angle_outlier );
    print( '*******************************************************');
    df_angle_outlier = df_base[bools_angle_outlier];
    df_angle_outlier.to_csv(outfile+'.csv');

    # DB with offset (x,y)
    #print('df_base offset(x,y)', df_base[['det_offset_x','det_offset_y']]);
    df_offset = df_base.dropna(subset=['det_offset_x','det_offset_y']) # drop Nan in det_offset_x/y
    # Additional quality cut: remove outliers in angles (possible mis-label)
    df_offset = df_offset[abs(df_offset['diff_angle'])<20.] 

    pandas.set_option('display.max_columns', 50)
    print('************df_base*********************');
    print(df_base.head());
    print('************df_angle_outlier*************');
    print(df_angle_outlier.head());
    print('****************************************');
    pandas.set_option('display.max_columns', 5)
    print('Total number of bolos after base selection = {}'.format(len(df_base)));
    print('Total number of bolos after base selection2 (mislabel==False) = {}'.format(len(df_notmislabel)));
    print('Total number of outliers after base selection = {}'.format(len(df_angle_outlier)));


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

        y = df_select['diff_angle'];
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


    ####################
    # Pol. angle plots #
    ####################

    Nrow    = 4;
    Ncolumn = 3;
    fig, axs = plt.subplots(Nrow,Ncolumn);
    fig.set_size_inches(6*Ncolumn,6*Nrow);
    #fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    # 2D plot 1
    axs[0][0].plot(df_notmislabel[dataname2], df_notmislabel[dataname],marker='o', markersize=1.0, linestyle='');
    axs[0][0].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][0].grid(True);
    axs[0][0].set_title('Bolos with correct labels originally (no mis-label)');
    axs[0][0].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle" in focalplane database)',fontsize=16);
    axs[0][0].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ - 90 [deg.]',fontsize=16);
    axs[0][0].set_xticks(np.arange(-360,360,45));
    axs[0][0].set_yticks(np.arange(-360,360,45));
    axs[0][0].set_xlim(-22.5,180);
    axs[0][0].set_ylim(-22.5,180);
    axs[0][0].tick_params(labelsize=12);

    # 2D plot 2
    axs[0][1].plot(df_base[dataname2], df_base[dataname],marker='o', markersize=1.0, linestyle='', label='all');
    axs[0][1].plot(df_angle_outlier[dataname2], df_angle_outlier[dataname],marker='o', markersize=1.0, linestyle='',color='r', label='outliers');
    axs[0][1].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
    axs[0][1].grid(True);
    axs[0][1].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=1.0);
    axs[0][1].set_title('Bolos with base cuts after label correction (all/outliers)');
    axs[0][1].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle" in focalplane database)',fontsize=16);
    axs[0][1].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ - 90 [deg.]',fontsize=16);
    axs[0][1].set_xticks(np.arange(-360,360,45));
    axs[0][1].set_yticks(np.arange(-360,360,45));
    axs[0][1].set_xlim(-22.5,180);
    axs[0][1].set_ylim(-22.5,180);
    axs[0][1].tick_params(labelsize=12);


    # Diff. angle plot 1
    iselections1=[0,1];
    centers1 = [];
    for i in iselections1 :
        selection = selections[i];
        result = fit_results[i];
        center = result.params['center'];
        sigma  = result.params['sigma'];
        if center.value  is None : center.value  = 0.
        if center.stderr is None : center.stderr = 0.
        if sigma.value  is None : sigma.value  = 0.
        if sigma.stderr is None : sigma.stderr = 0.
        axs[1][0].plot(fit_bins[i][1:], result.best_fit, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$'.format(selection[1],center.value,center.stderr,sigma.value,sigma.stderr));
        centers1.append(center);
        pass;
    __datas = [ data_selects[i] for i in iselections1 ];
    __labels= [ labels[i]       for i in iselections1 ];
    __colors= [ colors[i]       for i in iselections1 ];
    axs[1][0].hist(__datas, bins=nbins, range=xbinrange, histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=__colors, alpha=0.4, label=__labels, stacked=False);
    axs[1][0].set_title(baseselect[1] if len(baseselect)>1 else '');
    axs[1][0].set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=16);
    axs[1][0].set_ylabel(r'# of bolometers',fontsize=16);
    axs[1][0].set_xticks(np.arange(xbinrange[0],xbinrange[1],5));
    axs[1][0].tick_params(labelsize=12);
    axs[1][0].grid(True);
    axs[1][0].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.,labelspacing=1.2);
    center_ave = sum([ center.value for center in centers1])/float(len(centers1));
    center_ave_err = np.sqrt(sum([ center.stderr**2. for center in centers1]))/float(len(centers1));
    axs[1][0].text(-20,20, 'Averaged center:\n {:.2f} $\pm$ {:.2f}'.format(center_ave,center_ave_err), fontsize=10, color='tab:blue');

    # Diff. angle plot 2
    iselections2=[2,3];
    centers2 = [];
    for i in iselections2 :
        selection = selections[i];
        result = fit_results[i];
        center = result.params['center'];
        sigma  = result.params['sigma'];
        if center.value  is None : center.value  = 0.
        if center.stderr is None : center.stderr = 0.
        if sigma.value  is None : sigma.value  = 0.
        if sigma.stderr is None : sigma.stderr = 0.
        axs[1][1].plot(fit_bins[i][1:], result.best_fit, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$'.format(selection[1],center.value,center.stderr,sigma.value,sigma.stderr));
        centers2.append(center);
        pass;
    __datas = [ data_selects[i] for i in iselections2 ];
    __labels= [ labels[i]       for i in iselections2 ];
    __colors= [ colors[i]       for i in iselections2 ];
    axs[1][1].hist(__datas, bins=nbins, range=xbinrange, histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=__colors, alpha=0.4, label=__labels, stacked=False);
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
 

    # Focal plane plot 1
    im0= axs[2][0].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset[dataname], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
    axs[2][0].set_title(dataname +' [deg.]');
    axs[2][0].grid(True);
    axs[2][0].set_xlabel('x offset', fontsize=16);
    axs[2][0].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im0, ax=axs[2][0]);

    # Focal plane plot 2
    im1= axs[2][1].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset[dataname2], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
    axs[2][1].set_title(dataname2+' [deg.]');
    axs[2][1].grid(True);
    axs[2][1].set_xlabel('x offset', fontsize=16);
    axs[2][1].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im1, ax=axs[2][1]);
 
    # Focal plane plot 3
    im2= axs[2][2].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset['diff_angle'], marker='o', s=10, cmap='coolwarm', vmin=-10., vmax=10.);
    axs[2][2].set_title(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
    axs[2][2].grid(True);
    axs[2][2].set_xlabel('x offset', fontsize=16);
    axs[2][2].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im2, ax=axs[2][2]);
    # Calculate offset
    center_offset_x = np.mean(df_offset['det_offset_x']);
    center_offset_y = np.mean(df_offset['det_offset_y']);
    df_offset_R  = df_offset.query('det_offset_x-{0[x0]}>=det_offset_y-{0[y0]}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    df_offset_L  = df_offset.query('det_offset_x-{0[x0]}<det_offset_y-{0[y0]}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    diffangle_mean   = np.mean(df_offset[dataname] - df_offset[dataname2]);
    diffangle_mean_R = np.mean(df_offset_R[dataname] - df_offset_R[dataname2]);
    diffangle_mean_L = np.mean(df_offset_L[dataname] - df_offset_L[dataname2]);
    print('Right bolo selection : det_offset_x-{0[x0]:.2f} >= det_offset_y-{0[y0]:.2f}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    print('Left  bolo selection : det_offset_x-{0[x0]:.2f} <  det_offset_y-{0[y0]:.2f}'.format({'x0':center_offset_x,'y0':center_offset_y}));
    print('diff. angle mean in all   bolos = {}'.format(diffangle_mean));
    print('diff. angle mean in right bolos = {}'.format(diffangle_mean_R));
    print('diff. angle mean in left  bolos = {}'.format(diffangle_mean_L));
    # Add points/lines
    xlim = axs[2][2].get_xlim();
    ylim = [xlim[0]-center_offset_x+center_offset_y, xlim[1]-center_offset_x+center_offset_y];
    axs[2][2].plot([xlim[0],xlim[1]],[ylim[0],ylim[1]],c='k',marker='',linestyle='--',linewidth=1);
    axs[2][2].scatter([center_offset_x],[center_offset_y],c='k',marker='*',s=200);
    axs[2][2].scatter([center_offset_x],[center_offset_y],c='yellow',marker='*',s=50);
    axs[2][2].set_xlim(xlim);
    axs[2][2].set_ylim(ylim);

    # Focal plane plot 4 (band freq. check)
    im3= axs[1][2].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset['band'], marker='o', s=10, cmap='coolwarm', vmin=90., vmax=150.);
    axs[1][2].set_title('Band frequency [GHz]');
    axs[1][2].grid(True);
    axs[1][2].set_xlabel('x offset', fontsize=16);
    axs[1][2].set_ylabel('y offset', fontsize=16);
    fig.colorbar(im3, ax=axs[1][2]);


    # Other histogram 1
    # Polarization offset r, theta
    # select good bolos
    df_base.loc[:,'r0'    ] = np.sqrt(df_base['x0']**2.+df_base['y0']**2.);
    df_base.loc[:,'r0_err'] = np.sqrt(np.power(df_base['x0']*df_base['x0_err'],2.)+np.power(df_base['y0']*df_base['y0_err'],2.))/df_base['r0'];
    #-----
    df_base_center = df_base;
    #df_base_center = df_base.query('theta0_err/theta0<0.1&r0_err/r0<0.1');
    #print('before r0_err, theta0_err selection: # of bolos = {}'.format(len(df_base)));
    #print('after  r0_err, theta0_err selection: # of bolos = {}'.format(len(df_base_center)));
    #-----
    # plot: polarization offset theta
    axs[3][0].hist(df_base_center['r0'], bins=50, range=[0.,500.], histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[0], alpha=1.0, label='r0', stacked=False);
    axs[3][0].set_title('r0 of wire grid cal.');
    axs[3][0].set_xlabel(r'$r_0$ [mK]');
    axs[3][0].set_ylabel(r'# of bolometers');
    axs[3][0].grid(True);

    # Other histogram 2
    # plot: fake-polarization offset (theta0) with respect to theta_wire0
    theta = (df_base_center['theta_wire0']-df_base_center['theta0'])/2.; 
    #theta = rad_to_deg_pitopi(theta) # range [-pi,pi]
    theta = rad_to_deg(theta0topi(theta)); # range [0,pi]
    axs[3][1].hist(theta, bins=90, range=[0.,180.], histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[0], alpha=1.0, label='theta0', stacked=False);
    axs[3][1].set_title('Polarization offset angles in wire angle of wire grid cal.');
    axs[3][1].set_xlabel(r'$\theta_{\mathrm{wire0}}$ [deg.]');
    axs[3][1].set_ylabel(r'# of bolometers');
    axs[3][1].grid(True);

    # Other histogram 3
    # plot: tau
    axs[3][2].hist(df_base['tau']*1000., bins=100, range=[0.,50.], histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[0], alpha=1.0, label='theta0', stacked=False);
    axs[3][2].set_title(r'Time constant $\tau$ measured by stimulator [msec]');
    axs[3][2].set_xlabel(r'Time constant $\tau$ [msec]');
    axs[3][2].set_ylabel(r'# of bolometers');
    axs[3][2].grid(True);



    # Save fig
    print('savefig to '+outfile+'.png');
    fig.savefig(outfile+'.png');



    ########################
    # each wire angle plot #
    ########################

    # Define columns names
    wire_angles = [0,22.5,45.,67.5,90.,112.5,135.,157.5]; # wire_angles[0] will be used as reference point.
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
    columns = df_all.columns.values;

    # Check columns names
    column_ok = True;
    for name in columns_x + columns_x_err + columns_y + columns_y_err :
        if not name in columns : 
            print('Error! There is no {}'.format(name));
            column_ok = False;
        pass;

    # calculate r for each x,y
    columns_r       = [];
    columns_r_err   = [];
    columns_r_ratio = [];
    columns_theta      = [];
    columns_theta_err  = [];
    columns_theta_diff = [];
    alabels = [];
    x0 = df_base['x0'];
    y0 = df_base['y0'];
    for n, angle in enumerate(wire_angles) :
        x     = df_base[columns_x[n]    ] - x0;
        x_err = df_base[columns_x_err[n]];
        y     = df_base[columns_y[n]    ] - y0;
        y_err = df_base[columns_y_err[n]];
        r0    = df_base['r']; # r for circle
        (r, r_err), (theta, theta_err)= calculateRTheta(x, y ,x_err, y_err); # (r, theta) for each wire angle point
        r_ratio    = r/r0;
        if n==0: theta0 = theta;
        theta = rad_to_deg_0to2pi(theta0 - theta);
        theta_diff = theta - angle*2.;
        r_column       = 'r_{}deg'.format(angle);
        r_err_column   = 'r_err_{}deg'.format(angle);
        r_ratio_column = 'r_ratio_{}deg'.format(angle);
        theta_column      = 'theta_{}deg'.format(angle);
        theta_err_column  = 'theta_err_{}deg'.format(angle);
        theta_diff_column = 'theta_diff_{}deg'.format(angle);
        columns_r      .append(r_column      );
        columns_r_err  .append(r_err_column  );
        columns_r_ratio.append(r_ratio_column);
        columns_theta     .append(theta_column     );
        columns_theta_err .append(theta_err_column );
        columns_theta_diff.append(theta_diff_column);
        df_base.loc[:,r_column      ] = r      ;
        df_base.loc[:,r_err_column  ] = r_err  ;
        df_base.loc[:,r_ratio_column] = r_ratio;
        df_base.loc[:,theta_column     ] = theta     ;
        df_base.loc[:,theta_err_column ] = theta_err ;
        df_base.loc[:,theta_diff_column] = theta_diff;
        alabels.append('wire={} deg'.format(angle));
        pass;
 

    # Making plots...
    if column_ok :
        print('Found all the columns for each wire angles');

        selection_sets = [
                {'outname':'', 'sels':[''], 'labels':['']},
                {'outname':'_pixel-handed', 'sels':["pixel_handedness=='A'", "pixel_handedness=='B'"], 'labels':['pixel A', 'pixel B'     ]},
                {'outname':'_bolo-TB'     , 'sels':["bolo_type=='T'"       , "bolo_type=='B'"       ], 'labels':['Top bolo', 'Bottom bolo']},
                {'outname':'_pixel-UQ'    , 'sels':["pixel_type=='U'"      , "pixel_type=='Q'"      ], 'labels':['U pixel' , 'Q pixel'    ]},
                {'outname':'_band'        , 'sels':["band==90"             , "band==150"            ], 'labels':['90 GHz'  , '150 GHz'    ]},
                {'outname':'_polangle0-45-90'  , 'sels':["pol_angle==0" , "pol_angle==45", "pol_angle==90" ], 'labels':['0 deg'  , '45 deg', '90 deg' ]},
                {'outname':'_polangle15-60-105', 'sels':["pol_angle==15", "pol_angle==60", "pol_angle==105"], 'labels':['15 deg' , '60 deg', '105 deg']},
                {'outname':'_polangle135-150'  , 'sels':["pol_angle==135", "pol_angle==150"], 'labels':['135 deg', '150 deg']},
                {'outname':'_tau10', 'sels':['tau*1000.<10.', 'tau*1000.>=10.'], 'labels':['tau>10 msec', 'tau<10 msec']},
                {'outname':'_FPpos', 'sels':['det_offset_x<det_offset_y', 'det_offset_x>=det_offset_y'], 'labels':['Right top in FP', 'Left bottom in FP']},
                ];
        
        
        for selection_set in selection_sets :

            sels    = selection_set['sels'];
            slabels = selection_set['labels'];
            nsel = len(sels);
            dfs = [df_base.query(sel) if sel!='' else df_base for sel in sels];

            # calculate r for each x,y
            r_set = [];
            r_err_set = [];
            r_ratio_set = [];
            r_ratio_mean_set = [];
            r_ratio_std_set = [];
            theta_set = [];
            theta_err_set = [];
            theta_diff_set = [];
            theta_diff_mean_set = [];
            theta_diff_std_set = [];
            for df in dfs :
                rs = [];
                r_errs = [];   
                r_ratios = [];
                r_ratio_means = [];
                r_ratio_stds  = [];
                thetas = [];
                theta_errs = [];
                theta_diffs = [];
                theta_diff_means = [];
                theta_diff_stds = [];
                for n, angle in enumerate(wire_angles) :
                    r       = df[columns_r[n]      ];
                    r_err   = df[columns_r_err[n]  ];
                    r_ratio = df[columns_r_ratio[n]];
                    theta      = df[columns_theta[n]     ];
                    theta_err  = df[columns_theta_err[n] ];
                    theta_diff = df[columns_theta_diff[n]];
                    rs.append(r);
                    r_errs.append(r_err);
                    r_ratios.append(r_ratio);
                    r_ratio_means.append(np.mean(r_ratio));
                    r_ratio_stds .append(np.std(r_ratio));
                    thetas.append(theta);
                    theta_errs.append(theta_err);
                    theta_diffs.append(theta_diff);
                    theta_diff_means.append(np.mean(theta_diff));
                    theta_diff_stds .append(np.std(theta_diff));
                    pass;
                r_set.append(rs);
                r_err_set.append(r_errs);
                r_ratio_set.append(r_ratios);
                r_ratio_mean_set.append(r_ratio_means);
                r_ratio_std_set.append(r_ratio_stds);
                theta_set.append(thetas);
                theta_err_set.append(theta_errs);
                theta_diff_set.append(theta_diffs);
                theta_diff_mean_set.append(theta_diff_means);
                theta_diff_std_set.append(theta_diff_stds);
                pass;


            #########################
            # r for each wire angle #
            #########################

            i_figs = 3;
            j_figs = 4;
            fig2, axs2 = plt.subplots(i_figs,j_figs);
            fig2.set_size_inches(6*j_figs,6*i_figs);
            #fig2.tight_layout(rect=[0,0,1,1]);
            plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)
         
            # All histograms of r/r0 (r_ratio) for each wire angles
            stacked=False;
            for s in range(nsel) :
                axs2[0][s].hist(r_ratio_set[s], bins=100, range=(0.8,1.2), histtype='stepfilled',
                    align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                    color=colors[0:len(r_ratio_set[s])], alpha=0.4, label=alabels, stacked=stacked);
                axs2[0][s].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
                axs2[0][s].grid(True);
                axs2[0][s].set_title(r'Signal power ($r$) ratio for {}'.format(slabels[s])+ ( '(Stacked)' if stacked else '(Not stacked)'));
                axs2[0][s].set_xlabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$ for each angle',fontsize=16);
                axs2[0][s].set_ylabel(r'# of bolometers',fontsize=16);
                pass;
              
            # Plots of wire angle v.s. r/r0 mean
            for s in range(nsel) :
                axs2[0][nsel].errorbar(wire_angles, r_ratio_mean_set[s], yerr=r_ratio_std_set[s], c=colors[s], marker='o', markersize=3.,capsize=2.,linestyle='',label=slabels[s]);
                pass;
            axs2[0][nsel].grid(True);
            axs2[0][nsel].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
            axs2[0][nsel].set_title(r'Signal power ($r$) ratio for each wire angles');
            axs2[0][nsel].set_xlabel(r'Wire angle [deg.]',fontsize=16);
            axs2[0][nsel].set_ylabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}} \pm $ std.',fontsize=16);
         
         
            # Each histogram of r/r0 (r_ratio) for each wire angles
            for n, angle in enumerate(wire_angles) :
                i = (int)(n/j_figs) +1;
                j = n%j_figs;

                #r_ratio_set_nangle = [ r_ratios[n] for r_ratios in r_ratio_set ];
                __r_ratio_set_nangle = np.array(r_ratio_set)[:,n];
                __colors = colors[n*nsel:n*nsel+nsel];
                __labels = slabels;
                # if only one dataset
                if len(__r_ratio_set_nangle)==1:
                    __datas    = __r_ratio_set_nangle[0];
                    __colors   = __colors[0];
                    __labels   = __labels[0];
                else :
                    __datas  = __r_ratio_set_nangle;
                    pass;
                print(__r_ratio_set_nangle);
                #print('colors =', __colors);
         
                __axs2 = axs2[i][j];
                __axs2.hist(__datas, bins=100, range=(0.8,1.2), histtype='stepfilled',
                    align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                    color=__colors, alpha=0.4, label=__labels, stacked=False);
                __axs2.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
                __axs2.grid(True);
                __axs2.set_title(r'Signal power ($r$) ratio: {} deg.'.format(angle));
                __axs2.set_xlabel(r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$ for $\theta_{\mathrm{wire}}=$'+'{} deg.'.format(angle),fontsize=16);
                __axs2.set_ylabel(r'# of bolometers',fontsize=16);
                __xlim = __axs2.get_xlim();
                __ylim = __axs2.get_ylim();
                for s, slabel in enumerate(slabels) :
                    __nbolo = len(__r_ratio_set_nangle[s]);
                    print('# of bolos for {} = {}'.format(slabel, __nbolo));
                    __axs2.text(__xlim[0]+0.05,__ylim[1]*(0.35-0.05*s), 'Mean for {:10s} = {:.3f} +- {:.3f} (std.) ({} bolos)'.format(slabel, r_ratio_mean_set[s][n], r_ratio_std_set[s][n], __nbolo), 
                            fontsize=10, color='tab:blue');
                    pass;
                pass;
         
         
            print('savefig to '+outfile+'_eachwire-r{}.png'.format(selection_set['outname']));
            fig2.savefig(outfile+'_eachwire-r{}.png'.format(selection_set['outname']));


            #############################
            # theta for each wire angle #
            #############################

            i_figs = 3;
            j_figs = 4;
            fig2, axs2 = plt.subplots(i_figs,j_figs);
            fig2.set_size_inches(6*j_figs,6*i_figs);
            #fig2.tight_layout(rect=[0,0,1,1]);
            plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)
         
            # All histograms of theta - theta0 - angle*2 (theta_diff from expected one) for each wire angles
            stacked=False;
            for s in range(nsel) :
                axs2[0][s].hist(theta_diff_set[s], bins=100, range=(-10,10), histtype='stepfilled',
                    align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                    color=colors[0:len(theta_diff_set[s])], alpha=0.4, label=alabels, stacked=stacked);
                axs2[0][s].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
                axs2[0][s].grid(True);
                axs2[0][s].set_title(r'Theta diff. from expected one for {}'.format(slabels[s])+ ( '(Stacked)' if stacked else '(Not stacked)'));
                axs2[0][s].set_xlabel(r'$\theta_{\mathrm{meas.}} - \theta_{\mathrm{exp.}}$ for each angle [deg.]',fontsize=16);
                axs2[0][s].set_ylabel(r'# of bolometers',fontsize=16);
                pass;
              
            # Plots of wire angle v.s. theta_diff mean
            for s in range(nsel) :
                axs2[0][nsel].errorbar(wire_angles, theta_diff_mean_set[s], yerr=theta_diff_std_set[s], c=colors[s], marker='o', markersize=3.,capsize=2.,linestyle='',label=slabels[s]);
                pass;
            axs2[0][nsel].grid(True);
            axs2[0][nsel].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
            axs2[0][nsel].set_title(r'Theta diff. from expected one for {}'.format(slabels[s])+ ( '(Stacked)' if stacked else '(Not stacked)'));
            axs2[0][nsel].set_xlabel(r'Wire angle [deg.]',fontsize=16);
            axs2[0][nsel].set_ylabel(r'$\theta_{\mathrm{meas.}} - \theta_{\mathrm{exp.}}$ for each angle [deg.]',fontsize=16);
         
         
            # Each histogram of r/r0 (theta_diff) for each wire angles
            for n, angle in enumerate(wire_angles) :
                i = (int)(n/j_figs) +1;
                j = n%j_figs;

                #theta_diff_set_nangle = [ theta_diffs[n] for theta_diffs in theta_diff_set ];
                __theta_diff_set_nangle = np.array(theta_diff_set)[:,n];
                __colors = colors[n*nsel:n*nsel+nsel];
                __labels = slabels;
                # if only one dataset
                if len(__theta_diff_set_nangle)==1:
                    __datas    = __theta_diff_set_nangle[0];
                    __colors   = __colors[0];
                    __labels   = __labels[0];
                else :
                    __datas  = __theta_diff_set_nangle;
                    pass;
                print(__theta_diff_set_nangle);
                #print('colors =', __colors);
         
                __axs2 = axs2[i][j];
                __axs2.hist(__datas, bins=100, range=(-10,10), histtype='stepfilled',
                    align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                    color=__colors, alpha=0.4, label=__labels, stacked=False);
                __axs2.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.,labelspacing=1.0);
                __axs2.grid(True);
                __axs2.set_title(r'Theta diff. from expected one: {} deg'.format(angle));
                __axs2.set_xlabel(r'$\theta_{\mathrm{meas.}} - \theta_{\mathrm{exp.}}$' + ' for {} deg. wire [deg.]'.format(angle),fontsize=16);
                __axs2.set_ylabel(r'# of bolometers',fontsize=16);
                __xlim = __axs2.get_xlim();
                __ylim = __axs2.get_ylim();
                for s, slabel in enumerate(slabels) :
                    __nbolo = len(__theta_diff_set_nangle[s]);
                    print('# of bolos for {} = {}'.format(slabel, __nbolo));
                    __axs2.text(__xlim[0]+1.,__ylim[1]*(0.3-0.05*s), 'Mean for {:10s} = {:.3f} +- {:.3f} (std.) ({} bolos)'.format(slabel, theta_diff_mean_set[s][n], theta_diff_std_set[s][n], __nbolo), 
                            fontsize=10, color='tab:blue');
                    pass;
                pass;
         
         
            print('savefig to '+outfile+'_eachwire-theta{}.png'.format(selection_set['outname']));
            fig2.savefig(outfile+'_eachwire-theta{}.png'.format(selection_set['outname']));

            pass;
        pass;

    return 0;


if __name__=='__main__' :
    check_absolute();