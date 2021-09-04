#!bin/python
import os, sys;
import numpy as np;
import sqlite3, pandas;
import copy;
from utils import theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta, deg0to180;
from matplotlib import pyplot as plt;
from lmfit.models import GaussianModel

ver='_ver9';
ignoreMisLabel=False;

def drawAngleHist(ax, iselections, selections, fit_models, fit_results, xbinrange, baseselect, showText=True) :
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
        center = result.params['center'];
        sigma  = result.params['sigma'];
        redchi = result.redchi;
        chi    = result.chisqr;
        nfree  = result.nfree;
        if center.value  is None : center.value  = 0.
        if center.stderr is None : center.stderr = 0.
        if sigma.value  is None : sigma.value  = 0.
        if sigma.stderr is None : sigma.stderr = 0.
        if redchi is None : redchi = 0.;
        if chi    is None : chi = 0.;
        if nfree  is None : nfree = 0.;
        x = np.linspace(xlim[0],xlim[1],1000);
        y = func(x, result.params['amplitude'].value, result.params['center'].value, result.params['sigma'].value );
        ax.plot(x, y, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$\n  $\chi^2/n={:.1f}/{:.1f}={:.1f}$'.format(selection[1],center.value,center.stderr,sigma.value,sigma.stderr,chi,nfree,redchi));
        centers.append(center);
        pass;
    ax.set_title(baseselect[1] if len(baseselect)>1 else '');
    ax.set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=16);
    ax.set_ylabel(r'# of bolometers',fontsize=16);
    #dxtick = 5*(int)((xbinrange[1]-xbinrange[0])/5) if (xbinrange[1]-xbinrange[0])>=20. else 2*(int)((xbinrange[1]-xbinrange[0])/2);
    #ax.set_xticks(np.arange(xbinrange[0],xbinrange[1],dxtick));
    ax.tick_params(labelsize=12);
    ax.grid(True);
    if showText: ax.legend(mode = 'upper right',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.,labelspacing=1.2);
    center_ave = sum([ center.value for center in centers])/float(len(centers));
    center_ave_err = np.sqrt(sum([ center.stderr**2. for center in centers]))/float(len(centers));
    xlim = ax.get_xlim();
    dxlim= xlim[1]-xlim[0];
    ylim = ax.get_ylim();
    dylim= ylim[1]-ylim[0];
    if showText: ax.text(xlim[0]+dxlim*0.05,ylim[0]+dylim*0.05, 'Averaged center:\n {:.2f} $\pm$ {:.2f}'.format(center_ave,center_ave_err), fontsize=10, color='tab:blue');
    return 0;



def check_HWPzeroangle(outfile='out_check_HWPzeroangle/check_HWPzeroangle'+ver):

    # make output directory
    dirname = os.path.dirname(outfile);
    if not os.path.isdir(dirname): os.makedirs(dirname);

    # Configure for base selections
    stim_quality_cut = 'tau>0.';
    wg_quality_cut   = 'theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi);
    if ignoreMisLabel :
        additional_cut = 'mislabel==0'; # Ignore mislabeled bolometers due to uncompleted label correction
    else :
        additional_cut = ''; 
        pass;
    base_selection_name = additional_cut.replace('.','').replace('=','');

    base_cuts      = [stim_quality_cut, wg_quality_cut, additional_cut];
    base_cuts      = [ cut for cut in base_cuts if cut!='']; # remove empty cut
    base_selection = '&'.join(base_cuts);

    outfile =outfile+ ('_' if base_selection_name!='' else '') + base_selection_name;


    # Configure for inputs
    database_wiregrid = 'output{}/db/all_pandas_correct_label.db'.format(ver);
    tablename_wiregrid = 'wiregrid';
    columns_wiregrid   = '*';
    # Get some variables from another DB
    database_mapping = None; # If this is None, there is no additional DB.
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


    # Add variables
    # Add theta_det_angle column for comparison with design theta_det [deg]
    convertF = 180./np.pi; # convert from rad --> deg
    df_all['theta_det_angle'] = theta0topi(df_all['theta_det'], upper=180.*np.pi/180.)*convertF;
    # Add design theta_det column [deg]
    df_all['theta_det_design_angle'] = deg0to180(df_all['pol_angle']-90., upper=180.);
    # Add diff. between theta_det_angle and theta_det_deisgn_angle [-90., 90] [deg]
    df_all['diff_angle']      = deg90to90(df_all['theta_det_angle'] - df_all['theta_det_design_angle']);
    mean_absdiff = np.mean(np.abs(df_all['diff_angle']));
    # shift the center from [-90,90] --> [0,180]
    if mean_absdiff>80. : 
        df_all['diff_angle'] = deg0to180( df_all['diff_angle'], upper=180. );
        xbinrange = [0,180];
    else :
        xbinrange = [-90,90];
        pass;


    # Create DB after selections

    # Base DB
    print('base selection = {}'.format(base_selection));
    if base_selection!='' : df_base = df_all.query(base_selection);
    else                  : df_base = df_all;

    # DB of outliers in angles (possible mis-label) (|diff-mean(diff)| > 45 deg.)
    #df_angle_outlier = df_base[diff_angle(deg_to_rad(df_base['pol_angle']),deg_to_rad(df_base['theta_det_angle']),upper90deg=True)>=np.pi/4.];
    mean_diff = np.mean(df_base['diff_angle']);
    bools_angle_outlier = np.abs(df_base['diff_angle']-mean_diff) >= 45.;
    print( '*** booleans for angle outliers (|diff.-mean(diff.)| > 45 deg.) ***');
    print( bools_angle_outlier );
    print( '*******************************************************');
    df_angle_outlier = df_base[bools_angle_outlier];
    df_angle_outlier.to_csv(outfile+'.csv');

    # Remove Outliers
    df_base = df_base[~bools_angle_outlier];

    pandas.set_option('display.max_columns', 50)
    '''
    print('************df_base*********************');
    print(df_base.head());
    print('************df_angle_outlier*************');
    print(df_angle_outlier.head());
    print('****************************************');
    '''
    pandas.set_option('display.max_columns', 5)
    print('Total number of bolos after base selection = {}'.format(len(df_base)));
    print('Total number of outliers after base selection = {}'.format(len(df_angle_outlier)));

    mean_diff_angle = np.mean(df_base['diff_angle']);
    print('Mean of diff_angle = {} deg'.format(mean_diff_angle));
    print('HWP offset angle = {} deg'.format(-0.5*(mean_diff_angle)));


    selections = [\
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
    dataname2= 'theta_det_design_angle';

    # histogram setting
    binwidth = 1.;
    nbins = int((xbinrange[1]-xbinrange[0])/binwidth);
    
    n_sels = [];
    fit_models  = [];
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
        bins_center = np.convolve(bins,[0.5,0.5],mode='valid');
        model = GaussianModel()
        params = model.guess(data=histo, x=bins_center);
        bools  = (y>xbinrange[0]) & (y<xbinrange[1]);
        #params['center'].set(value = np.median(y[bools])*2.);
        #params['sigma'].set(value = 2.);
        print('init for center of gauusian in {}th selection = {}'.format(i, params['center'].value));
        print('init for sigma  of gauusian in {}th selection = {}'.format(i, params['sigma'].value));
        printVar(histo);
        printVar(bins_center);
        result = model.fit(data=histo, x=bins_center, params=params)
        #newparams = result.params;
        #result = model.fit(data=histo, x=bins_center, params=newparams)
        print(result.fit_report());
        print('weights = ', result.weights);
        print('sqrt(N) = ', np.sqrt(result.data));
        print('red-chi-square = {}: chi-square={}/Nfree={}'.format(result.redchi, result.chisqr, result.nfree));
        print(result.ci_report());
        fit_models .append(copy.deepcopy(model));
        fit_results.append(copy.deepcopy(result));
        fit_bins   .append(copy.deepcopy(bins_center));
        #del result, bins, histo, model, params;
        pass;
    print('Sum of selected bolos = {}'.format(sum(n_sels)));


    ###################################################
    # Absolute angle plots (Measured - design angles) #
    ###################################################

    Nrow    = 3;
    Ncolumn = 4;
    abs_fig, abs_axs = plt.subplots(Nrow,Ncolumn);
    abs_fig.set_size_inches(6*Ncolumn,6*Nrow);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    # Diff. angle plot for 90GHz
    drawAngleHist(abs_axs[0,0], iselections=[0,1], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);

    # Diff. angle plot for 150GHz
    drawAngleHist(abs_axs[0,1], iselections=[2,3], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);

    # Diff. angle plot for 90GHz A-handed
    drawAngleHist(abs_axs[1,0], iselections=[0], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);
    # Diff. angle plot for 90GHz B-handed
    drawAngleHist(abs_axs[1,1], iselections=[1], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);
    # Diff. angle plot for 150GHz A-handed
    drawAngleHist(abs_axs[1,2], iselections=[2], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);
    # Diff. angle plot for 150GHz B-handed
    drawAngleHist(abs_axs[1,3], iselections=[3], selections=selections, fit_models=fit_models, fit_results=fit_results, xbinrange=xbinrange, baseselect=baseselect);
 
    # Save fig
    print('savefig to '+outfile+'.png');
    abs_fig.savefig(outfile+'.png');

    return 0;


if __name__=='__main__' :
    check_HWPzeroangle();
