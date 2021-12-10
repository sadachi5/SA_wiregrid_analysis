#!/bin/python

import numpy as np;
import sqlite3, pandas;
import copy;
from utils import deg0to180, theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta;
from matplotlib import pyplot as plt;
from matplotlib import cm as cm;
from lmfit.models import GaussianModel
#from scipy.stats import poisson

ver='_ver10';
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


def drawAngleHist(ax, iselections, selections, fit_models, fit_results, means, stds, nzerobins, xbinrange, baseselect, data_selects, labels, nbins, showText=True, zeroCenter=False, nogauss=False) :
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
    hist, bins, patches = ax.hist(__datas, bins=nbins, range=xbinrange, histtype='stepfilled',
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
        if center.value  is None : center.value  = 0.;
        if center.stderr is None : center.stderr = 0.;
        if sigma.value  is None : sigma.value  = 0.;
        if sigma.stderr is None : sigma.stderr = 0.;
        if redchi is None : redchi = 0.;
        if chi    is None : chi = 0.;
        if nfree  is None : nfree = 0.;
        x = np.linspace(xlim[0],xlim[1],1000);
        y = func(x+shift[i], result.params['amplitude'].value, result.params['center'].value, result.params['sigma'].value );
        if not nogauss: ax.plot(x, y, color=colors[i], linestyle='-', label='Fit result for {}:\n  Center = ${:.2f} \pm {:.2f}$\n  $\sigma = {:.2f} \pm {:.2f}$\n  $\chi^2/n={:.1f}/{:.1f}={:.1f}$'.format(selection[1],0 if zeroCenter else center.value,center.stderr,sigma.value,sigma.stderr,chi,nfree,redchi));
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
    means = np.array(means)[iselections]; # retrieve mean in iselections
    stds = np.array(stds)[iselections]; # retrieve std in iselections
    print(f'means = {means}, stds = {stds}');
    mean_ave = np.mean(means);
    std_ave  = np.sqrt(np.sum(stds*stds)/float(len(stds)));
    print(f'mean_ave = {mean_ave}, stds = {std_ave}');
    ax.set_ylim(-1., max(hist)*1.5)
    xlim = ax.get_xlim();
    ylim = ax.get_ylim();
    if showText and not zeroCenter and not nogauss: ax.text(xlim[0]+(xlim[1]-xlim[0])*0.05,ylim[0]+(ylim[1]-ylim[0])*0.3,
            'Fitted center:\n {:.2f} $\pm$ {:.2f}'.format(center_ave,center_ave_err), fontsize=10, color='tab:blue');
    if showText : ax.text(xlim[0]+(xlim[1]-xlim[0])*0.05,ylim[0]+(ylim[1]-ylim[0])*0.2,
            'Mean:\n {:.2f} $\pm$ {:.2f}'.format(mean_ave,std_ave), fontsize=10, color='tab:blue');
    return 0;




def check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute'+ver, additional_cut='', selectiontype=0, nogauss=False):

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

    # Additional quality cut: remove outliers in angles (possible mis-label)
    df_base = df_base[abs(df_base['diff_angle'])<10.]  # 10deg is Tight cut (default: 20deg)

    # DB with offset (x,y)
    #print('df_base offset(x,y)', df_base[['det_offset_x','det_offset_y']]);
    df_offset = df_base.dropna(subset=['det_offset_x','det_offset_y']) # drop Nan in det_offset_x/y

    '''
    pandas.set_option('display.max_columns', 50)
    print('************df_base*********************');
    print(df_base.head());
    print('************df_angle_outlier*************');
    print(df_angle_outlier.head());
    print('****************************************');
    pandas.set_option('display.max_columns', 5)
    '''
    print('Total number of bolos after base selection = {}'.format(len(df_base)));
    print('Total number of bolos after base selection2 (mislabel==False) = {}'.format(len(df_notmislabel)));
    print('Total number of outliers after base selection = {}'.format(len(df_angle_outlier)));


    if selectiontype==0 :
        selections0 = [\
            ["True", ''],\
            ["pol_angle==0", 'pol=0deg'],\
            ["pol_angle==15", 'pol=15deg'],\
            #["pol_angle==30", 'pol=30deg'],\ # No bolos
            ["pol_angle==45", 'pol=45deg'],\
            ["pol_angle==60", 'pol=60deg'],\
            ["pol_angle==90", 'pol=90deg'],\
            ["pol_angle==105", 'pol=105deg'],\
            #["pol_angle==130", 'pol=130deg'],\ # No bolos
            ["pol_angle==150", 'pol=150deg'],\
            ];
    elif selectiontype==1 :
        selections0 = [\
            ["True", 'All'],\
            ];
    else :
        print(f'Error! There is no selectiontype={selectiontype}!')
        print( '       --> End!')
        return -1;
        pass
    nsel0 = len(selections0);
    # wafer selections
    wafers = ['13.10', '13.11', '13.12', '13.13', '13.15', '13.28', '13.31']
    nwafer = len(wafers);
    # add wafer selections
    wafer_selections = []
    for wafer in wafers:
        wafer_selections.append(['wafer_number=="{}"'.format(wafer), wafer])
        pass

    baseselect = [[]];
    dataname = 'theta_det_angle';
    dataname2= 'theta_design_angle';

    # histogram setting
    binwidth = 1.;
    xbinrange = [-15,15];
    nbins = int((xbinrange[1]-xbinrange[0])/binwidth);
    
    n_sels = [];
    fit_models  = [];
    fit_results = [];
    fit_bins    = [];
    means       = [];
    stds        = [];
    data_selects = [];
    df_selects = [];
    selections = []
    labels = [];
    nzerobins   = [];
    for i, selectinfo in enumerate(selections0) :
        print('*********** i={}th selection ***************'.format(i));
        n_sels.append([]);
        fit_models.append([]);
        fit_results.append([]);
        fit_bins.append([]);
        means.append([]);
        stds.append([]);
        data_selects.append([]);
        df_selects.append([]);
        selections.append([]);
        labels.append([]);
        nzerobins.append([]);
        for j, waferinfo in enumerate(wafer_selections) :
            selection   = selectinfo[0] + '&' + waferinfo[0] + ('' if len(baseselect[0])==0 else ('&' + baseselect[0]));
            selectlabel = selectinfo[1] + '/' + waferinfo[1]  if (len(selectinfo)>1 and len(waferinfo)>1) else selection.strip().replace("'",'').replace('==','=').replace('_type','');
            selections[i].append(selection);
            labels[i].append(selectlabel);
            df_select = df_base.query(selection);
            n_sel     = len(df_select);
            print('selection = {}'.format(selection));
            print('    # of bolos = {}'.format(n_sel));
         
            y = df_select['diff_angle'].to_numpy();
            data_selects[i].append(y);
            df_selects[i].append(df_select);
            n_sels[i].append(n_sel);
            mean = np.mean(y);
            std  = np.std(y);
            print(f'y = {y}')
            print(f'mean = {mean}, std={std}')
            means[i].append(mean);
            stds[i].append(std);

            histo,bins = np.histogram(y,range=xbinrange,bins=nbins, density=False);
            bins_center = np.convolve(bins,[0.5,0.5],mode='valid');
            model = GaussianModel()
            params = model.guess(data=histo, x=bins_center);
            bools  = (y>xbinrange[0]) & (y<xbinrange[1]);
            #params['center'].set(value = np.median(y[bools])*2.);
            #params['sigma'].set(value = 2.);
            params['height'].set(min = 0.);
            print('init for center of gauusian in {}th selection = {}'.format(i, params['center'].value));
            print('init for sigma  of gauusian in {}th selection = {}'.format(i, params['sigma'].value));
            #printVar(histo);
            #printVar(bins_center);
            #intervals_diff = [ np.diff(poisson.interval(alpha=0.68268, mu=N))[0] if N>0. else 0. for N in histo ];
            #weights  = [ 1./(diff/2.) if diff>0. else 0. for diff in intervals_diff ]; # error = (interval_up - interval_down)/2.
            weights = [ 1./np.sqrt(N) if N>0. else 0. for N in histo ]; # weights will be multiply to (data - model). sum{(data-model)*weights)
            nzerobin = np.sum(histo==0.);
            if np.sum(histo)>10. :
                result = model.fit(data=histo, x=bins_center, params=params, weights=weights)
                print(result.fit_report());
                print('weights = ', result.weights);
                print('sqrt(N) = ', np.sqrt(result.data));
                print('red-chi-square = {}: chi-square={}/(Nfree-Nzero)=({}-{})'.format(result.chisqr/(result.nfree-nzerobin), result.chisqr, result.nfree, nzerobin));
                #print(result.ci_report());
            else:
                result = None
                pass
            fit_models[i] .append(copy.deepcopy(model));
            fit_results[i].append(copy.deepcopy(result));
            fit_bins[i]   .append(copy.deepcopy(bins_center));
            nzerobins[i]  .append(nzerobin);
            #del result, bins, histo, model, params;
            pass; # End of loop over wafers
        print('Sum of selected bolos for i th selection = {}'.format(sum(n_sels[i])));
        pass; # End of loop over selections0


    ###################################################
    # Absolute angle plots (Measured - design angles) #
    ###################################################

    # Merged plot
    Nrow    = 3;
    Ncolumn = 4;
    abs_fig_all, abs_axs_all = plt.subplots(Nrow,Ncolumn);
    abs_fig_all.set_size_inches(8*Ncolumn,6*Nrow);
    plt.subplots_adjust(wspace=0.5, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    for i, selectinfo in enumerate(selections0) :

        Nrow    = 3;
        Ncolumn = 4;
        abs_fig, abs_axs = plt.subplots(Nrow,Ncolumn);
        abs_fig.set_size_inches(6*Ncolumn,6*Nrow);
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

        for j, wafer in enumerate(wafers):
            m = (int)(j/Ncolumn);
            n = j%Ncolumn;
            ax = abs_axs[m,n];
            drawAngleHist(ax, iselections=[j], selections=selections[i], fit_models=fit_models[i], fit_results=fit_results[i], means=means[i], stds=stds[i], xbinrange=xbinrange, baseselect=baseselect, data_selects=data_selects[i], labels=labels[i], nbins=nbins, nzerobins=nzerobins[i], nogauss=nogauss);
            ax.set_title('Wafer {}'.format(wafer))
            pass;
        pass;

        # band (90 or 150) v.s. pixel handedness (A or B)
        ax = abs_axs[2,0];
        _dfall = pandas.concat(df_selects[i]);
        #print(df_selects[i])
        #print(_dfall.head());
        x = _dfall['band'];
        y = _dfall['pixel_handedness'];
        x = [ (int)(_x) for _x in x ];
        y = [ (0 if _y=='A' else (1 if _y=='B' else 2)) for _y in y ];
        #print(x);
        #print(y);
        h = ax.hist2d(x,y,bins=4,cmap=cm.jet);
        ax.set_title('Band v.s. Pixel-handedness');
        ax.set_xlabel('Band');
        ax.set_ylabel('Pixel-handedness');
        abs_fig.colorbar(h[3], ax=ax);

        # wafer v.s. center 
        ax = abs_axs[2,1];
        x = np.arange(nwafer);
        y    = np.array([ res.params['center'].value if res is not None else None for res in fit_results[i] ]);
        yerr = np.array([ res.params['center'].stderr if res is not None else None for res in fit_results[i] ]);
        nonzero = np.array(y!=None)
        ax.errorbar(x[nonzero],y[nonzero],yerr=yerr[nonzero],marker='o',markersize=2,linestyle='');
        ax.set_xticks(np.arange(nwafer));
        ax.set_xticklabels(wafers);
        ax.set_title('Wafer v.s. Center angle (measured - design angles)');
        ax.set_xlabel('Wafer');
        ax.set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');

        abs_axs_all[0,0].errorbar(x[nonzero],y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',label=selectinfo[1]);
        abs_axs_all[0,0].set_xticks(np.arange(nwafer));
        abs_axs_all[0,0].set_xticklabels(wafers);
        abs_axs_all[0,0].set_title('Wafer v.s. Center angle (measured - design angles)');
        abs_axs_all[0,0].set_xlabel('Wafer');
        abs_axs_all[0,0].set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
        abs_axs_all[0,0].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.05,1));
        abs_axs_all[0,0].set_ylim(-5.,5.);

        # wafer v.s. mean
        ax = abs_axs[2,2];
        x = np.arange(nwafer);
        y    = np.array(means[i]);
        yerr = np.array(stds[i]);
        nonzero = np.array(y!=None)
        print(y);
        print(nonzero);
        ax.errorbar(x[nonzero],y[nonzero],yerr=yerr[nonzero],marker='o',markersize=2,linestyle='');
        ax.set_xticks(np.arange(nwafer));
        ax.set_xticklabels(wafers);
        ax.set_title('Wafer v.s. Mean angle (measured - design angles)');
        ax.set_xlabel('Wafer');
        ax.set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');

        abs_axs_all[0,1].errorbar(x[nonzero],y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',label=selectinfo[1]);
        abs_axs_all[0,1].set_xticks(np.arange(nwafer));
        abs_axs_all[0,1].set_xticklabels(wafers);
        abs_axs_all[0,1].set_title('Wafer v.s. Mean angle (measured - design angles)');
        abs_axs_all[0,1].set_xlabel('Wafer');
        abs_axs_all[0,1].set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
        abs_axs_all[0,1].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.05,1));
        abs_axs_all[0,1].set_ylim(-5.,5.);
 
 
        # Save fig
        suffix = selectinfo[1].replace(' ','').replace('=','').replace('&','').replace("'","").replace('"','')
        print('savefig to '+outfile+'_{}.png'.format(suffix));
        abs_fig.savefig(outfile+'_{}.png'.format(suffix));
        pass

    # Save fig
    print('savefig to '+outfile+'.png');
    abs_fig_all.savefig(outfile+'.png');



    if additional_plots:

        #########################################################################
        # Misc. plots (Measured v.s. design, Histogram, Focalplane plots, etc.) #
        #########################################################################

        Nrow    = 5;
        Ncolumn = 3;
        fig, axs = plt.subplots(Nrow,Ncolumn);
        fig.set_size_inches(6*Ncolumn,6*Nrow);
        #fig.tight_layout(rect=[0,0,1,1]);
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

        # 2D plot 1 (Measured angle v.s. design angle)
        axs[0][0].plot(df_notmislabel[dataname2], df_notmislabel[dataname],marker='o', markersize=1.0, linestyle='');
        axs[0][0].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
        axs[0][0].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
        axs[0][0].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
        axs[0][0].grid(True);
        axs[0][0].set_title('Bolos with correct labels originally (no mis-label)');
        axs[0][0].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle-90deg" in focalplane database)',fontsize=16);
        axs[0][0].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ [deg.]',fontsize=16);
        axs[0][0].set_xticks(np.arange(-360,360,45));
        axs[0][0].set_yticks(np.arange(-360,360,45));
        axs[0][0].set_xlim(-22.5,180);
        axs[0][0].set_ylim(-22.5,180);
        axs[0][0].tick_params(labelsize=12);

        # 2D plot 2 (Measured angle v.s. design angle)
        axs[0][1].plot(df_base[dataname2], df_base[dataname],marker='o', markersize=1.0, linestyle='', label='all');
        axs[0][1].plot(df_angle_outlier[dataname2], df_angle_outlier[dataname],marker='o', markersize=1.0, linestyle='',color='r', label='outliers');
        axs[0][1].plot([-360,360],[-360,360],linestyle='-',color='k',linewidth=0.5);
        axs[0][1].plot([-360,360],[0,0],linestyle='-',color='k',linewidth=0.5);
        axs[0][1].plot([0,0],[-360,360],linestyle='-',color='k',linewidth=0.5);
        axs[0][1].grid(True);
        axs[0][1].legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=1.0);
        axs[0][1].set_title('Bolos with base cuts after label correction (all/outliers)');
        axs[0][1].set_xlabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle-90deg" in focalplane database)',fontsize=16);
        axs[0][1].set_ylabel(r'$\theta_{\mathrm{det,wiregrid}}$ [deg.]',fontsize=16);
        axs[0][1].set_xticks(np.arange(-360,360,45));
        axs[0][1].set_yticks(np.arange(-360,360,45));
        axs[0][1].set_xlim(-22.5,180);
        axs[0][1].set_ylim(-22.5,180);
        axs[0][1].tick_params(labelsize=12);


        # Focal plane plot 1
        im0= axs[2][0].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset[dataname], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
        axs[2][0].set_title(dataname +' [deg.]');
        axs[2][0].grid(True);
        axs[2][0].set_xlabel('x offset', fontsize=16);
        axs[2][0].set_ylabel('y offset', fontsize=16);
        axs[2][0].set_xlim([-fp_width,fp_width]);
        axs[2][0].set_ylim([-fp_width,fp_width]);
        fig.colorbar(im0, ax=axs[2][0]);

        # Focal plane plot 2
        im1= axs[2][1].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset[dataname2], marker='o', s=10, cmap='coolwarm', vmin=0., vmax=180.);
        axs[2][1].set_title(dataname2+' [deg.]');
        axs[2][1].grid(True);
        axs[2][1].set_xlabel('x offset', fontsize=16);
        axs[2][1].set_ylabel('y offset', fontsize=16);
        axs[2][1].set_xlim([-fp_width,fp_width]);
        axs[2][1].set_ylim([-fp_width,fp_width]);
        fig.colorbar(im1, ax=axs[2][1]);
     
        # Focal plane plot 3
        im2= axs[2][2].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset['diff_angle'], marker='o', s=10, cmap='coolwarm', vmin=-10., vmax=10.);
        axs[2][2].set_title(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
        axs[2][2].grid(True);
        axs[2][2].set_xlabel('x offset', fontsize=16);
        axs[2][2].set_ylabel('y offset', fontsize=16);
        axs[2][2].set_xlim([-fp_width,fp_width]);
        axs[2][2].set_ylim([-fp_width,fp_width]);
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
        # Correct det_offset_x/y
        df_base.loc[:,'det_offset_x_corr'] = df_base['det_offset_x'] - center_offset_x;
        df_base.loc[:,'det_offset_y_corr'] = df_base['det_offset_y'] - center_offset_y;
        # Add points/lines
        xlim = axs[2][2].get_xlim();
        ylim = [xlim[0]-center_offset_x+center_offset_y, xlim[1]-center_offset_x+center_offset_y];
        axs[2][2].plot([xlim[0],xlim[1]],[ylim[0],ylim[1]],c='k',marker='',linestyle='--',linewidth=1);
        axs[2][2].scatter([center_offset_x],[center_offset_y],c='k',marker='*',s=200);
        axs[2][2].scatter([center_offset_x],[center_offset_y],c='yellow',marker='*',s=50);
        axs[2][2].set_xlim(xlim);
        axs[2][2].set_ylim(ylim);

        # Focal plane plot 4 (band freq. check)
        im3= axs[0][2].scatter(df_offset['det_offset_x'], df_offset['det_offset_y'], c=df_offset['band'], marker='o', s=10, cmap='coolwarm', vmin=90., vmax=150.);
        axs[1][2].set_title('Band frequency [GHz]');
        axs[1][2].grid(True);
        axs[1][2].set_xlabel('x offset', fontsize=16);
        axs[1][2].set_ylabel('y offset', fontsize=16);
        axs[1][2].set_xlim([-fp_width,fp_width]);
        axs[1][2].set_ylim([-fp_width,fp_width]);
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

        # Other histogram 3 (2D histogram)
        # plot: r0 v.s. theta0
        h1 = axs[3][2].hist2d(df_base_center['r0'], theta, bins=40, cmap='jet', range=[[0.,500.],[0.,180.]]);
        axs[3][2].set_title(r'r0 v.s. $\theta 0$ of fake pol.');
        axs[3][2].set_xlabel(r'$r_0$ [mK]');
        axs[3][2].set_ylabel(r'$\theta_0$ [deg.]');
        axs[3][2].grid(True);
        fig.colorbar(h1[3], ax=axs[3][2]);



        # Other histogram 4
        # plot: tau
        axs[4][0].hist(df_base['tau']*1000., bins=100, range=[0.,50.], histtype='stepfilled',
                 align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                 color=colors[0], alpha=1.0, label='tau', stacked=False);
        axs[4][0].set_title(r'Time constant $\tau$ measured by stimulator [msec]');
        axs[4][0].set_xlabel(r'Time constant $\tau$ [msec]');
        axs[4][0].set_ylabel(r'# of bolometers');
        axs[4][0].grid(True);

        # Other histogram 5
        # plot: r
        axs[4][1].hist(df_base['r']*1e-3, bins=100, range=[0.,10.], histtype='stepfilled',
                 align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
                 color=colors[0], alpha=1.0, label='r', stacked=False);
        axs[4][1].set_title(r'Amplitude $r$ [K]');
        axs[4][1].set_xlabel(r'Amplitude $r$ [K]');
        axs[4][1].set_ylabel(r'# of bolometers');
        axs[4][1].grid(True);


        # Save fig
        print('savefig to '+outfile+'_misc.png');
        fig.savefig(outfile+'_misc.png');



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
                    {'outname':'_FPposTopBottom', 'sels':['det_offset_y_corr>=0.', 'det_offset_y_corr<0.'], 'labels':['Top in FP', 'Bottom in FP']},
                    {'outname':'_FPposLeftRight', 'sels':['det_offset_x_corr<0.', 'det_offset_x_corr>0.'], 'labels':['Left in FP', 'Right in FP']},
                    {'outname':'_FPpos', 'sels':['det_offset_x<det_offset_y', 'det_offset_x>=det_offset_y'], 'labels':['Top Right in FP', 'Bottom Left in FP']},
                    {'outname':'_FPposTopLR', 'sels':['det_offset_x_corr+2.5<det_offset_y_corr', '2.5-det_offset_x_corr<det_offset_y_corr'], 'labels':['Top Left in FP', 'Top Right in FP']},
                    {'outname':'_FPposBottomLR', 'sels':['-2.5-det_offset_x_corr>det_offset_y_corr', 'det_offset_x_corr-2.5>det_offset_y_corr'], 'labels':['Bottom Left in FP', 'Bottom Right in FP']},
                    {'outname':'_FPposTR-BL', 'sels':['2.5-det_offset_x_corr<det_offset_y_corr', '-2.5-det_offset_x_corr>det_offset_y_corr'], 'labels':['Top Right in FP', 'Bottom Left in FP']},
                    {'outname':'_FPposTR-BL2', 'sels':['2-det_offset_x_corr<det_offset_y_corr', '-2-det_offset_x_corr>det_offset_y_corr'], 'labels':['Top Right in FP', 'Bottom Left in FP']},
                    {'outname':'_FPposTL-BR', 'sels':['det_offset_x_corr+2.5<det_offset_y_corr', 'det_offset_x_corr-2.5>det_offset_y_corr'], 'labels':['Top Left in FP', 'Bottom Right in FP']},
                    {'outname':'_FPposCenter-Out', 'sels':['wafer_number=="13.13"', 'wafer_number!="13.13"'], 'labels':['Center wafer', 'Outer wafers']},
                    ];
            
            
            for selection_set in selection_sets :

                sels    = selection_set['sels'];
                slabels = selection_set['labels'];
                nsel = len(sels);
                dfs = [df_base.query(sel) if sel!='' else df_base for sel in sels];
                for n, df in enumerate(dfs) :
                    print('# of bolos for {}({}) = {}'.format(sels[n], slabels[n], len(df)));
                    pass;

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
                theta_diffmean_set = [];
                theta_diffmean_mean_set = [];
                theta_diffmean_std_set = [];
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
                        r       = df[columns_r[n]      ].values;
                        r_err   = df[columns_r_err[n]  ].values;
                        r_ratio = df[columns_r_ratio[n]].values;
                        theta      = df[columns_theta[n]     ].values;
                        theta_err  = df[columns_theta_err[n] ].values;
                        theta_diff = df[columns_theta_diff[n]].values;
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
                    theta_diffmeans = [];
                    theta_diff_mean_mean = np.mean(theta_diff_means);
                    for n, angle in enumerate(wire_angles):
                        theta_diffmeans.append(theta_diffs[n]-theta_diff_mean_mean);
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
                    theta_diffmean_set.append(theta_diffmeans);
                    theta_diffmean_mean_set.append(theta_diff_means-theta_diff_mean_mean);
                    theta_diffmean_std_set.append(theta_diff_stds);
                    pass;
                #print('r_ratio_set', r_ratio_set);


                #########################
                # r for each wire angle #
                #########################

                plotEachWire(dfs, r_ratio_set, r_ratio_mean_set, r_ratio_std_set, slabels, 
                    wire_angles, alabels,
                    histbins=100, histxrange=(0.9,1.1),
                    title=r'Signal power ($r$) ratio', 
                    vartitle=r'$r(\theta _{\mathrm{wire}})/r_{\mathrm{circle}}$',
                    mean_ref = 1., mean_yrange = (0.97,1.03),
                    outfilename = outfile+'_eachwire-{}{}.png'.format('r',selection_set['outname']),
                    );

                #############################
                # theta for each wire angle #
                #############################

                plotEachWire(dfs, theta_diff_set, theta_diff_mean_set, theta_diff_std_set, slabels, 
                    wire_angles, alabels,
                    histbins=100, histxrange=(-10,10),
                    title=r'Theta diff. from expected one', 
                    vartitle=r'$\theta_{\mathrm{meas.}} - \theta_{\mathrm{exp.}}$',
                    mean_ref = 0., mean_yrange = (-4.,4.),
                    outfilename = outfile+'_eachwire-{}{}.png'.format('theta',selection_set['outname']),
                    );

                ##############################################
                # theta relative to mean for each wire angle #
                ##############################################

                plotEachWire(dfs, theta_diffmean_set, theta_diffmean_mean_set, theta_diffmean_std_set, slabels, 
                    wire_angles, alabels,
                    histbins=100, histxrange=(-10,10),
                    title=r'Theta diff. from mean', 
                    vartitle=r'$\theta_{\mathrm{meas.}} - \theta_{\mathrm{exp. from mean}}$',
                    mean_ref = 0., mean_yrange = (-3.,3.),
                    outfilename = outfile+'_eachwire-{}{}.png'.format('theta_from_mean',selection_set['outname']),
                    );

                pass;
            pass;

        pass; # End of additional_plots

    return selections0, fit_results, means, stds;


if __name__=='__main__' :
    check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_90A-TorB'+ver, additional_cut="band==90&pixel_handedness=='A'&(bolo_type=='T'|bolo_type=='B')", nogauss=True);
    #check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_90A'+ver, additional_cut="band==90&pixel_handedness=='A'");
    #check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_90B'+ver, additional_cut="band==90&pixel_handedness=='B'");
    #check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_150A'+ver, additional_cut="band==150&pixel_handedness=='A'");
    #check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_150B'+ver, additional_cut="band==150&pixel_handedness=='B'");


    g_sel0 = [];
    g_fits = []; 
    g_means= []; 
    g_stds = [];

    '''
    outfile = f'out_check_absolute/check_absolute_eachwafer_allpol_Ahanded_compare{ver}.png';
    g_labels= ['90 GHz A Top', '90GHz A Bottom', '150GHz A Top', '150GHz A Bottom']
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90ATop'+ver, additional_cut="band==90&pixel_handedness=='A'&bolo_type=='T'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90ABottom'+ver, additional_cut="band==90&pixel_handedness=='A'&bolo_type=='B'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150ATop'+ver, additional_cut="band==150&pixel_handedness=='A'&bolo_type=='T'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150ABottom'+ver, additional_cut="band==150&pixel_handedness=='A'&bolo_type=='B'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    #'''

    '''
    outfile = f'out_check_absolute/check_absolute_eachwafer_allpol_Bhanded_compare{ver}.png';
    g_labels= ['90 GHz B Top', '90GHz B Bottom', '150GHz B Top', '150GHz B Bottom']
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90BTop'+ver, additional_cut="band==90&pixel_handedness=='B'&bolo_type=='T'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90BBottom'+ver, additional_cut="band==90&pixel_handedness=='B'&bolo_type=='B'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150BTop'+ver, additional_cut="band==150&pixel_handedness=='B'&bolo_type=='T'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150BBottom'+ver, additional_cut="band==150&pixel_handedness=='B'&bolo_type=='B'", selectiontype=1);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    #'''

    outfile = f'out_check_absolute/check_absolute_eachwafer_allpol_A-B_compare{ver}.png';
    g_labels= ['90 GHz A Top', '90 GHz B Top', '90GHz A Bottom', '90GHz B Bottom', '150GHz A Top', '150GHz B Top', '150GHz A Bottom', '150GHz B Bottom']
    g_labels2= ['90 GHz Top', '90GHz Bottom', '150GHz Top', '150GHz Bottom']
    
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90ATop'+ver, additional_cut="band==90&pixel_handedness=='A'&bolo_type=='T'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90BTop'+ver, additional_cut="band==90&pixel_handedness=='B'&bolo_type=='T'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90ABottom'+ver, additional_cut="band==90&pixel_handedness=='A'&bolo_type=='B'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_90BBottom'+ver, additional_cut="band==90&pixel_handedness=='B'&bolo_type=='B'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150ATop'+ver, additional_cut="band==150&pixel_handedness=='A'&bolo_type=='T'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150BTop'+ver, additional_cut="band==150&pixel_handedness=='B'&bolo_type=='T'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150ABottom'+ver, additional_cut="band==150&pixel_handedness=='A'&bolo_type=='B'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    sel0, fits, means, stds = check_absolute(additional_plots=False, outfile='out_check_absolute/check_absolute_eachwafer_allpol_150BBottom'+ver, additional_cut="band==150&pixel_handedness=='B'&bolo_type=='B'", selectiontype=1, nogauss=True);
    g_sel0.append(sel0); g_fits.append(fits); g_means.append(means); g_stds.append(stds);
    #'''

    Nrow = 3;
    Ncolumn = 3;
    fig, axs = plt.subplots(Nrow,Ncolumn);
    fig.set_size_inches(8*Ncolumn,6*Nrow);
    plt.subplots_adjust(wspace=0.5, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)
    wafers = ['13.10', '13.11', '13.12', '13.13', '13.15', '13.28', '13.31']
    nwafer = len(wafers);

    icolor=0;
    dphis = [];
    dphi_errs = [];
    for n, label in enumerate(g_labels):
        for i, sel in enumerate(g_sel0[n]):

            fit_results = g_fits[n];
            # wafer v.s. gauss center 
            ax = axs[0,0];
            x = np.arange(nwafer);
            y    = np.array([ res.params['center'].value if res is not None else None for res in fit_results[i] ]);
            yerr = np.array([ res.params['center'].stderr if res is not None else None for res in fit_results[i] ]);
            nonzero = np.array(y!=None)
            ax.errorbar(x[nonzero]+0.1*n,y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',color=colors[icolor],label=f'{label} {sel[1]}');
            ax.set_title('Wafer v.s. Gaussian center angle (measured - design angles)');
            ax.set_xlabel('Wafer');
            ax.set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
            ax.set_ylim(-7.5,7.5);
            ax.grid(True)
            ax.set_xticks(np.arange(-0.5,nwafer+0.5,1));
            ax.set_xticklabels(['']+wafers, ha='right');
            ax.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.02,1));

            means = g_means[n];
            stds  = g_stds[n];
            # wafer v.s. mean
            ax = axs[0,1];
            x = np.arange(nwafer);
            y    = np.array(means[i]);
            yerr = np.array(stds[i]);
            nonzero = np.array(y!=None)
            ax.errorbar(x[nonzero]+0.1*n,y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',color=colors[icolor],label=f'{label} {sel[1]}');
            ax.set_title('Wafer v.s. Mean angle (measured - design angles)');
            ax.set_xlabel('Wafer');
            ax.set_ylabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
            ax.set_ylim(-7.5,7.5);
            ax.grid(True)
            ax.set_xticks(np.arange(-0.5,nwafer+0.5,1));
            ax.set_xticklabels(['']+wafers, ha='right');
            ax.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.02,1));
         
            if n<len(g_labels)/2 and len(g_labels2)>n:
                # wafer v.s. mean
                label2 = g_labels2[n];
                ax = axs[1,1];
                x = np.arange(nwafer);
                y    = np.array(g_means[n*2][i]) - np.array(g_means[n*2+1][i]); # A-B
                yerr = np.sqrt( np.array(g_stds[n*2][i])**2. +  np.array(g_stds[n*2+1][i])**2. );
                dphi = np.mean(y); # average on wafers
                dphi_err = 1./len(y) * np.sqrt(np.sum(yerr*yerr)) # average on wafers
                dphis.append(dphi); 
                dphi_errs.append(dphi_err);
                nonzero = np.array(y!=None)
                ax.errorbar(x[nonzero]+0.1*n,y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',color=colors[icolor],label=f'{label2} {sel[1]}');
                ax.set_title('Wafer v.s. A - B');
                ax.set_xlabel('Wafer');
                ax.set_ylabel(r'$\Delta \phi_{A-B}$ [deg.]');
                ax.set_ylim(-7.5,7.5);
                ax.grid(True)
                ax.set_xticks(np.arange(-0.5,nwafer+0.5,1));
                ax.set_xticklabels(['']+wafers, ha='right');
                ax.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.02,1));

                ax = axs[1,2];
                x = np.arange(nwafer);
                y    = np.array(g_means[n*2][i]) + np.array(g_means[n*2+1][i]); # A+B
                yerr = np.sqrt( np.array(g_stds[n*2][i])**2. +  np.array(g_stds[n*2+1][i])**2. );
                nonzero = np.array(y!=None)
                ax.errorbar(x[nonzero]+0.1*n,y[nonzero],yerr=yerr[nonzero],marker='o',markersize=4,linestyle='',color=colors[icolor],label=f'{label2} {sel[1]}');
                ax.set_title('Wafer v.s. A + B');
                ax.set_xlabel('Wafer');
                ax.set_ylabel(r'$\phi_{A} + \phi_{B}$ [deg.]');
                ax.set_ylim(-10,10);
                ax.grid(True)
                ax.set_xticks(np.arange(-0.5,nwafer+0.5,1));
                ax.set_xticklabels(['']+wafers, ha='right');
                ax.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 10, title='',borderaxespad=0.,labelspacing=0.8,bbox_to_anchor=(1.02,1));
                pass;

            icolor += 1;

            pass;
        pass;

    if len(dphis)>0:
        for i, label in enumerate(g_labels2) :
            print(f'dphi(A-B) ({label}) = {dphis[i]} +- {dphi_errs[i]} deg');
            pass;
        pass;

    # Save fig
    print('savefig to '+outfile);
    fig.savefig(outfile);




