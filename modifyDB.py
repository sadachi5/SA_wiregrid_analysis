import sys, os;
import numpy as np;
import sqlite3, pandas;
import copy;
from utils import deg0to180, theta0topi, colors, printVar, rad_to_deg, rad_to_deg_pitopi, rad_to_deg_0to2pi, deg_to_rad, deg90to90, calculateRTheta;
from matplotlib import pyplot as plt;



def modify(infile='out_ver10/db/all_pandas_correct_label.db', outfile='out_ver10/db/pb2a_wiregrid_ver10', plotdir='output_ver10/db/modifyDB/', opts=[], isHWPSS=False):

    # Configure for base selections
    # stim_quality_cut = 'tau>0.';
    if not isHWPSS:
        print('Go to wiregrid mode!')
        tablename = 'wiregrid';
        new_tablename = 'pb2a_wiregrid';
        wg_quality_cut = 'theta_det_err*180./{}<0.5&pol_angle>=0'.format(np.pi);
        keep_columns   = [
            'bolo_name',
            'readout_name',
            'wafer_number',
            'band',
            'pixel_name',
            'pixel_number',
            'pixel_type',
            'pixel_handedness',
            'bolo_type',
            'pol_angle',
            'theta_det',
            'theta_det_err',
            'theta_det_err_total',
            'tau'
            ];
    else:
        print('Go to HWPSS mode!')
        tablename = 'hwpss';
        new_tablename = 'pb2a_hwpss';
        wg_quality_cut = ''
        keep_columns   = [
            'bolo_name',
            'readout_name',
            'wafer_number',
            'band',
            'pixel_name',
            'pixel_number',
            'pixel_type',
            'pixel_handedness',
            'bolo_type',
            'pol_angle',
            'theta_det',
            'theta_det_err',
            'theta_det_err_total',
            'entry',
            'mu',
            'sigma',
            'mu_error',
            'sigma_error',
            'flag',
            ];
        pass;
    outlier_deg = 15.;
    # base_cuts      = [stim_quality_cut, wg_quality_cut];
    base_cuts      = [wg_quality_cut];
    base_cuts      = [ cut for cut in base_cuts if cut!='']; # remove empty cut
    base_selection = '&'.join(base_cuts);


    # Get pandas
    conn_w = sqlite3.connect(infile);
    df_w=pandas.read_sql_query('SELECT {} FROM {}'.format(','.join(keep_columns), tablename), conn_w);
    print(f'# of bolos: {len(df_w)} (initial)')
    conn_w.close();

    #############
    # Selection #
    #############

    # Select bolometers passing basecuts
    if base_selection!='' : 
        df_w = df_w.query(base_selection);
        print(f'# of bolos: {len(df_w)} after base_selection = {base_selection}')

    # Drop bolometers with pol_angle = Nan
    df_w = df_w.dropna(subset=['pol_angle']);
    print(f'# of bolos: {len(df_w)} after dropping Nan pol_angle')
    if not isHWPSS:
        # Drop bolometers with tau = Nan
        df_w = df_w.dropna(subset=['tau']);
        print(f'# of bolos: {len(df_w)} after dropping Nan tau')
        pass;

    # Drop bolometers with pixel_name = Nan
    df_w = df_w.dropna(subset=['pixel_name']);
    print(f'# of bolos: {len(df_w)} after dropping Nan pixel_name')
    # Drop bolometers with bolo_type = Nan
    df_w = df_w.dropna(subset=['bolo_type']);
    print(f'# of bolos: {len(df_w)} after dropping Nan bolo_type')
    # Drop bolometers with band = Nan
    df_w = df_w.dropna(subset=['band']);
    print(f'# of bolos: {len(df_w)} after dropping Nan band')
    # Drop bolometers with band = Nan
    df_w = df_w[df_w['band']>0];
    print(f'# of bolos: {len(df_w)} after dropping band<=0')

    # Modify theta_det angle for HWPSS
    if isHWPSS:
        df_w['theta_det'] = theta0topi(df_w['theta_det'] + deg_to_rad(16.5-45.))
        pass;

    # Calculate diff_angle
    calib_angle  = rad_to_deg( theta0topi(df_w['theta_det'] + 2.*deg_to_rad(-16.5), upper=np.pi) ); 
    design_angle = deg0to180(df_w['pol_angle']-90., upper=180.);
    df_w['diff_angle'] = deg90to90(calib_angle-design_angle);
    # Remove outliers
    df_w = df_w[ np.abs(df_w['diff_angle'])<outlier_deg ];
    print(f'# of bolos: {len(df_w)} after removing outliers in angle')

    # Drop duplicated bolometers
    df_w = df_w.drop_duplicates()
    print(f'# of bolos: {len(df_w)} after dropping duplicated bolos')

    # Reset index
    df_w = df_w.reset_index()

    ##################
    # Remove columns #
    ##################
    # Remove columns used in selection but not kept
    if not isHWPSS: del df_w['tau'];
    del df_w['theta_det_err'];
    del df_w['diff_angle'];


    ################
    # Modification #
    ################

    # Shift theta_det by 90 deg.
    # Wrapped by [0,pi]
    df_w['theta_det'] = theta0topi(df_w['theta_det']-np.pi/2., upper=np.pi);

    # Rename: theta_det_err_total --> theta_det_err
    df_w = df_w.rename(columns={'theta_det_err_total':'theta_det_err'});

    # Check boloname
    n_bolo = len(df_w);
    is_nanbolo = df_w['bolo_name'].isnull() | (df_w['bolo_name'].str.len()==0);
    print('# of Nan bolo_name', sum(is_nanbolo));
    df_w_nanbolo = df_w[is_nanbolo];
    print('Nan boloname Data (size={}/{}):'.format(len(df_w_nanbolo), n_bolo));
    print(df_w_nanbolo);
    expected_bolonames = np.array([ '{}.{}{}'.format(__a,__b,__c) for __a,__b,__c in zip(df_w['pixel_name'],df_w['band'],df_w['bolo_type']) ]);
    print('Expected bolonames = ',expected_bolonames[is_nanbolo]);
    if sum(is_nanbolo)>0 : 
        print('set new boloname')
        # modify bolo_name if is_nanbolo==True
        print(df_w['bolo_name'].mask(is_nanbolo, expected_bolonames));
        df_w['bolo_name'] = df_w['bolo_name'].mask(is_nanbolo, expected_bolonames);
        #print('# of Nan bolo_name after bolo_name correction', sum(df_w['bolo_name'].isnull() | (df_w['bolo_name'].str.len()==0)));
        pass;

    #########
    # Check #
    #########

    # bolo_name check
    new_is_nanbolo =df_w['bolo_name'].isnull() | (df_w['bolo_name'].str.len()==0);
    print('# of NaN bolo_names after boloname correction = {} / {}'.format(sum(new_is_nanbolo), n_bolo));
    expected_bolonames = np.array([ '{}.{}{}'.format(__a,__b,__c) for __a,__b,__c in zip(df_w['pixel_name'],df_w['band'],df_w['bolo_type']) ]);
    iscorrect_boloname = (df_w['bolo_name'] == expected_bolonames);
    print('# of correct bolo_names = {} / {}'.format(sum(iscorrect_boloname), n_bolo));
    print('# of incorrect bolo_names = {} / {}'.format(sum(~iscorrect_boloname), n_bolo));
    #'''
    print('Incorrect bolo_name DB:');
    pandas.set_option('display.max_columns', 50)
    print(df_w[~iscorrect_boloname]);
    pandas.set_option('display.max_columns', 5)
    #'''
    # band in bolo_name 
    band_in_boloname = df_w['bolo_name'].str.split('.',expand=True); # divide by '.'
    band_in_boloname = (band_in_boloname[2].str[:-1]); # obtain band numbers in string
    #for a,b,c in zip(band_in_boloname, df_w['band'], df_w['bolo_name']): print(a,b,len(c));
    band_in_boloname = band_in_boloname.astype(int)
    iscorrect_band = (band_in_boloname == df_w['band']);
    print('# of correct band = {} / {}'.format(sum(iscorrect_band), n_bolo));
    print('# of incorrect band = {} / {}'.format(sum(~iscorrect_band), n_bolo));
    #'''
    print('Incorrect band DB:');
    pandas.set_option('display.max_columns', 50)
    print(df_w[~iscorrect_band]);
    pandas.set_option('display.max_columns', 5)
    #'''
    # wafer in bolo_name 
    wafer_in_boloname = df_w['bolo_name'].str.split('_',expand=True); # divide by '_'
    wafer_in_boloname = (wafer_in_boloname[0]); # obtain wafer numbers in string
    iscorrect_wafer = (wafer_in_boloname == df_w['wafer_number']);
    print('# of correct wafer = {} / {}'.format(sum(iscorrect_wafer), n_bolo));
    print('# of incorrect wafer = {} / {}'.format(sum(~iscorrect_wafer), n_bolo));
    #'''
    print('Incorrect wafer DB:');
    pandas.set_option('display.max_columns', 50)
    print(df_w[~iscorrect_wafer]);
    pandas.set_option('display.max_columns', 5)
    #'''


    # Nan check
    for key in df_w.keys():
        print('# of Nan ({}) = {}'.format(key, sum(df_w[key].isnull())) );
        pass;

    #print('New DB keys:');
    #print(df_w.keys());

    #############
    # Save df_w #
    #############

    # Save to pickle file
    outputname = outfile + '.pkl';
    print('Saving the wiregrid DB to a pickle file ({})...'.format(outputname));
    df_w.to_pickle(outputname);
    # Save to sqlite3 file
    outputname = outfile + '.db';
    print('Saving the wiregrid DB to a sqlite3 file ({})...'.format(outputname));
    new_conn = sqlite3.connect(outputname);
    df_w.to_sql(new_tablename,new_conn,if_exists='replace',index=None);
    new_conn.close();

    ##############
    # Make plots #
    ##############

    if True:
        i_figs = 4;
        j_figs = 4;
        fig, axs = plt.subplots(i_figs,j_figs);
        fig.set_size_inches(6*j_figs,6*i_figs);
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

        ax = axs[0][0];
        ax.hist(rad_to_deg(df_w['theta_det']), bins=90, range=[0.,180.], histtype='stepfilled',
            align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
            color=colors[0], alpha=0.5, label=[''], stacked=False);
        ax.set_title(r'$\theta_{\mathrm{det}}$ [deg.]');
        ax.set_xlabel(r'$\theta_{\mathrm{det}}$ [deg.]',fontsize=12);
        ax.set_ylabel('# of bolometers',fontsize=12);
 
        ax = axs[0][1];
        ax.hist(df_w['pol_angle'], bins=90, range=[0.,180.], histtype='stepfilled',
            align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
            color=colors[0], alpha=0.5, label=[''], stacked=False);
        ax.set_title(r'$\theta_{\mathrm{design}}$ [deg.]');
        ax.set_xlabel(r'$\theta_{\mathrm{design}}$ [deg.]',fontsize=12);
        ax.set_ylabel('# of bolometers',fontsize=12);
 
        x = deg90to90(rad_to_deg(df_w['theta_det']) - df_w['pol_angle']);
        ax = axs[0][2];
        hist = ax.hist(x, bins=90, range=[-45.,45.], histtype='stepfilled',
            align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
            color=colors[0], alpha=0.5, label=[''], stacked=False);
        ax.set_title(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]');
        ax.set_xlabel(r'$\theta_{\mathrm{det}} - \theta_{\mathrm{design}}$ [deg.]',fontsize=12);
        ax.set_ylabel('# of bolometers',fontsize=12);
        xlim = ax.get_xlim();
        ylim = ax.get_ylim();
        ax.text(xlim[0]+(xlim[1]-xlim[0])*0.1, ylim[0]+(ylim[1]-ylim[0])*0.3, '# of total = {:.0f} bolos'.format(sum(hist[0])),fontsize=8);
        ax.text(xlim[0]+(xlim[1]-xlim[0])*0.1, ylim[0]+(ylim[1]-ylim[0])*0.2, 'Mean = {:.1f} deg.'.format(np.mean(x)),fontsize=8);

        if not os.path.isdir(plotdir):
            os.mkdir(plotdir);
            pass;
        plotfile = f'{plotdir}/plot.png'
        print(f'Saving figure...: {plotfile}');
        fig.savefig(plotfile);
        pass;


    return 0;


if __name__=='__main__' :
    ver='ver10.2';
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
    plotdir = f'output_{ver}/db/modifyDB/'
    infile  = f'output_{ver}/db/all_pandas_correct_label.db'
    outfile = f'output_{ver}/db/pb2a_wiregrid_{ver}{suffix}'
    modify(infile=infile, outfile=outfile, plotdir=plotdir, opts=opts);
    pass;


