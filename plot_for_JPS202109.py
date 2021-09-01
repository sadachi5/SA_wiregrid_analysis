import os, sys
import argparse;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle
import libg3py3 as libg3
import math
from datetime import datetime
from scipy.optimize import curve_fit
from spt3g import core

from DBreaderStimulator import DBreaderStimulator;
from DBreader import DBreader;

from slowdaq_map import *;
from loadbolo import loadbolo;
from utils import colors, printVar, theta0to2pi, rad_to_deg;
import Out;
out = Out.Out(verbosity=0);

# tentative setting 
filename_tmp='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
boloname_tmp='PB20.13.13_Comb01Ch03';
startStr_tmp = "20210205_180230";
endStr_tmp   = "20210205_180400";



def plot(filename, boloname=None, 
        start=None, end=None, calibGain=True,
        outname='output', outdir='plot', loaddata=True, loadWHWP=True,loadSlow=True) :

    # initialize data list
    bolonames  = [];
    time       = [];
    whwp_angle = [];
    bolotime_boloarray = [];
    y_boloarray        = [];

    outpath = outdir+'/'+outname;
    out.OUT('boloname = {}'.format(boloname));
    out.OUT('outpath = {}'.format(outpath));
    # check directory
    if not os.path.isdir(outdir) : 
        out.WARNING('There is no output directory: {}'.format(outdir));
        out.WARNING('--> Make the directory.');
        os.mkdir(outdir);
        pass;
    # open & read pickle file
    if not loaddata :
        if os.path.isfile(outpath+'.pkl') :
            outfile = open(outpath+'.pkl', 'rb');
            bolonames = pickle.load(outfile);
            time      = pickle.load(outfile);
            whwp_angle= pickle.load(outfile);
            bolotime_boloarray  = pickle.load(outfile);
            y_boloarray         = pickle.load(outfile);
            if loadSlow :
                LStemp_time = pickle.load(outfile);
                LStemp_data = pickle.load(outfile);
                SIMtemp_time = pickle.load(outfile);
                SIMtemp_data = pickle.load(outfile);
                pass;
        else : # if there is no pickle file, load g3 datafile
            loaddata = True;
            pass;
        pass;

    # get stimulator gain
    gaincal = [];
    for bolo in boloname :
        if calibGain :
            # amp is half height of the modulation. It is not the full height.
            db_stim = DBreaderStimulator('./data/pb2a_stimulator_run223_20210223.db');
            stimulator_amp  = [db_stim.getamp(22300607, bolo), db_stim.getamp(22300610, bolo),]; # [ADC counts] Run22300607, Run22300610
            out.OUT(stimulator_amp,-2);
            if stimulator_amp[0][0]==0. or stimulator_amp[1][0]==0. :
                out.WARNING('There is no matched stimulator amplitude data for {}'.format(bolo));
                out.WARNING('The output is {}'.format(stimulator_amp));
                pass;
            out.OUT('stimulator amp (run 22300607) = {} +- {}'.format(stimulator_amp[0][0], stimulator_amp[0][1]), 0);
            out.OUT('stimulator amp (run 22300610) = {} +- {}'.format(stimulator_amp[1][0], stimulator_amp[1][1]), 0);
            # Get stimulator temperature
            # Old version
            #stimulator_temp = 52. ; # 52. [mK_RJ/amp[ADC counts]] @ 90GHz, 103. [mK_RJ/amp[ADC counts]] @ 150GHz
            # New version: Get the number from stimulator template DB
            db_stim_temp = DBreaderStimulator('./data/pb2a_stim_template_20210607.db', tablename='pb2a_stim_template');
            #temps = db_stim_temp.getintensity(channelname=bolo,nearRunID=22300610,source='Jupiter'); # [K_RJ/amp, K_CMB/amp]
            #stimulator_temp = temps[0] * 1000.; # near run jupiter [mK_RJ/amp]
            temps = db_stim_temp.getintensity(channelname=bolo,nearRunID=None,source='Jupiter'); # [K_RJ/amp, K_CMB/amp]
            stimulator_temp = np.mean(np.array(temps)[:,0]) * 1000.; # averaged jupiter [mK_RJ/amp]
            # set calibration constant and its error from ADC output to mK_RJ
            cal     = stimulator_temp/stimulator_amp[1][0]  if stimulator_amp[1][0]>0. else 0.; # chose 90GHz after calibration
            cal_err = cal/stimulator_amp[1][0] * stimulator_amp[1][1] if stimulator_amp[1][0]>0. else 0.; # chose 90GHz after calibration 
            out.OUT('calibration constant (ADC->mK_RJ) = {:.3f} +- {:.3f}'.format(cal,cal_err),0);
            gaincal.append(cal);
        else :
            gaincal.append(1.);
            pass;
        pass;


    # retrieve the TOD data from g3 datafile
    if loaddata :
        # open pickle file to write
        outfile = open(outpath+'.pkl', 'wb');

        # get bolometer instance
        g3c, bolonames, time, start_mjd, end_mjd = loadbolo(filename, boloname, start, end, loadWHWP=loadWHWP, loadSlow=loadSlow);
 
        # get whwp angle
        if loadWHWP : whwp_angle = g3c.angle%(2.*np.pi); # [rad.] / g3c.angle is not-repeated value.
        #if loadWHWP : whwp_angle = (g3c.angle%(2.*np.pi)) * 360./(2.*np.pi) ; # [deg.]
        else        : whwp_angle = np.array([]);
        out.OUT('whwp angle data = {}'.format(whwp_angle),-1);

        # get slow daq data
        if 'slowData' in vars(g3c).keys() : 
            slowData = g3c.slowData;

            LStemp_slowtime = slowData['Lakeshore151']['time'];
            LStemp_slowData = slowData['Lakeshore151']['MODEL370_370A4A_T'];
            LStemp_time= np.array([ datetime.utcfromtimestamp(time) for time in LStemp_slowtime ]);
            # initialize data array
            LStemp_data = {label:[] for label in LStemp_map};
            #printVar(LStemp_slowData);
            for data in LStemp_slowData: # loop over time
                for k, label in enumerate(LStemp_map) : # loop over thermometers
                    LStemp_data[label].append(data[k]);
                    pass;
                pass;
            pass;

            SIMtemp_slowtime = slowData['SIM900']['time'];
            SIMtemp_slowData = slowData['SIM900']['SIM900']['TEMP'];
            SIMtemp_labels   = slowData['SIM900']['SIM900']['LABELS']; # array for each samplings
            SIMtemp_time= np.array([ datetime.utcfromtimestamp(time) for time in SIMtemp_slowtime ]);
            # initialize data array
            SIMtemp_data = {label:[] for label in SIMtemp_labels[0]};
            for i, data in enumerate(SIMtemp_slowData): # loop over time
                for k, label in enumerate(SIMtemp_labels[i]) : # loop over thermometers
                    SIMtemp_data[label].append(data[k]);
                    pass;
                pass;
            pass;

        # loop over bolonames to retrieve TOD data from g3 datafile
        for n, name in enumerate(bolonames):
            bolo=g3c.loadbolo(name,start=start_mjd,end=end_mjd)
            print(vars(g3c).keys());
            
            bolotime = g3c.bolotime;
            y=bolo[0].real * gaincal[n];
            # append to boloarray
            bolotime_boloarray.append(bolotime);
            y_boloarray       .append(y);
            pass;

        # write data to the pickle file
        pickle.dump(bolonames, outfile);
        pickle.dump(time, outfile);
        pickle.dump(whwp_angle, outfile);
        pickle.dump(bolotime_boloarray, outfile);
        pickle.dump(y_boloarray, outfile);
        if loadSlow :
            pickle.dump(LStemp_time, outfile);
            pickle.dump(LStemp_data, outfile);
            pickle.dump(SIMtemp_time, outfile);
            pickle.dump(SIMtemp_data, outfile);
        pass;

    # loop over bolonames to make plots
    all_fig, all_axs = plt.subplots(3,1);
    all_fig.tight_layout(rect=[0,0,0.8,0.96])
    plt.subplots_adjust(wspace=0.2, hspace=0.7, bottom=0.12, left=0.15)
    all_fig.set_size_inches(14,6.5);
    all_fig.suptitle(outname);
    for i, name in enumerate(bolonames):
        bolotime = bolotime_boloarray[i];
        y        = y_boloarray[i];

        fig, axs = plt.subplots(3,1);
        fig.tight_layout(rect=[0,0,0.8,0.96])
        plt.subplots_adjust(wspace=0.2, hspace=0.7, bottom=0.12, left=0.15)
        fig.set_size_inches(14,6.5);
        fig.suptitle(name)
        time_ax    = axs[0];
        timeTemp_ax= axs[1];
        angle_ax   = axs[2];
        all_time_ax    = all_axs[0];
        all_timeTemp_ax= all_axs[1];
        all_angle_ax   = all_axs[2];
    
        # average subtraction
        ave = np.average(y);
        y_subave = y - ave;
        # linear fit of bolo output
        linearfunc=np.poly1d(np.polyfit(bolotime,y,1))
        y_subDC = y - linearfunc(bolotime);

        # Draw x: time / y: output
        #time_ax.plot(time,y,label='Raw data', linestyle='-')
        time_ax.plot(time,y_subave,label='Raw data - ave. ({:.1e})'.format(ave), linestyle='-')
        #all_time_ax.plot(time,y,label='Raw data: {}'.format(name), linestyle='-', color=colors[i])
        all_time_ax.plot(time,y_subave,label='Raw data - ave. ({:.1e}): {}'.format(ave,name), linestyle='-', color=colors[i])
        # Plot cosmetic
        for ax in [time_ax, all_time_ax] :
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x',labelrotation=0,labelsize=16);
            ax.tick_params(axis='y',labelrotation=0,labelsize=16);
            ax.set_title('TOD', fontsize=10);
            ax.set_xlabel('Time', fontsize=16);
            ax.set_xlim(time[0],time[-1]);
            ax.set_ylabel('ADC output' if not calibGain else r'Power [$mK_\mathrm{RJ}$]', fontsize=16);
            ax.grid(True);
            ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0), framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
            pass;

        # Draw temperature
        if loadSlow :
            for k, (label, data) in enumerate(LStemp_data.items()) :
                #if data[0]>0. and k==1 : timeTemp_ax.plot(LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]);
                pass;
            for k, (label, data) in enumerate(SIMtemp_data.items()) :
                if data[0]>0. and k==3 : timeTemp_ax.plot(SIMtemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]);
                pass;
            # Plot cosmetic
            for ax in [timeTemp_ax, all_timeTemp_ax] :
                ax.set_ylim(3.85,3.95);
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
                ax.tick_params(axis='x',labelrotation=0,labelsize=16);
                ax.tick_params(axis='y',labelrotation=0,labelsize=16);
                ax.set_title('Temperature [K]', fontsize=10);
                ax.set_xlabel('Time', fontsize=16);
                ax.set_xlim(time[0],time[-1]);
                ax.set_ylabel('Temperature [K]', fontsize=16);
                ax.grid(True);
                #ax.set_yscale('log')
                ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
                pass;
            pass;

        # Draw x: WHWP angle / y: output
        if loadWHWP :
            #angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y,label='Raw data'.format(ave), marker='.', markersize=1,linestyle='',color='tab:blue')
            angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subave,label='Raw data - average({:e})'.format(ave), marker='.', markersize=1,linestyle='',color='tab:orange')
            #angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subDC,label='Raw data - DC', marker='.', markersize=1,linestyle='',color='tab:green')
            #all_angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y,label='Raw data: {}'.format(name), marker='.', markersize=1,linestyle='',color=colors[i])
            all_angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subDC,label='Raw data - average({:e}): {}'.format(ave,name), marker='.', markersize=1,linestyle='',color=colors[i])
            #all_angle_ax.plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subDC,label='Raw data - DC: {}'.format(name), marker='.', markersize=1,linestyle='',color=colors[i])
            pass;
        # Plot cosmetic
        for ax in [angle_ax, all_angle_ax] :
            ax.set_title('HWP angle', fontsize=10);
            ax.set_xlabel('HWP angle [deg.]', fontsize=16);
            ax.set_ylabel('ADC output' if not calibGain else r'Power [$mK_\mathrm{RJ}$]', fontsize=16);
            ax.tick_params(axis='x',labelrotation=0,labelsize=16);
            ax.tick_params(axis='y',labelrotation=0,labelsize=16);
            ax.grid(True);
            ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
            pass;

        out.OUT('Saving plot ({}.png) for [{}]:{}'.format(outpath, i, name),0);
        fig.savefig(outpath+'_{}.png'.format(name))
        fig.clear();
        pass

    all_fig.savefig(outpath+'.png');

    return 0;


if __name__=='__main__' :

    verbose = 1;
    filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
    boloname='PB20.13.13_Comb01Ch03,PB20.13.13_Comb01Ch24';
    outdir ='plot_for_JPS202109';
    #boloname=None;

    #startStr = None;
    #endStr   = None;

    startStr = "20210205_180510";
    endStr   = "20210205_180530";
    outname ='tod_A2_22.5deg';

    #startStr = "20210205_180230";
    #endStr   = "20210205_180400";
    #outname ='tod_A0_0deg';

    #startStr = "20210205_180230";
    #endStr   = "20210205_180231";
    #outname ='tod_A0_0deg_1sec';

    #startStr = "20210205_180200";
    #endStr   = "20210205_181830";
    #outname ='tod_A0-8_0-157.5angles';

    parser = argparse.ArgumentParser();
    parser.add_argument('--filename', default=filename, help='input g3 filename (default: {})'.format(filename));
    parser.add_argument('--boloname', default=boloname, help='boloname (default: {})'.format(boloname));
    parser.add_argument('--outdir', default=outdir, help='output directory for the plots (default: {})'.format(outdir));
    parser.add_argument('--outname', default=outname, help='output filename (default: {})'.format(outname));
    parser.add_argument('--start', default=startStr, help='start time string (default: {})'.format(startStr));
    parser.add_argument('--end', default=endStr, help='end time string (default: {})'.format(endStr));
    parser.add_argument('--noHWP', dest='loadWHWP', default=True, action='store_false', help='Not load WHWP data (default: {})'.format(True));
    parser.add_argument('-L', '--loadpickle', dest='loaddata', action='store_false', default=True, 
            help='Whether load g3 data file or not. If not load it, it will load pickle file before run. (default: True)');
    parser.add_argument('-v', '--verbose', default=verbose, type=int, help='verbosity level: A larger number means more printings. (default: {})'.format(verbose));
    args = parser.parse_args();

    out.setverbosity(args.verbose);
    out.OUT('outname = {}'.format(args.outname),1)
    out.OUT('loaddata = {}'.format(args.loaddata),1)
    if not (',' in boloname) : boloname = [args.boloname];
    else                     : boloname = args.boloname.split(',');
    plot(args.filename,boloname,args.start,args.end,outname=args.outname,outdir=args.outdir,loaddata=args.loaddata,loadWHWP=args.loadWHWP);
    pass;
