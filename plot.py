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

from slowdaq_map import *;
from loadbolo import loadbolo;
from utils import colors, printVar, rad_to_deg;
import Out;
out = Out.Out(verbosity=0);

# tentative setting 
filename_tmp='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
boloname_tmp='PB20.13.13_Comb01Ch01';
startStr_tmp = "20210205_175000";
endStr_tmp   = "20210205_175010";


"""
# return : (g3c, bolonames, time, start_time(mjd), end_time(mjd) )
def loadbolo(
        filename=filename_tmp, 
        boloname=boloname_tmp, 
        start=startStr_tmp, 
        end=endStr_tmp,
        loadWHWP=True, loadSlow=False) :
    g3c=libg3.G3Compressed(filename,loadtime=False)
    out.OUT('filename = {}'.format(filename),0);
    out.OUT('boloname = {}'.format(boloname),0);
    out.OUT('time period = {}~{}'.format(start, end),0);
    out.OUT('loadWHWP   = {}'.format(loadWHWP),0);
    out.OUT(g3c.readout ,0);
    out.OUT(g3c.boloprop,0);
    out.OUT(np.array(g3c.bolonames_all),0);
    
    bolonames=[];
    if boloname==None :
        bolonames = np.array(g3c.bolonames_all) 
    else :
        bolonames = np.array([boloname]);
        pass;
 
    # get G3Time of start/end time
    start_mjd = None if start==None else core.G3Time(start).mjd;
    end_mjd   = None if end  ==None else core.G3Time(end  ).mjd;
    out.OUT('start_mjd = {}'.format(start_mjd),0);
    out.OUT('end_mjd   = {}'.format(end_mjd  ),0);

    # get time array
    # load time from start_mjd to end_mjd
    g3c.loadbolo('Do not load bolo.', start=start_mjd, end=end_mjd)
    time=[]
    for t in g3c.bolotime:
        #out.OUT(t,1);
        time.append(datetime.utcfromtimestamp(t/100000000))
        pass 
    out.OUT('# of data points = {}'.format(len(time)),0);
    
    # load whwp data
    if loadWHWP :
        g3c.loadwhwp();
        pass;

    # load slow daq data
    if loadSlow :
        g3c.loadslowlog();
        pass;

    return g3c, bolonames, time, start_mjd, end_mjd;
"""


def plot(filename, boloname=None, 
        start=None, end=None,
        outname='output', outdir='plot', loaddata=True, loadWHWP=True,loadSlow=True) :

    # initialize data list
    bolonames  = [];
    time       = [];
    whwp_angle = [];
    bolotime_boloarray = [];
    y_boloarray        = [];

    outpath = outdir+'/'+outname;
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
        for name in bolonames:
            bolo=g3c.loadbolo(name,start=start_mjd,end=end_mjd)
            print(vars(g3c).keys());
            
            bolotime = g3c.bolotime;
            y=bolo[0].real
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
    for i, name in enumerate(bolonames):
        bolotime = bolotime_boloarray[i];
        y        = y_boloarray[i];

        fig, axs = plt.subplots(3,1);
        fig.tight_layout(rect=[0,0,0.8,0.96])
        plt.subplots_adjust(wspace=0.2, hspace=0.7)
        fig.set_size_inches(14,6);
        fig.suptitle(outname)
        time_ax    = axs[0];
        timeTemp_ax= axs[1];
        angle_ax   = axs[2];
    
        # average subtraction
        ave = np.average(y);
        y_subave = y - ave;
        # linear fit of bolo output
        linearfunc=np.poly1d(np.polyfit(bolotime,y,1))
        y_subDC = y - linearfunc(bolotime);

        # Draw x: time / y: output
        time_ax.plot(time,y_subave,label='Raw data - ave. ({:.1e})'.format(ave), linestyle='-')
        #time_ax.plot(time,y_subDC,label='Raw data - DC', linestyle='--')
        # Plot cosmetic
        time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        time_ax.tick_params(axis='x',labelrotation=0,labelsize=10);
        time_ax.set_title('TOD');
        time_ax.set_xlabel('Time');
        time_ax.set_xlim(time[0],time[-1]);
        time_ax.set_ylabel('ADC output');
        time_ax.grid(True);
        time_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0), framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);

        # Draw temperature
        if loadSlow :
            for k, (label, data) in enumerate(LStemp_data.items()) :
                #if data[0]>0. and k==1 : timeTemp_ax.plot(LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]);
                pass;
            for k, (label, data) in enumerate(SIMtemp_data.items()) :
                if data[0]>0. and k==3 : timeTemp_ax.plot(SIMtemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]);
                pass;
            # Plot cosmetic
            timeTemp_ax.set_ylim(3.85,3.95);
            timeTemp_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            timeTemp_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            timeTemp_ax.tick_params(axis='x',labelrotation=0,labelsize=10);
            timeTemp_ax.set_title('Temperature [K]');
            timeTemp_ax.set_xlabel('Time');
            timeTemp_ax.set_xlim(time[0],time[-1]);
            timeTemp_ax.set_ylabel('Temperature [K]');
            timeTemp_ax.grid(True);
            #timeTemp_ax.set_yscale('log')
            timeTemp_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
            pass;

        # Draw x: WHWP angle / y: output
        if loadWHWP :
            angle_ax.plot(rad_to_deg(whwp_angle),y_subave,label='Raw data - average({:e})'.format(ave), marker='.', markersize=1,linestyle='',color='tab:orange')
            angle_ax.plot(rad_to_deg(whwp_angle),y_subDC,label='Raw data - DC', marker='.', markersize=1,linestyle='',color='tab:blue')
            pass;
        # Plot cosmetic
        angle_ax.set_title('WHWP angle');
        angle_ax.set_xlabel('WHWP angle [deg.]');
        angle_ax.set_ylabel('ADC output');
        angle_ax.grid(True);
        angle_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);

        out.OUT('Saving plot ({}.png) for [{}]:{}'.format(outpath, i, boloname),0);
        fig.savefig(outpath+'.png')
        plt.close()
        pass
#axs[0][0].set_xlim(0.01, 50)
#axs[0][0].set_title('time')
#axs[0][0].set_xscale('log')
#axs[0][0].set_xlabel('Frequency [Hz]')
#axs[0][0].set_ylabel('Power Spectrum Density(I) [dB/Hz]')
#axs[0][0].set_ylabel('PSD of I [dB/Hz]')
#axs[0][0].grid(True)
#axs[1][1].xaxis.set_major_locator(mdates.MinuteLocator);
#axs[1][1].xaxis.set_major_formatter(mdates.DateFormatter(g_datetimeformat));
#axs[1][1].xaxis.set_minor_locator(mdates.SecondLocator);
#pyplot.legend(loc ='upper left',
#  bbox_to_anchor=(1.00, 1),
#  #mode = 'expand',
#  #framealpha = 1,
#  frameon = False,
#  fontsize = 7,
#  title=legendtitle,
#  borderaxespad=0., );
    return 0;


if __name__=='__main__' :

    verbose = 1;
    filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
    boloname='PB20.13.13_Comb01Ch01';
    outdir ='aho';
    outname ='aho';
    #boloname=None;

    #startStr = None;
    #endStr   = None;

    startStr = "20210205_174900";
    endStr   = "20210205_175000";
    #outname  = '1749-1759';

    #startStr = "20210205_174930";
    #endStr   = "20210205_175100";
    #outname  = 'A0_0deg';

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
    out.OUT('loaddata = {}'.format(args.loaddata),1)
    plot(args.filename,args.boloname,args.start,args.end,args.outname,outdir=args.outdir,loaddata=args.loaddata,loadWHWP=args.loadWHWP);
    pass;
