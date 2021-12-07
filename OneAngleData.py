import os, sys
import argparse;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle
import math
import copy
from scipy.optimize import curve_fit
from datetime import datetime

# database
from slowdaq_map import *;

# my library
import Out;
from utils import rad_to_deg, theta0to2pi, colors;

# Use pipeline to read data
# Use runID / Not use filename in OneAngleData
usePipeline = True; 
from loadbolo_pipeline import loadbolo;

# Use library implemented by Satoru to read data
# Use filename / Not use runID in OneAngleData
#usePipeline = False; 
#from loadbolo import loadbolo;
#from loadbolo_v2 import loadbolo;

class OneAngleData :


    # runID will be used only if usePipeline=True.
    # filename will be used only if usePipeline=False.
    def __init__(   self        ,   
                    runID=0, filename='', boloname=None , 
                    start=None  , end=None      ,
                    loaddata=True, loadWHWP=True, loadSlow=True,
                    outname='output'    , outdir='plot' , 
                    loadpickledir='',
                    out=None            , verbosity=0   ) :

        # initialize data variables
        self.m_bolonames  = []; # bolometer readout name array
        self.m_time       = []; # time array
        self.m_whwp_angle = []; # whwp angle array
        self.m_bolotime_array = []; # bolotime-array x bolos
        self.m_y_array        = []; # y-array x bolos
        self.m_LStemp_time   = []; # time for Lakeshore151
        self.m_LStemp_data   = {}; # Lakeshore151 temperature data {'label':[TOD data],..}
        self.m_SIMtemp_time   = []; # time for SIM900
        self.m_SIMtemp_data   = {}; # SIM900 temperature data {'label':[TOD data],..}
         
        # initialize input variables
        self.m_runID = 0;
        self.m_filename = '';
        self.m_boloname = '';
        self.m_start = '';
        self.m_end   = '';
        self.m_loaddata = True;
        self.m_loadWHWP = True;
        self.m_loadSlow = True;
        self.m_outname  = '';
        self.m_outdir   = '';
        self.m_outpath  = '';
        self.m_loadpickldir = '';
        self.m_out = None; # class Out

        # assign input variables
        self.m_runID     = runID;
        self.m_filename  = filename;
        self.m_boloname  = boloname;
        self.m_start     = start;
        self.m_end       = end;
        self.m_loaddata  = loaddata;
        self.m_loadWHWP  = loadWHWP;
        self.m_loadSlow  = loadSlow;
        self.m_outname   = outname;
        self.m_outdir    = outdir;
        self.m_loadpickledir = loadpickledir;

        # initialize Out
        if out==None : self.m_out = Out.Out(verbosity=verbosity);
        else         : self.m_out = out;

        # Check output directory
        if not os.path.isdir(self.m_outdir) : os.makedirs(self.m_outdir);

        # open & read pickle file
        if not self.m_loaddata :
            loadfilename   = '';
            checkfilename  = '{}/{}.pkl'.format(self.m_outdir       , self.m_outname);
            checkfilename2 = '{}/{}.pkl'.format(self.m_loadpickledir, self.m_outname);
            self.m_out.OUT('candidate input pickle file1: {}'.format(checkfilename ),0);
            self.m_out.OUT('candidate input pickle file2: {}'.format(checkfilename2),0);
            if   os.path.isfile(checkfilename ) : loadfilename = checkfilename ;
            elif os.path.isfile(checkfilename2) : loadfilename = checkfilename2;
            self.m_out.OUT('load data from {}'.format(loadfilename),0);

            if len(loadfilename)>0:
                outfile = open(loadfilename, 'rb');
                self.m_bolonames = pickle.load(outfile);
                self.m_time      = pickle.load(outfile);
                self.m_whwp_angle= pickle.load(outfile);
                self.m_bolotime_array  = pickle.load(outfile);
                self.m_y_array         = pickle.load(outfile);
                if self.m_loadSlow:
                    self.m_LStemp_time = pickle.load(outfile);
                    self.m_LStemp_data = pickle.load(outfile);
                    self.m_SIMtemp_time = pickle.load(outfile);
                    self.m_SIMtemp_data = pickle.load(outfile);
                    self.m_out.OUTVar(self.m_LStemp_time,-1);
                    self.m_out.OUTVar(self.m_LStemp_data,-1);
                    pass;
                self.m_out.OUT(f'time (size={len(self.m_time)})= {self.m_time}',-1);
                self.m_out.OUT(f'whwp angle data (size={len(self.m_whwp_angle)})= {self.m_whwp_angle}',-1);
                self.m_out.OUT(f'bolo time[0] (size={len(self.m_bolotime_array[0])})= {self.m_bolotime_array[0]}',-1);
                self.m_out.OUT(f'bolo data[0] (size={len(self.m_y_array[0])})= {self.m_y_array[0]}',-1);
            else : # if there is no pickle file, load g3 datafile
                self.m_loaddata = True;
                pass;
            pass;
        self.m_outpath = self.m_outdir+'/'+self.m_outname;

    

        # retrieve the TOD data from g3 datafile
        if self.m_loaddata :
            # open pickle file to write
            self.m_out.OUT('load raw data and write to {}.pkl'.format(self.m_outpath),-1);
            outfile = open(self.m_outpath+'.pkl', 'wb');
    
            # get bolometer instance
            detector_names = []; # list to loop over detectors
            if usePipeline :
                self.m_out.OUT(f'loadbolo({self.m_runID}, {self.m_boloname}, {self.m_start}, {self.m_end}, loadHWP={self.m_loadWHWP}, loadSlow={self.m_loadSlow}, out={self.m_out})',0)
                g3c, self.m_bolonames, detector_names, self.m_time, start_mjd, end_mjd = \
                    loadbolo(self.m_runID, self.m_boloname, self.m_start, self.m_end,
                            loadHWP=self.m_loadWHWP, loadSlow=self.m_loadSlow, out=self.m_out);
            else :
                g3c, self.m_bolonames, self.m_time, start_mjd, end_mjd = \
                    loadbolo(self.m_filename, self.m_boloname, self.m_start, self.m_end,
                            loadHWP=self.m_loadWHWP, loadSlow=self.m_loadSlow, out=self.m_out);
                detector_names = self.m_bolonames;
                pass;
            self.m_out.OUT(f'time (size={len(self.m_time)})= {self.m_time}',-1);
     
            # get whwp angle
            if self.m_loadWHWP : self.m_whwp_angle = copy.deepcopy(g3c.angle)%(2.*np.pi); # [rad.] / g3c.angle is not-repeated value.
            #if self.m_loadWHWP : self.m_whwp_angle = (g3c.angle%(2.*np.pi)) * 360./(2.*np.pi) ; # [deg.]
            else  : self.m_whwhp_angle = np.array([]);
            self.m_out.OUT(f'whwp angle data (size={len(self.m_whwp_angle)})= {self.m_whwp_angle}',-1);

            # get slow daq data
            if self.m_loadSlow and 'slowData' in vars(g3c).keys() : 
                slowData = g3c.slowData;
         
                LStemp_slowtime = copy.deepcopy(slowData['Lakeshore151']['time']);
                LStemp_slowData = copy.deepcopy(slowData['Lakeshore151']['MODEL370_370A4A_T']);
                self.m_LStemp_time= np.array([ datetime.utcfromtimestamp(time) for time in LStemp_slowtime ]);
                # initialize data array
                self.m_LStemp_data = {label:[] for label in LStemp_map};
                self.m_out.OUTVar(LStemp_slowData,-1);
                for data in LStemp_slowData: # loop over time
                    for k, label in enumerate(LStemp_map) : # loop over thermometers
                        self.m_LStemp_data[label].append(data[k]);
                        pass;
                    pass;
                pass;
         
                SIMtemp_slowtime = copy.deepcopy(slowData['SIM900']['time']);
                SIMtemp_slowData = copy.deepcopy(slowData['SIM900']['SIM900']['TEMP']);
                SIMtemp_labels   = copy.deepcopy(slowData['SIM900']['SIM900']['LABELS']); # array for each samplings
                self.m_SIMtemp_time= np.array([ datetime.utcfromtimestamp(time) for time in SIMtemp_slowtime ]);
                # initialize data array
                self.m_SIMtemp_data = {label:[] for label in SIMtemp_labels[0]};
                for i, data in enumerate(SIMtemp_slowData): # loop over time
                    for k, label in enumerate(SIMtemp_labels[i]) : # loop over thermometers
                        self.m_SIMtemp_data[label].append(data[k]);
                        pass;
                    pass;
                pass;

    
            # loop over bolonames to retrieve TOD data from g3 datafile
            for name in detector_names:
                self.m_out.OUT('detector name = {}'.format(name), 1);
                bolos = g3c.loadbolo(name) if usePipeline else g3c.loadbolo(name,start=start_mjd,end=end_mjd)
                bolotime = copy.deepcopy(g3c.bolotime);
                self.m_out.OUT('bolo = {}'.format(bolos),-1);
                y=copy.deepcopy(bolos[0].real)
                self.m_out.OUT(f'bolo time (size={len(bolotime)}= {bolotime}',-1);
                self.m_out.OUT(f'bolo data (size={len(y)}= {y}',-1);
                # append to boloarray
                self.m_bolotime_array.append(bolotime);
                self.m_y_array       .append(y);
                pass;
    
            # write data to the pickle file
            pickle.dump(self.m_bolonames, outfile);
            pickle.dump(self.m_time, outfile);
            pickle.dump(self.m_whwp_angle, outfile);
            pickle.dump(self.m_bolotime_array, outfile);
            pickle.dump(self.m_y_array, outfile);
            if self.m_loadSlow :
                pickle.dump(self.m_LStemp_time, outfile);
                pickle.dump(self.m_LStemp_data, outfile);
                pickle.dump(self.m_SIMtemp_time, outfile);
                pickle.dump(self.m_SIMtemp_data, outfile);
                pass;
            outfile.close();
            del g3c;
            pass;
        # End of loaddata
            
        return;
    ### End of __init__ ###



    def plot(self) :
    
        # loop over bolonames to make plots
        for i, boloname in enumerate(self.m_bolonames):
            bolotime = self.m_bolotime_array[i];
            y        = self.m_y_array[i];
    
            fig, axs = plt.subplots(3,1);
            fig.tight_layout(rect=[0,0,0.8,0.96])
            plt.subplots_adjust(wspace=0.2, hspace=1.0)
            fig.set_size_inches(14,6);
            fig.suptitle(self.m_outname)
            time_ax  = axs[0];
            temp_ax  = axs[1];
            angle_ax = axs[2];
        
            # average subtraction
            ave = np.average(y);
            y_subave = y - ave;
            # linear fit of bolo output
            linearfunc=np.poly1d(np.polyfit(bolotime,y,1))
            y_subDC = y - linearfunc(bolotime);
    
            # Draw x: time / y: output
            time_ax.plot(self.m_time,y_subave,label='Raw data - ave. ({:.1e})'.format(ave), linestyle='-')
            #time_ax.plot(self.m_time,y_subDC,label='Raw data - DC', linestyle='--')
            # Plot cosmetic
            time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            time_ax.tick_params(axis='x',labelrotation=0,labelsize=10);
            time_ax.set_title('TOD');
            time_ax.set_xlabel('Time');
            time_ax.set_ylabel('ADC output');
            time_ax.grid(True);
            time_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);

            # Draw temperature
            if self.m_loadSlow :
                for k, (label, data) in enumerate(self.m_LStemp_data.items()) :
                    #if data[0]>0. and k==1 : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # 350mK stripline
                    if data[0]>0. and k==4 : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # 350mK stage
                    #if data[0]>0. and k==10 : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # 250mK stage top
                    #if data[0]>0. and k==12 : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # 250mK stage bottom
                    #if data[0]>0. and k==13 : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # 250mK stage top
                    #if data[0]>0. : temp_ax.plot(self.m_LStemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]);
                    pass;
                for k, (label, data) in enumerate(self.m_SIMtemp_data.items()) :
                    #if data[0]>0. and k==3 : temp_ax.plot(self.m_SIMtemp_time,data,label='{}'.format(label), linestyle='-',color=colors[k]); # MAIN PLATE
                    pass;
                # Plot cosmetic
                #temp_ax.set_ylim(3.85,3.95); # MAIN PLATE
                #temp_ax.set_ylim(0.178,0.182); # 250mK stage
                temp_ax.set_ylim(0.418,0.422); # 350mK stage
                temp_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                temp_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
                temp_ax.tick_params(axis='x',labelrotation=0,labelsize=10);
                temp_ax.set_title('Temperature [K]');
                temp_ax.set_xlabel('Time');
                temp_ax.set_xlim(self.m_time[0],self.m_time[-1]);
                temp_ax.set_ylabel('Temperature [K]');
                temp_ax.grid(True);
                #temp_ax.set_yscale('log')
                temp_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
                pass;
    
            # Draw x: WHWP angle / y: output
            if self.m_loadWHWP :
                angle_ax.plot(rad_to_deg(theta0to2pi(self.m_whwp_angle)),y_subave,label='Raw data - average({:e})'.format(ave), marker='.', markersize=1,linestyle='',color='tab:orange')
                angle_ax.plot(rad_to_deg(theta0to2pi(self.m_whwp_angle)),y_subDC,label='Raw data - DC', marker='.', markersize=1,linestyle='',color='tab:blue')
                pass;
            # Plot cosmetic
            angle_ax.set_title('WHWP angle');
            angle_ax.set_xlabel('WHWP angle [deg.]');
            angle_ax.set_ylabel('ADC output');
            angle_ax.grid(True);
            angle_ax.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.02,1.0),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
    
            if not os.path.isdir(outdir) : 
                out.WARNING('There is no output directory: {}'.format(outdir));
                out.WARNING('--> Make the directory.');
                os.makedirs(outdir);
                pass;
    
            out.OUT('Saving plot ({}_{}.png) for [{}]:{}'.format(self.m_outpath, boloname, i, boloname),1);
            fig.savefig('{}_{}.png'.format(self.m_outpath, boloname));
            plt.close()
            pass; # end of loop over bolonames

        return 0;
        ### End of plot() ###


    #===== End of OneAngleData() =====#





if __name__=='__main__' :

    verbose = 0;
    filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
    runID   =22300609;
    boloname='PB20.13.13_Comb01Ch01';
    outname ='aho';
    outdir  ='./plot';
    #boloname=None;

    #startStr = None;
    #endStr   = None;

    startStr = "20210205_174900";
    endStr   = "20210205_175900";
    outname  = '1749-1759';

    #startStr = "20210205_174930";
    #endStr   = "20210205_175100";
    #outname  = 'A0_0deg';


    parser = argparse.ArgumentParser();
    parser.add_argument('--runID', default=runID, type=int, help='runID (default: {}, ONLY used if usePipeline=True)'.format(runID));
    parser.add_argument('--filename', default=filename, help='input g3 filename (default: {}, ONLY used if usePipeline=False)'.format(filename));
    parser.add_argument('--boloname', default=boloname, help='boloname (default: {})'.format(boloname));
    parser.add_argument('--outdir', default=outdir, help='output directory for the plots (default: {})'.format(outdir));
    parser.add_argument('--outname', default=outname, help='output filename (default: {})'.format(outname));
    parser.add_argument('--start', default=startStr, help='start time string (default: {})'.format(startStr));
    parser.add_argument('--end', default=endStr, help='end time string (default: {})'.format(endStr));
    parser.add_argument('--noHWP', dest='loadWHWP', default=True, action='store_false', help='Not load WHWP data (default: {})'.format(True));
    parser.add_argument('--loadSlow', dest='loadSlow', default=False, action='store_true', help='Load slow DAQ data (default: {})'.format(False));
    parser.add_argument('-L', '--loadpickle', dest='loaddata', action='store_false', default=True, 
            help='Whether load g3 data file or not. If not load it, it will load pickle file before run. (default: True)');
    parser.add_argument('-v', '--verbose', default=verbose, type=int, help='verbosity level: A larger number means more printings. (default: {})'.format(verbose));
    args = parser.parse_args();


    out = Out.Out(args.verbose);
    out.OUT('loaddata = {}'.format(args.loaddata),1)


    onedata = OneAngleData(args.runID, args.filename, args.boloname, 
            args.start, args.end, 
            args.loaddata, args.loadWHWP, args.loadSlow, 
            outname=args.outname,outdir=args.outdir,out=out);
    onedata.plot();
    pass;
