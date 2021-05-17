# plot2.py : Use simons_array_offline_softawre
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import libg3py3 as libg3
import math
from datetime import datetime
from scipy.optimize import curve_fit
from spt3g import core

from library.simons_array_offline_software.simons_array_python.sa_datafile import SimonsArrayDataFile

import Out;
verbosity=1;
out = Out.Out(verbosity=verbosity);

# tentative setting
filename_tmp='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
boloname_tmp='PB20.13.13_Comb01Ch01';
startStr_tmp = "20210205_175000";
endStr_tmp   = "20210205_175010";



# return : (g3c, bolonames, time, start_time(mjd), end_time(mjd) )
def loadbolo(
        filename=filename_tmp, 
        boloname=boloname_tmp, 
        start=startStr_tmp, 
        end=endStr_tmp,
        doWHWP=True) :
    g3c=libg3.G3Compressed(filename,loadtime=False)
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

    tmp = SimonsArrayDataFile(filename,start_mjd,end_mjd,verbose=True,do_PM=False, has_HWP=True, turnarounds=False);
    print('aho');
    # get time array
    # load time from start_mjd to end_mjd
    g3c.loadbolo('Do not load bolo.', start=start_mjd, end=end_mjd)
    time=[]
    for t in g3c.bolotime:
        #out.OUT(t,1);
        time.append(datetime.utcfromtimestamp(t/100000000))
        pass 
    
    # load whwp data
    if(doWHWP) :
        g3c.loadwhwp();
        pass;

    return g3c, bolonames, time, start_mjd, end_mjd;


def plot(filename, boloname=None, 
        start=None, end=None,
        outname='output', outdir='plot') :
    g3c, bolonames, time, start_mjd, end_mjd = loadbolo(filename, boloname, start, end);

    # get whwp angle
    whwp_angle = g3c.angle;

    # loop over bolonames
    for name in bolonames:
        fig, axs = plt.subplots(3,1);
        fig.set_size_inches(6,3);
        fig.suptitle(outname)
        time_ax  = axs[0];
        angle_ax = axs[1];
        bolo=g3c.loadbolo(name,start=start_mjd,end=end_mjd)
    
        y=bolo[0].real
        # linear fit of bolo output
        linearfunc=np.poly1d(np.polyfit(time,y,1))
        y_subDC = y - linearfunc(time);

        # Draw x: time / y: output
        time_ax.plot(time,y,label='Raw data', linestyle='-')
        time_ax.plot(time,y_subDC,label='Raw data - DC', linestyle='--')
    
        # Plot cosmetic
        time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        time_ax.xtics(rotation=30,fontsize=10);
        time_ax.set_title('TOD');
        time_ax.set_xlabel('Time');
        time_ax.set_ylabel('ADC output');

        # Draw x: WHWP angle / y: output
        angle_ax.plot(whwp_angle,y,label='Raw data', markerstyle='.', markersize=1,color='tab:orange')
        angle_ax.plot(whwp_angle,y_subDC,label='Raw data - DC', markerstyle='.', markersize=1,color='tab:blue')

        # Plot cosmetic
        angle_ax.set_title('WHWP angle');
        angle_ax.set_xlabel('WHWP angle [??]');
        angle_ax.set_ylabel('ADC output');

        if not os.path.isdir(outdir) : 
            out.WARNING('There is no output directory: {}'.format(outdir),0);
            out.WARNING('--> Make the directory.',0);
            os.mkdir(outdir);
            pass;

        fig.savefig(outdir+'/'+outname+'.png')
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

 
    return 0;


if __name__=='__main__' :


    filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
    boloname='PB20.13.13_Comb01Ch01';
    #boloname=None;

    #startStr = None;
    #endStr   = None;
    startStr = "20210205_175000";
    endStr   = "20210205_175010";


    outname='aho';

    plot(filename,boloname,startStr,endStr,outname);
    pass;
