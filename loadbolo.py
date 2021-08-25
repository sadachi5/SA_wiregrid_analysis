import os, sys
import numpy as np
import libg3py3 as libg3
from spt3g import core
from datetime import datetime
import Out;

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
        loadWHWP=True, loadSlow=False, out=None) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=0);

    # load g3 compressed file
    g3c=libg3.G3Compressed(filename,loadtime=False)
    out.OUT('filename = {}'.format(filename),0);
    out.OUT('boloname = {}'.format(boloname),0);
    out.OUT('time period = {}~{}'.format(start, end),0);
    out.OUT('loadWHWP   = {}'.format(loadWHWP),0);
    out.OUT(g3c.readout ,0);
    out.OUT(g3c.boloprop,0);
    out.OUT(np.array(g3c.bolonames_all),0);
    
    bolonames=[];
    if boloname is None or boloname=="" :
        bolonames = np.array(g3c.bolonames_all) 
    elif isinstance(boloname, str) : 
        bolonames = np.array([boloname]);
    else :
        bolonames = np.array(boloname);
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
    if(loadWHWP) :
        g3c.loadwhwp();
        pass;

    # load slow daq data
    if loadSlow :
        g3c.loadslowlog();
        pass;

    return g3c, bolonames, time, start_mjd, end_mjd;



