# loadbolo by using sa offline analysis pipeline
import os, sys
import numpy as np
from spt3g import core
from datetime import datetime
import Out;


# Import pipline libraries
sys.path.append(os.path.join(os.path.dirname(topdir), './library/simons_array_offline_software'))
# major pipeline libraries
import simons_array_python.sa_pipeline_inputs as sa_pi;
import simons_array_python.sa_observation as sa_ob;
import simons_array_python.sa_pipeline_filters as sa_pf;
# operator of time clip
import simons_array_python.sa_timestream_operators as sa_op;
# HWP angle calculator
import simons_array_python.sa_hwp as sa_hwp;


# tentative setting 
filename_tmp='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
boloname_tmp='PB20.13.13_Comb01Ch01';
startStr_tmp = "20210205_175000";
endStr_tmp   = "20210205_175010";



def loadbolo(
        runID=runID, 
        boloname=boloname_tmp, 
        start=startStr_tmp, 
        end=endStr_tmp,
        subUD=0,
        loadWHWP=True, loadSlow=False, out=None) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=0);

    # get all bolometer names
    out.OUT('Get all detector names',0);
    all_detectors = sa_op.gen_bolo_list();
    if boloname is None or boloname=="" :
        bolonames = np.array(all_detectors);
    elif isinstanec(boloname, str):
        if not boloname in all_detectors : 
            out.ERROR(f'There is no matched boloname for {boloname}.');
            return None
        else:
            bolonames = np.array([boloname])
            pass
    else: # if boloname is list of bolonames
        bolonames = []
        for bolo in boloname :
            if bolo in all_detectors : 
                bolonames.append(bolo);
            else:
                out.WARNING(f'There is no matched boloname for {bolo}.');
                out.WARNING(f'  --> Skip!!');
                pass;
            pass;
        bolonames = np.array(bolonames)
        if len(bolonames)==0:
            out.ERROR(f'There is no matched bolonames for {boloname}.');
            return None
        pass

    out.OUT('Initialize observation instance', 0)
    observation_tuple = (runID,subID);
    ob = sa_ob.Observation(observation_tuple)
    ob.detectors = bolonames
    ob.load_metadata()

    out.OUT('Data load operation', 0)
    pi = sa_pi.InputLevel0CachedByObsID(all_detectors=all_detectors, n_per_cache=len(all_detectors), 
        load_g3=True, load_gcp=True,
        load_slowdaq=loadSlow, load_hwp=loadWHWP, 
        load_dets=True, ignore_faulty_frame=True, record_frame_time=True);
    op_dataload = sa_pf.OperatorDataInitializer(pi)
    op_dataload.filter_obs(ob)

    out.OUT('Time clip operation', 0)
    out.OUT(f'  start = {start}')
    out.OUT(f'  end   = {end}')
    start_mjd = None if start==None else core.G3Time(start).mjd;
    end_mjd   = None if end  ==None else core.G3Time(end  ).mjd;
    op_timeclip = sa_pf.OperatorClipBeginEnd(begin_mjd=start_mjd, end_mjd=end_mjd)
    op_timeclip.filter_obs(ob)

    out.OUT('HWP angle calculation', 0)
    op_hwp = sa_hwp.HWPAngleCalculator(encoder_reference_angle=0.)
    op_hwp.filter_obs(ob)

    # Get TOD
    tod = ob.tod_list[0];
    out.OUT(f'keys {tod.cache.keys()}', 1);

'''
printlist(tod.read_times(), 'times');
printlist(tod.read('bolo_time'), 'bolo_times');
printlist(tod.read('raw_antenna_time_mjd'), 'raw_antenna_time_mjd');
printlist(tod.read('raw_az_pos'), 'raw_az_pos');
printlist(tod.read('13.13_15.90T-I'), 'TOD (13.13_15.90T-I)');
printlist(tod.read('hwp_angle'), 'HWP angle');
'''
    time = tod.read('bolo_time');
    out.OUT(f'time = {time}', 1);

    return tod, bolonames, time, start_mjd, end_mjd;










