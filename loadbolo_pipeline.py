# loadbolo by using sa offline analysis pipeline
import os, sys
import numpy as np
import copy
from spt3g import core
from datetime import datetime
import Out


# Import pipline libraries
topdir = os.environ['PWD']
sys.path.append(os.path.join(os.path.dirname(topdir), './library/simons_array_offline_software'))
# major pipeline libraries
import simons_array_python.sa_pipeline_inputs as sa_pi;
import simons_array_python.sa_observation as sa_ob;
import simons_array_python.sa_pipeline_filters as sa_pf;
# operator of time clip
import simons_array_python.sa_timestream_operators as sa_op;
# HWP angle calculator
import simons_array_python.sa_hwp as sa_hwp;
# for my_gen_bolo_list()
import simons_array_python.sa_sql as sa_sql;


class MyBolo:
    def __init__(self, I, Q):
        self.real = I;
        self.imag = Q;
        pass;


class MyTOD:
    # NOTE: loadSlow is NOT implemented.
    def __init__(self, tod, loadHWP=False, loadSlow=False) :
        self.tod = tod;
        self.loadHWP = loadHWP;
        self.loadSlow = loadSlow;
        if self.loadHWP: self.angle = copy.deepcopy(self.tod.read('hwp_angle'));
        if self.loadSlow: 
            self.slowData = [];
        self.bolotime = copy.deepcopy(tod.read('bolo_time'));
        self.bolo = []; # filled by loadbolo()
        pass;

    def __del__(self):
        del self.tod;
        del self.loadHWP;
        del self.loadSlow;
        del self.bolotime;
        del self.bolo;

    def keys(self): return self.tod.cache.keys();

    def loadbolo(self, bolonames):
        if isinstance(bolonames, str): bolonames = [bolonames];
        mybolos = [];
        for boloname in bolonames :
            I = copy.deepcopy(self.tod.read(f'{boloname}-I'));
            Q = copy.deepcopy(self.tod.read(f'{boloname}-Q'));
            mybolo = MyBolo(I,Q);
            mybolos.append(mybolo);
            pass;
        return mybolos;

# Original in simons_array_offline_software/simons_array_python/sa_timestream_operators.py
def my_gen_bolo_list(out=None):
    bolodb = sa_sql.db_impl.focalplane_db;
    hwm_db = sa_sql.db_impl.hwm_db;
    detectors_to_use = [];
    rns = [];
    
    for fp_row in bolodb:
        boloname = fp_row.bolo_name;
        fp_readout_name = fp_row.readout_name;
        if fp_readout_name == None: # skip bolos not known to be connected in fp.db
            print(f'skip {boloname}')
            continue
        #print(boloname, fp_readout_name);
        detectors_to_use.append(boloname);
        rns.append(fp_readout_name);
        pass;

    return detectors_to_use, rns;

def getbolonames(readoutname=None, out=None):
    # initialize Out
    if out==None : out = Out.Out(verbosity=0);
    all_bolonames, all_readoutnames = my_gen_bolo_list(out=out);
    if readoutname is None or readoutname=="" :
        bolonames = np.array(all_bolonames);
        readoutnames = np.array(all_readoutnames);
    elif isinstance(readoutname, str):
        if not readoutname in all_readoutnames : 
            out.ERROR(f'There is no matched readoutname for {readoutname}.');
            return None;
        else:
            print('readoutname index = ',all_readoutnames.index(readoutname), len(all_bolonames));
            readoutnames = np.array([readoutname]);
            bolonames = np.array([all_bolonames[all_readoutnames.index(readoutname)]]);
            pass;
    else: # if readoutname is list of readoutnames
        readoutnames = [];
        bolonames = [];
        for rname in readoutname :
            if rname in all_readoutnames : 
                readoutnames.append(rname);
                bolonames.append(all_bolonames[all_readoutnames.index(rname)]);
            else:
                out.WARNING(f'There is no matched readoutname for {rname}.');
                out.WARNING(f'  --> Skip!!');
                pass;
            pass;
        readoutnames = np.array(readoutnames);
        bolonames = np.array(bolonames);
        if len(readoutnames)==0:
            out.ERROR(f'There is no matched readoutnames for {readoutname}.');
            return None;
        pass;
    return readoutnames, bolonames;



def loadbolo(
        runID, 
        readoutname, 
        start, 
        end,
        subID=0,
        loadHWP=True, loadSlow=False, out=None) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=0);

    # get all bolometer names
    out.OUT('Get all detector names',0);
    readoutnames, bolonames = getbolonames(readoutname, out);
    out.OUT(f'readoutnames (size={len(readoutnames)}): {readoutnames}',1);
    out.OUT(f'bolonames (size={len(bolonames)}): {bolonames}',1);

    out.OUT('Initialize observation instance', 0);
    observation_tuple = (runID,subID);
    ob = sa_ob.Observation(observation_tuple);
    ob.detectors = bolonames;
    ob.load_metadata();

    out.OUT('Data load operation', 0);
    pi = sa_pi.InputLevel0CachedByObsID(all_detectors=bolonames, n_per_cache=len(bolonames), 
        load_g3=True, load_gcp=True,
        load_slowdaq=loadSlow, load_hwp=loadHWP, 
        load_dets=True, ignore_faulty_frame=True, record_frame_time=True);
    op_dataload = sa_pf.OperatorDataInitializer(pi);
    op_dataload.filter_obs(ob);

    out.OUT('Time clip operation', 0);
    out.OUT(f'  start = {start}');
    out.OUT(f'  end   = {end}');
    start_mjd = None if start==None else core.G3Time(start).mjd;
    end_mjd   = None if end  ==None else core.G3Time(end  ).mjd;
    op_timeclip = sa_pf.OperatorClipBeginEnd(begin_mjd=start_mjd, end_mjd=end_mjd);
    op_timeclip.filter_obs(ob);

    out.OUT('HWP angle calculation', 0);
    op_hwp = sa_hwp.HWPAngleCalculator(encoder_reference_angle=0.);
    op_hwp.filter_obs(ob);

    # Get TOD
    tod = ob.tod_list[0];
    out.OUT(f'keys {tod.cache.keys()}', 1);
    time = tod.read('bolo_time');
    out.OUT(f'time = {time}', 1);
    mytod = MyTOD(tod, loadHWP=loadHWP, loadSlow=loadSlow);
    out.OUT(f'mytod = {mytod}',1);
    out.OUT(f'bolonames = {bolonames}', 1);
    out.OUT(f'time = {time}', 1);
    out.OUT(f'start_mjd = {start_mjd}', 1);
    out.OUT(f'end_mjd = {end_mjd}', 1);

    return mytod, readoutnames, bolonames, time, start_mjd, end_mjd;



if __name__=='__main__':

    # tentative setting 
    runID = 22300609;
    #boloname_tmp='13.13_15.90T';
    readoutname_tmp='PB20.13.13_Comb01Ch01';
    startStr_tmp = "20210205_175000";
    endStr_tmp   = "20210205_175010";

    g3c, bolonames, time, start_mjd, end_mjd = loadbolo(
        runID=runID, 
        readoutname=readoutname_tmp, 
        start=startStr_tmp, 
        end=endStr_tmp,
        subID=0,
        loadHWP=True, loadSlow=False);

    bolos = g3c.loadbolo(bolonames);
    bolotime = g3c.bolotime;
    y = bolos[0].real;

    print(f'bolos (size={len(bolos)}) = {bolos}');
    print(f'bolotime (size={len(bolotime)}) = {bolotime}');
    print(f'y (size={len(y)}) = {y}');

    pass;
