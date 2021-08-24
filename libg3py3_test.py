from __future__ import division, absolute_import, print_function#, unicode_literals
import sys
import os
import glob
import pickle
import struct
import numpy as np
import tarfile
from matplotlib import pyplot as plt
import pandas
try:
    from pydfmux.core.utils.dfmux_exceptions import DFMUXError
except(ImportError):
    try:
        from . import pydfmux_dummy
    except(ValueError, ImportError):
        import pydfmux_dummy
    sys.modules['pydfmux'] = pydfmux_dummy
    sys.modules['pydfmux.core'] = pydfmux_dummy
    sys.modules['pydfmux.core.utils'] = pydfmux_dummy
    sys.modules['pydfmux.core.utils.dfmux_exceptions'] = pydfmux_dummy
    sys.modules['pydfmux.core.tuber'] = pydfmux_dummy
    from pydfmux.core.utils.dfmux_exceptions import DFMUXError

try:
    import numpy.core.multiarray
    import numpy.core._multiarray_umath
except (ImportError):
    try:
        import numpy.core.multiarray
        sys.modules['numpy.core._multiarray_umath'] = numpy.core.multiarray
    except(ImportError):
        import numpy.core._multiarray_umath
        sys.modules['numpy.core.multiarray'] = numpy.core._multiarray_umath

from spt3g import core, calibration, dfmux, gcp
try:
    from load_slowdaq import load_slow
except(ImportError):
    print('WARNING! There is an ImportError to import load_slowdaq.');
    pass
def mjd_to_G3Time(t):
    t2 = (int(np.floor(t)) - 40587) * int(core.G3Units.day)
    t2 += int((t % 1.) * core.G3Units.day)
    return core.G3Time(t2)

BOLO_SAMPLE_RATE = 20e6/2**17 # 152.587890625 Hz

dataroot = os.getenv('PB2DATA')
if dataroot is None:
    dataroots = {
        'tostada': "/data/pb2/ChileData",
        'nersc': "/global/project/projectdirs/polar/data/pb2a",
        'kekcc': "/group/cmb/polarbear/data/pb2a",
#        'kekcc-run38': "/group/cmb/polarbear/data/pb2a_integration/run38",
    }
    for dataroot in dataroots.values():
        if os.path.exists(dataroot):
            break

def set_dataroot(path):
    global dataroot, g3dir, slowdaqdir, slowdaqkeys, tuningdatadir
    dataroot = path
    g3dir = os.path.join(dataroot, "g3compressed")
    slowdaqdir = os.path.join(dataroot, "slowdaq")
    tuningdatadir = os.path.join(dataroot, "dropbolo_logs")
    missing = 0
    for subdir in [g3dir,slowdaqdir,tuningdatadir]:
        if not os.path.exists(subdir):
            print("WARNING: Cannot find %s"%subdir, file=sys.stderr)
            missing += 1
    if missing == 3:
        print("WARNING: libg3 cannot find data in %s"%(dataroot), file=sys.stderr)
        print(
            "If you have the data in different location, please run\n"\
            "    libg3.set_dataroot('/your/path/to/data/root')\n"\
            "or set $PB2DATA environment by adding the following line in ~/.bashrc\n"\
            "    export PB2DATA=/your/path/to/data/root", file=sys.stderr)
    print('slowdaqdir = {}'.format(slowdaqdir));
    #print(glob.glob(os.path.join(slowdaqdir, '2???????/2???????/slowdaq_????????_??????_Run????????_*.log.gz')));
    slowdaqkeys = sorted(list(set(
                    [os.path.basename(fn)[:-12] for fn in
                    glob.glob(os.path.join(slowdaqdir, '*_??-??:??:??'))] +
                    [os.path.basename(fn).split('_201')[0] for fn in
                    glob.glob(os.path.join(slowdaqdir, '*_20??????_??????*.pkl'))]
                    #+ glob.glob(os.path.join(slowdaqdir, '2???????/2???????/slowdaq_????????_??????_Run????????_*.log.gz'))
                    )))
    #print('slowdaqkeys = {}'.format(np.array(slowdaqkeys)));
    return

set_dataroot(dataroot)

def fillnan(x):
    bad = np.isnan(x).astype(int)
    if not bad.any():
        return x
    #print "Warning: %d nans are filled!"%bad.sum(), np.where(bad)[0]
    starts = np.where(np.diff(bad)>0)[0] + 1
    ends=np.where(np.diff(bad)<0)[0] + 1
    if bad[0] == 1:
        starts = np.append(0, starts)
    if bad[-1] == 1:
        ends = np.append(ends, x.size)
    assert starts.size == ends.size
    if starts[0]==0 and ends[0]==x.size:
        x[:] = 0.
        return x
    for s,e in zip(starts, ends):
        if s == 0:
            x[:e] = x[e]
        elif e == x.size:
            x[s:] = x[s-1]
        else:
            x[s:e] = np.linspace(x[s-1], x[e], e-s+2)[1:-1]
    return x

def fill_gaps(v, gaploc, gaplen, zero_fill=False):
    gaploc = np.asarray(gaploc)
    if gaploc.size == 0:
        return v
    assert len(v.shape) == 1
    total_gaplen = np.sum(gaplen)
    newlen = v.size + total_gaplen
    out = np.zeros(newlen, dtype=v.dtype)
    out_start = 0
    in_start = 0
    for gloc, glen in zip(gaploc, gaplen):
        nsamp = gloc - in_start
        out[out_start:out_start + nsamp] = v[in_start:gloc]
        if zero_fill:
            out[out_start + nsamp
                :out_start + nsamp + glen] = 0
        else:
            yo =  np.linspace(0.,
                              float(v[gloc] - v[gloc - 1]),
                              glen + 2)[1:-1] + v[gloc - 1]
            if out.dtype.kind in 'biu':
                yo = np.around(yo) # Note Numpy rounds *.5 to the nearest even value
            out[out_start + nsamp
                :out_start + nsamp + glen] = yo
        out_start += nsamp + glen
        in_start = gloc
    out[out_start:] = v[gloc:]
    return out


def findslowdaq(key, start=None, end=None):
    """ find slowdaq files for key between start and end """
    if not key in slowdaqkeys:
        raise KeyError(key)
    if start is None:
        start = -np.inf
    else:
        start = core.G3Time(start).GetFileFormatString()
        start = int(start[:8] + start[9:15])
    if end is None:
        end = np.inf
    else:
        end = core.G3Time(end).GetFileFormatString()
        end = int(end[:8] + end[9:15])
    #print('start = {}'.format(start));
    #print('end   = {}'.format(end  ));
    out1 = []
    '''
    to_int = lambda x: int('2017' + x[-11:-9] + x[-8:-6] + x[-5:-3] + x[-2:] + '00')
    fns1 = sorted(glob.glob(os.path.join(slowdaqdir,
                                         key.replace('*','[*]').replace('?','[?]')
                                         + '_??-??:??:??')))
    '''
    to_int = lambda x: int(x.split('_')[-4][:4]+x.split('_')[-4][4:8]+x.split('_')[-3]);
    fns1 = [key.split(slowdaqdir)[1]];
    for fn in fns1:
        i = to_int(fn)
        #print(fn, i);
        if i >= start and i < end:
            #out1.append(fn)
            out1.append(slowdaqdir+fn)
            continue
        elif i >= end:
            #out1.append(fn)
            out1.append(slowdaqdir+fn)
            return out1

    def to_int(x):
        x = os.path.basename(x)[len(key) + 1:][:15]
        return int(x[:8]+x[9:])
    fns2 = sorted(glob.glob(os.path.join(slowdaqdir,
                                         key.replace('*','[*]').replace('?','[?]')
                                         + '_20??????_??????*.pkl')))

    out2 = []
    for j,fn in enumerate(fns2):
        i = to_int(fn)
        if i == start:
            out2.append(fn)
        elif (i > start) and (i < end):
            if j == 0 and fns1 and fns1[-1] not in out1:
                out1.append(fns1[-1])
            elif j >= 1 and fns2[j-1] not in out2:
                out2.append(fns2[j-1])
            out2.append(fn)
            continue
        elif i >= end:
            if j == 0 and fns1 and fns1[-1] not in out1:
                out1.append(fns1[-1])
            elif j >= 1 and fns2[j-1] not in out2:
                out2.append(fns2[j-1])
            break
    return out1 + out2

def reshape_start_end(start, end):
    reftime = core.G3Time("20170101_000000")
    if start is not None:
        if isinstance(start, float) and start > reftime.mjd:
            istart = None
            tstart = mjd_to_G3Time(start).time
        elif isinstance(start, int) and core.G3Time(start) < reftime:
            istart = start
            tstart = None
        else:
            istart = None
            tstart = core.G3Time(start).time
    else:
        istart = tstart = None
    if end is not None:
        if isinstance(end, float) and end > reftime.mjd:
            iend = None
            tend = mjd_to_G3Time(end).time
        elif isinstance(end, int) and core.G3Time(end) < reftime:
            iend = end
            tend = None
        else:
            iend = None
            tend = core.G3Time(end).time
    else:
        iend = tend = None
    return (istart, iend, tstart, tend)

def isbetween(tstart=None, tend=None, istart=None, iend=None,
              i0=None, bolotime1=None):
    if tstart is None and tend is None and istart is None and iend is None:
        return slice(None)
    elif tstart is None and tend is None:
        assert i0 is not None, 'set i0'
        start = max(0, istart-i0) if istart is not None else None
        end = min(max(0, iend-i0), len(bolotime1)) if iend is not None else None
        return slice(start, end)
    elif istart is None and iend is None:
        assert bolotime1 is not None, 'set bolotime'
        t0 = min(bolotime1[0], tstart)
        start = int(np.searchsorted(bolotime1-t0, tstart-t0)) if tstart is not None else None
        end = int(np.searchsorted(bolotime1-t0, tend-t0)) if tend is not None else None
        return slice(start, end)
    else:
        raise ValueError('Combination of index and time is not supported.')

class BoloTimestreamBuffer(object):
    def __init__(self, bolonames,
                 tstart=None, tend=None, istart=None, iend=None):
        self.bolonames = bolonames
        self.tstart = tstart
        self.tend = tend
        self.istart = istart
        self.iend = iend
        #self.buffer_dets_I = []
        #self.buffer_dets_Q = []
        self.bolo = []
        self.bolotime = []
        self.samplerate = []
        self.finished = False
        self.i0 = 0

    def __call__(self, frame):
        try:
            if (frame.type is not core.G3FrameType.Scan
                or not 'RawTimestreams_I' in frame):
                del(frame)
                return
            if self.finished:
                return
            if 'DetectorSampleTimes' in frame:
                bolotime1 = np.asarray(
                    [x.time for x
                     in frame['DetectorSampleTimes']],
                    dtype=np.uint64)
            else:
                bolotime1 = np.asarray(
                    [x.time for x
                     in frame['RawTimestreams_I'].times()],
                    dtype=np.uint64)
            nsmpl0 = bolotime1.size
            mask = isbetween(self.tstart, self.tend,
                             self.istart, self.iend,
                             self.i0, bolotime1)
            self.i0 += nsmpl0
            if self.iend is not None:
                self.finished = (self.i0 >= self.iend)
            elif self.tend is not None:
                self.finished = (bolotime1[-1] >= self.tend)
            if nsmpl0 > 1:
                samplerate1 = (nsmpl0 - 1) / (bolotime1[-1] - bolotime1[0]) # same but faster than frame['RawTimestreams_I'].sample_rate
            else:
                samplerate1 = np.nan
            bolotime1 = bolotime1[mask]
            nsmpl = bolotime1.size
            if not nsmpl:
                del(nsmpl, bolotime1, frame)
                return
            self.bolotime.append(bolotime1)
            self.samplerate.append(samplerate1 * 1e8 * np.ones(bolotime1.shape, dtype=np.float32))
            del(bolotime1, samplerate1)
            if self.bolonames.size==0:
                del(nsmpl, frame)
                return

            reduced_ts_map_I = core.G3TimestreamMap()
            reduced_ts_map_Q = core.G3TimestreamMap()
            dets_to_load = self.bolonames
            if nsmpl0 == 1:
                for det in dets_to_load:
                    reduced_ts_map_I[det] = frame['RawTimestreams_I'][det]
                    reduced_ts_map_Q[det] = frame['RawTimestreams_Q'][det]
            else:
                for det in dets_to_load:
                    reduced_ts_map_I[det] = frame['RawTimestreams_I'][det][mask]
                    reduced_ts_map_Q[det] = frame['RawTimestreams_Q'][det][mask]
            #self.buffer_dets_I.append(reduced_ts_map_I)
            #self.buffer_dets_Q.append(reduced_ts_map_Q)
            try:
                self.bolo.append(
                    np.asarray(reduced_ts_map_Q, dtype=np.complex64) * np.complex64(1j)
                    + np.asarray(reduced_ts_map_I, dtype=np.float32))
            except(TypeError):
                self.bolo.append(np.vstack(
                    [ np.array(reduced_ts_map_Q[n], dtype=np.complex64) * np.complex64(1j)
                      + np.array(reduced_ts_map_I[n], dtype=np.float32)
                      for n in dets_to_load]))
            del(dets_to_load, reduced_ts_map_I, reduced_ts_map_Q, nsmpl, frame)
        except(RuntimeError):
            return

class GCPTimestreamBuffer(object):
    def __init__(self, tstart=None, tend=None, load_command=False):
        self.tstart = tstart
        self.tend = tend
        self.gcptime = []
        self.az = []
        self.el = []
        self.scan_flag = []
        if load_command:
            self.az_command = []
            self.el_command = []
        self.finished = False

    def __call__(self, frame):
        try:
            if (frame.type is not core.G3FrameType.Scan
                or not 'TrackerStatus' in frame):
                del(frame)
                return
            if self.finished:
                return
            tsf = frame['TrackerStatus']
            gcptime1 = np.asarray(
                [x.time for x
                 in tsf.time],
                dtype=np.uint64)
            tstart = self.tstart
            if self.gcptime:
                if tstart is None:
                    tstart = self.gcptime[-1][-1] + 1e5 # 0.001 s offset not to include the same sample
                else:
                    tstart = max(tstart, self.gcptime[-1][-1] + 1e5)
            mask = isbetween(tstart, self.tend,
                             bolotime1=gcptime1)
            if self.tend is not None:
                self.finished = (gcptime1[-1] >= self.tend)
            gcptime1 = gcptime1[mask]
            nsmpl = gcptime1.size
            if not nsmpl:
                del(nsmpl, gcptime1, mask, frame)
                return
            self.gcptime.append(gcptime1)
            self.az.append(np.asarray(
                tsf.az_pos, dtype=np.float32)[mask])
            self.el.append(np.asarray(
                tsf.el_pos, dtype=np.float32)[mask])
            self.scan_flag.append(np.asarray(
                tsf.scan_flag, dtype=np.bool)[mask])
            if hasattr(self, 'az_command'):
                self.az_command.append(np.asarray(
                    tsf.az_command, dtype=np.float32)[mask])
                self.el_command.append(np.asarray(
                    tsf.el_command, dtype=np.float32)[mask])
            del(nsmpl, gcptime1, mask, frame)
            return
        except(RuntimeError):
            return

class G3Compressed(object):
    def __init__(self,
                 g3files,
                 name=None,
                 loadbolo=False,
                 loadtime=True,
                 hwm=None):
        if len(g3files[0]) == 1:
            g3files = [g3files]
        if ((len(g3files) == 1)
            and not g3files[0].endswith('.g3')):
            g3files = os.path.join(g3files[0],'*.g3')
            g3files = sorted(glob.glob(g3files))
        print(g3files)
        self.g3files = g3files
        self.bolonames_all = None
        self.bolonames = None
        self.bolotime = None
        self.bolo = None
        self.hwm = hwm
        self.loadmisc()
        if loadbolo and loadtime:
            self.loadbolo(name)
        elif loadtime:
            self.loadbolo('Do not load bolo.')
            self.bolonames = None

    def loadmisc(self):
        readout = None
        boloprop = None
        housekeeping = None
        for frame in core.G3File(self.g3files[0]):
            if 'ReadoutSystem' in frame:
                readout = frame
            if 'NominalBolometerProperties' in frame:
                boloprop = frame['NominalBolometerProperties']
            if 'DfMuxHousekeeping' in frame:
                try:
                    housekeeping = frame['DfMuxHousekeeping']
                except(RuntimeError, MemoryError):
                    if housekeeping is None:
                        print('WARNING: Reading DfMuxHousekeeping raised error!')
                        housekeeping = 'cannot read'
            if (readout is not None and
                boloprop is not None and
                housekeeping is not None):
                break
        if isinstance(housekeeping, str):
            housekeeping = None
        if readout is not None:
            self.bolonames_all = sorted(
                readout['WiringMap'].keys())
        elif boloprop is not None:
            self.bolonames_all = sorted(boloprop.keys())
        if readout is not None and housekeeping is not None:
            housekeeping = HouseKeeping(readout['WiringMap'], housekeeping)
        if self.hwm is not None:
            boloprop2 = loadboloprop(self.hwm)
            if boloprop is None:
                if readout is not None:
                    boloprop = calibration.BolometerPropertiesMap()
                    for n in self.bolonames_all:
                        if n in boloprop2:
                            boloprop[n] = boloprop2[n]
                        else:
                            boloprop[n] = calibration.BolometerProperties()
                else:
                    boloprop = boloprop2
            else:
                for n in boloprop2.keys():
                    if n in boloprop:
                        boloprop[n] = boloprop2[n]
            del(boloprop2)
        self.readout = readout
        self.boloprop = boloprop
        self.housekeeping = housekeeping
        return

    def loadbolo(self, name=None, start=None, end=None):
        if name is not None and isinstance(name, str):
            self.bolonames = np.array(
                [n for n in self.bolonames_all
                 if n.startswith(name)], dtype=str)
        elif name is not None:
            self.bolonames = np.asarray(name, dtype=str)
        elif self.bolonames is None:
            self.bolonames = np.array(
                self.bolonames_all, dtype=str)
        else:
            self.bolonames = np.asarray([self.bolonames], dtype=str).flatten()
        #print 'load %d bolos'%len(self.bolonames)
        istart, iend, tstart, tend = reshape_start_end(start, end)
        print('libg3py3.py | istart={}, iend={}, tstart={}, tend={}'.format(istart, iend, tstart, tend))

        ts_buff = BoloTimestreamBuffer(self.bolonames, tstart, tend, istart, iend)
        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=self.g3files)
        pipe.Add(ts_buff)
        try:
            pipe.Run()
        except(RuntimeError):
            pass

        self.bolotime = np.concatenate(ts_buff.bolotime)
        self.samplerate = np.concatenate(ts_buff.samplerate)
        self.nojump = np.isclose(self.samplerate, BOLO_SAMPLE_RATE, atol=0., rtol=5.e-4)

        if not self.bolonames.size==0:
            #tod_map_I = core.G3TimestreamMap.concatenate(ts_buff.buffer_dets_I)
            #tod_map_Q = core.G3TimestreamMap.concatenate(ts_buff.buffer_dets_Q)
            #self.bolo = np.vstack(
            #    [ np.array(tod_map_Q[n], dtype=np.complex64) * np.complex64(1.j)
            #      + np.array(tod_map_I[n], dtype=np.float32)
            #      for n in self.bolonames ])
            bolo = np.hstack(ts_buff.bolo)
            bolo = bolo[self.bolonames.argsort().argsort()]
            self.combs = {}
            for i, name in enumerate(self.bolonames):
                comb = name.split('Ch')[0]
                if not comb in self.combs:
                    self.combs[comb] = [i]
                else:
                    self.combs[comb].append(i)
            self.combmask = {}
            self.bolotimes = {}
            nsamples = {}
            for comb in self.combs:
                self.combs[comb] = np.array(self.combs[comb])
                self.combmask[comb] = ~(np.isnan(bolo[self.combs[comb]]).all(axis=0))
                self.bolotimes[comb] = self.bolotime[self.combmask[comb]]
                nsamples[comb] = self.bolotimes[comb].size
                if self.samplerate.size > 0 and (nsamples[comb] == 0):
                    print('WARNING: %s has no valid data.'%comb)
            n0 = nsamples[comb]
            if all([(nsamples[comb] == n0) for comb in nsamples]):
                self.bolotime = np.median(np.vstack(list(self.bolotimes.values())), axis=0)
                if len(nsamples.keys()) > 1 and n0 > 0:
                    t0 = self.bolotime[-1]
                    for comb in self.combs:
                        if self.bolotimes[comb].size == 0:
                            continue
                        t1 = self.bolotimes[comb][-1]
                        if t1 != t0:
                            dt = t1 * 1.e-5 - t0 * 1.e-5
                            print('WARNING: %s has %f ms offset.'%(comb, dt))
            else:
                print('WARNING: Not equal samples! %s'%str(nsamples))
            bolo2 = []
            for name, b in zip(self.bolonames, bolo):
                comb = name.split('Ch')[0]
                b = b[self.combmask[comb]]
                fillnan(b)
                bolo2.append(b)
            self.bolo = np.asarray(bolo2)
            del(bolo2, bolo)
            if hasattr(self, 'TuningData'):
                self.loadtuningdata()
            return self.bolo
        del(ts_buff)
        if hasattr(self, 'whwp'):
            self.whwp.store_angle(self)
        return

    def loadgcp(self, start=None, end=None, interpolate=True, load_command=False):
        reftime = core.G3Time("20170101_000000")
        if start is not None:
            if isinstance(start, float) and start > reftime.mjd:
                istart = None
                tstart = mjd_to_G3Time(start).time
            elif isinstance(start, int) and core.G3Time(start) < reftime:
                istart = None
                tstart = self.bolotime[start]
            else:
                istart = None
                tstart = core.G3Time(start).time
        else:
            istart = tstart = None
        if end is not None:
            if isinstance(end, float) and end > reftime.mjd:
                iend = None
                tend = mjd_to_G3Time(end).time
            elif isinstance(end, int) and core.G3Time(end) < reftime:
                iend = None
                tend = self.bolotime[end]
            else:
                iend = None
                tend = core.G3Time(end).time
        else:
            iend = tend = None

        ts_buff = GCPTimestreamBuffer(tstart, tend, load_command=load_command)
        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=self.g3files)
        pipe.Add(ts_buff)
        try:
            pipe.Run()
        except(RuntimeError):
            pass

        self.gcptime = np.concatenate(ts_buff.gcptime)
        self.az = np.unwrap(np.concatenate(ts_buff.az))
        self.el = np.concatenate(ts_buff.el)
        self.scan_flag = np.concatenate(ts_buff.scan_flag)
        if hasattr(ts_buff, 'az_command'):
            self.az_command = np.concatenate(ts_buff.az_command)
            self.el_command = np.concatenate(ts_buff.el_command)
        if interpolate:
            if self.bolotime is None:
                self.loadbolo('Do not load bolo.', start=tstart, end=tend)
            self.az = np.interp(self.bolotime, self.gcptime, self.az)
            self.el = np.interp(self.bolotime, self.gcptime, self.el)
            self.scan_flag = np.interp(self.bolotime, self.gcptime, self.scan_flag.astype(float)) > 0.5
        dsf = np.diff(self.scan_flag.astype(int))
        rise = np.where(dsf == 1)[0] + 1
        fall = np.where(dsf == -1)[0] + 1
        if self.scan_flag[0]:
            rise = np.append(0, rise) #fall = fall[1:]
        if self.scan_flag[-1]:
            fall = np.append(fall, self.scan_flag.size)#rise = rise[:-1]
        scan_list = np.vstack([rise,fall]).T
        scanlength = np.diff(scan_list,axis=-1)[:,0]
        if len(scanlength) > 0:
            ok = np.isclose(scanlength, np.percentile(scanlength, 90),
                            atol=10, rtol=0)
            scan_list = scan_list[ok]
        self.scan_list = scan_list
        self.scan_flag[:] = False
        for s, e in self.scan_list:
            self.scan_flag[s:e] = True
        return

    def loadslowdaq(self, keys=None):
        if keys is None:
            keys = slowdaqkeys
        if self.bolotime is None:
            self.bolonames = []
            self.loadbolo()
            self.bolonames = None
        start = self.bolotime[0]
        end = self.bolotime[-1]
        print('start = {}'.format(start));
        print('end   = {}'.format(end  ));
        for key in keys:
            key_safe = 'Asterisk' if key == '*' else key
            if hasattr(self,key_safe):
                continue
            sdd = SlowDaqData(key, start, end)
            if sdd.time is None:
                continue
            setattr(self, key_safe, sdd)

    def loadslowlog(self):
        """loading slowdaq data using RunID from log files in json"""
        if not 'load_slowdaq' in sys.modules:
            #print 'Slowdaq is not imported. Quit loading slowdaq log files.'
            return

        runid = os.path.basename(self.g3files[0]).split('_')[0][3:]
        fn_slowdaqrunlist = os.path.join(slowdaqdir, "runlist_slowdaq.pkl")
        if os.path.exists(fn_slowdaqrunlist):
            with open(fn_slowdaqrunlist, 'rb') as f:
                slowdaqrunlist = pickle.load(f)
            runid = int(runid)
            if not slowdaqrunlist.has_key(runid):
                print('No slowdaq data for Run%d'%runid)
                return

            slowfiles = slowdaqrunlist[runid]['files']
            tstart = slowdaqrunlist[runid]['Tstart']
            tend = slowdaqrunlist[runid]['Tend']
            slowData = load_slow(slowfiles, tstart, tend, slowdaqdir)
        else:
            slowfiles = os.path.join(slowdaqdir,runid[:3]+'00000','*','slowdaq_*_Run%s_*.log.gz'%runid)
            slowfiles = sorted(glob.glob(slowfiles))
            if not slowfiles:
                print('No slowdaq data for Run%s'%runid)
                return
            tstart = self.bolotime[ 0]*1.e-8 - 10.;
            tend   = self.bolotime[-1]*1.e-8 + 10.;
            slowData = load_slow(slowfiles, tstart, tend, slowdir='')
        self.slowData = slowData

    def loadtuningdata(self):
        runid = os.path.basename(self.g3files[0]).split('_')[0]
        tuningdir = glob.glob(os.path.join(tuningdatadir,'????????',runid))[0]
        if self.bolonames is not None:
            tormod = lambda x: 'IceCrate_%03d.IceBoard_%d.Mezz_%d.ReadoutModule_%d'%(x.crate_serial,x.board_slot,x.module//4+1,x.module%4+1)
            rmods = sorted(list(set([tormod(self.readout['WiringMap'][ch]) for ch in self.bolonames])))
        else:
            rmods = None
        if hasattr(self,'TuningData'):
            self.TuningData.loadcomb(rmods)
        else:
            self.TuningData = TuningData(tuningdir,rmods)

    def loadwhwp(self, start=None, end=None):
        self.whwp = WHWPAngle(self, start=start, end=end)
        if self.bolotime is not None:
            self.whwp.store_angle(self)

class SlowDaqData(object):
    def __init__(self,key,start=None,end=None):
        self.key = key
        fns = findslowdaq(key,start,end)
        #print key+': '+' '.join(map(os.path.basename, fns))
        if len(fns)==0:
            self.time = None
            return
        print('filenames = {}'.format(fns));
        dfs = [pandas.read_pickle(fn) for fn in fns]
        # Read slow daq log and summarize to one dictionary
        for df in dfs :
            for msg in df :
                data = json.loads(msg);
                print(data);
                pass;
            pass;
        print(dfs);
        if len(fns)==1:
            self.df = dfs[0]
        else:
            self.df = pandas.concat(dfs)
        self.time = self.df['time'].values
        self.loaddata()
    def loaddata(self):
        for k in self.df.keys():
            if k == u'time' or k==u'index':
                continue
            k_safe = ''.join([x for x in k if x not in "!\"#$%&'()=~|-^\\`@[]{}+*;:,.<>/? "])
            if k_safe[0] in '0123456789':
                k_safe = '_' + k_safe
            if k == u'systime':
                toint = lambda x: core.G3Time(x).time
                x = map(toint,self.df[k].values)
                setattr(self,k,np.asarray(x))
            elif self.key=='Cryoboard'and (k == u't' or k == u'r'):
                toint = lambda x: struct.unpack('d'*16, x.split('"')[-2].decode('hex'))
                x = map(toint,self.df[k].values)
                setattr(self, k_safe, np.asarray(x))
            elif self.key=='StimEncoder' and (k == u'flagEdges'):
                toint = lambda x: struct.unpack('B'*20, x.split('"')[-2].decode('hex'))
                x = map(toint,self.df[k].values)
                setattr(self, k_safe, np.asarray(x))
            elif self.key=='StimEncoder' and (k == u'tEdges'):
                def toint(x):
                    format = 'd' if x.split('"')[5] == u'float64' else 'q'
                    return struct.unpack(format*20, x.split('"')[-2].decode('hex'))
                x = map(toint,self.df[k].values)
                setattr(self, k_safe, np.asarray(x, dtype=np.float64))
            elif isinstance(self.df[k][0],list):
                x = map(np.asarray, self.df[k].values)
                setattr(self, k_safe, np.asarray(x))
            else:
                setattr(self, k_safe, self.df[k].values)

class HouseKeeping(object):
    def __init__(self, wiringmap, housekeepingmap):
        self.wiringmap = wiringmap
        self.housekeepingmap = housekeepingmap

    def __getitem__(self, name):
        if 'Ch' in name:
            w = self.wiringmap[name]
            hkchaninfo = self.housekeepingmap[w.board_serial]\
                .mezz[w.module//4+1]\
                .modules[w.module%4+1]\
                .channels[w.channel+1]
        elif 'Comb' in name:
            for ch in range(1,42):
                if name+'Ch%02d'%ch in self.wiringmap:
                    break
            if ch==41:
                raise KeyError('%s does not exists!'%name)
            w = self.wiringmap[name+'Ch%02d'%ch]
            hkchaninfo = self.housekeepingmap[w.board_serial]\
                .mezz[w.module//4+1]\
                .modules[w.module%4+1]
        else:
            raise KeyError('set channel name or comb name')
        return hkchaninfo

    def select(self, condition):
        return np.array([ch for ch in self.wiringmap.keys() if condition(ch)])

    def selectstate(self, states):
        states = np.array([states],str).flatten()
        condition = lambda ch: str(self.__getitem__(ch).state) in states
        return self.select(condition)

    def selectrfrac(self, target=0.8, tolerance=0.05):
        condition = lambda ch: np.abs(self.__getitem__(ch).acheived_rfrac - target) < tolerance
        return self.select(condition)

class TuningDataBolo(object):
    def __init__(self,ch,d=None,fn=''):
        self.ch = ch
        self.rmod = '_'.join(os.path.basename(fn).split('_')[:5])
        if d is None or not ('RIV_log' in d or 'initial' in d):
            self.V = self.I_i = self.I_q = self.I = self.Cmag = np.array([np.nan])
            self.frequency = self.frequency0 = self.ADCtoA = self.acheived_rfrac = self.target_rfrac = np.nan
            self.state = 'nan'
            self.I_c = (self.I_i + 1.j*self.I_q)
            self.R_c = self.V / self.I_c
            return
        if 'RIV_log' in d:
            for k in 'V I_i I_q I Cmag'.split():
                if not k in d['RIV_log'][ch]:
                    setattr(self, k, np.array([np.nan]))
                else:
                    setattr(self, k, np.array(d['RIV_log'][ch][k],float))
            for k in 'frequency target_rfrac acheived_rfrac state'.split():
                if not k in d['subtargets'][ch]:
                    setattr(self, k, np.nan)
                else:
                    setattr(self, k, d['subtargets'][ch][k])
            self.frequency0 = self.frequency
            postkey = 'post_drop' if 'post_drop' in d else 'current'
            self.frequency = d[postkey][ch]['freq']
            self.ADCtoA = d[postkey][ch]['I']/d[postkey][ch]['Nmag']
        elif ('initial' in d):
            if not ch in d['initial']:
                self.V = self.I_i = self.I_q = self.I = self.Cmag = np.array([np.nan])
                self.frequency = self.frequency0 = self.ADCtoA = np.nan
            else:
                self.frequency0 = d['initial'][ch]['freq']
                if 'overbiased' in d and ch in d['overbiased']:
                    for k in 'V I_i I_q I Cmag'.split():
                        setattr(self, k, np.array([d['initial'][ch][k],d['overbiased'][ch][k]],float))
                    self.ADCtoA = d['overbiased'][ch]['I']/d['overbiased'][ch]['Nmag']
                    self.frequency = d['overbiased'][ch]['freq']
                else:
                    for k in 'V I_i I_q I Cmag'.split():
                        setattr(self, k, np.array([d['initial'][ch][k],],float))
                    self.ADCtoA = d['initial'][ch]['I']/d['initial'][ch]['Nmag']
                    self.frequency = self.frequency0
        if hasattr(self,'I_i') and hasattr(self,'I_q'):
            self.I_c = (self.I_i + 1.j*self.I_q)
            self.R_c = self.V / self.I_c

class TuningData(object):
    def __init__(self, fn, rmod=None):
        if isinstance(fn, list):
            fns = fn
        else:
            fns = [fn]
            if not fns[0].endswith('tar.gz'):
                fns = sorted(glob.glob(os.path.join(fn,'*.tar.gz')))
                if len(fns)==0:
                    raise ValueError("Cannot find tuning data. %s"%(fn))
        self.files = fns
        self.tfs = [tarfile.open(fn,'r') for fn in self.files]
        for i, tf in enumerate(self.tfs):
            fn2 = [n for n in tf.getnames() if n.endswith('data.tar.gz')]
            if fn2:
                self.tfs[i] = tarfile.open(
                    fileobj=tf.extractfile(fn2[0]),
                    mode='r')
        self.pkls = []
        for fn, tf in zip(self.files, self.tfs):
            if 'overbias_and_null' in fn:
                self.pkls.append(sorted([n for n in tf.getnames()
                                         if n.endswith('OUTPUT.pkl')
                                         or n.endswith('DFMUXError_data_dump.pkl')]))
            else:
                self.pkls.append(sorted([n for n in tf.getnames()
                                         if n.endswith('OUTPUT.pkl')
                                         or n.endswith('DFMUXError_data_dump.pkl')
                ]))
        self.data = {}
        self.opened = []
        self.loadcomb(rmod)

    def loadcomb(self,rmod=None):
        if rmod is not None:
            rmods = np.asarray(rmod,str).flatten()
            pkls2 = []
            missing = []
            for rmod in rmods:
                rmod = '_'.join(os.path.basename(rmod).split('_')[:5])
                print(rmod)
                found = [n for pkls1 in self.pkls for n in pkls1 if os.path.basename(n).startswith(rmod)]
                if not found:
                    missing.append(rmod)
                else:
                    pkls2 += found
            pkls2 = sorted(list(set(pkls2)))
            if missing:
                print('WARNING: Cannot find %s in %s'%(str(missing), str(self.pkls)))
        else:
            pkls2 = [n for pkls1 in self.pkls for n in pkls1]
        pkls2 = [fn for fn in pkls2 if not fn in self.opened]
        for i,fn in enumerate(pkls2):
            rmod = '_'.join(os.path.basename(fn).split('_')[:5])
            print('loading %s (%d/%d)'%(fn,i+1,len(pkls2)))
            tf = [tf for tf, pkls1 in zip(self.tfs, self.pkls)
                  if fn in pkls1][0]
            try:
                f = tf.extractfile(fn)
                d = pickle.load(f)
            except(UnicodeDecodeError):
                f = tf.extractfile(fn)
                d = pickle.load(f, encoding='latin1')
            f.close()
            self.readdict(d,fn)
            logfn = os.path.join(os.path.dirname(os.path.dirname(fn)), 'logs', rmod+'.txt')
            f = tf.extractfile(logfn)
            dlog = self.readlog(f)
            f.close()
            for ch in d['subtargets']:
                boloid = d['subtargets'][ch]['bolometer']
                newphase = dlog['newphase'][ch] if ch in dlog['newphase'] else np.nan
                oldphase = dlog['oldphase'][ch] if ch in dlog['oldphase'] else np.nan
                phase = np.ones_like(self.data[boloid].V) * oldphase
                if phase.size > 1:
                    phase[-1] = newphase
                self.data[boloid].phase = phase
            del(d, f)
            self.opened.append(fn)

    def readdict(self,d,fn=''):
        for ch in d['subtargets']:
            boloid = d['subtargets'][ch]['bolometer']
            if boloid in self.data:
                print('WARNING: Overwrite tuning data for %s'%(boloid))
            self.data[boloid] = TuningDataBolo(ch,d,fn)
        return

    def readlog(self,f):
        newphase = {}
        oldphase = {}
        for l in f:
            if b'new phase' in l:
                ch, offset, phase = (int(l.split()[17]), float(l.split()[20]), float(l.split()[23]))
                newphase[ch] = phase
                oldphase[ch] = phase + offset
        d = {}
        d['newphase'] = newphase
        d['oldphase'] = oldphase
        return d

    def keys(self):
        return sorted(self.data.keys())

    def __getitem__(self,k):
        if k in self.data:
            return self.data[k]
        else:
            self.data[k] = TuningDataBolo(k)
            return self.data[k]

    def getarray(self,bolonames,attr):
        return np.asarray([
                getattr(self.__getitem__(ch),attr)
                for ch in bolonames])

    def select(self,condition):
        return np.array([ch for ch in self.keys() if condition(ch)])

    def selectstate(self,states):
        states = np.array([states],str).flatten()
        condition = lambda ch: str(self.data[ch].state) in states
        return self.select(condition)

    def selectrfrac(self,target=0.8,tolerance=0.05):
        condition = lambda ch: np.abs(self.data[ch].acheived_rfrac - target) < tolerance
        return self.select(condition)

def loadboloprop(fn):
    fns = sorted(glob.glob(os.path.join(fn,'*','*bolos.csv')))
    bpm = calibration.BolometerPropertiesMap()
    for fn in fns:
        with open(fn,'r') as f:
            keys = None
            for line in f:
                if keys is None:
                    keys = line.split()
                    continue
                d = dict(zip(keys, line.split()))
                bp = calibration.BolometerProperties()
                wafer = 'None'
                if 'physical_name' in d:
                    bp.physical_name = d['physical_name']
                    wafer = d['physical_name'].split('_')[0]
                    bp.wafer_id = wafer
                if 'x_mm' in d:
                    bp.x_offset = float(d['x_mm'])
                if 'y_mm' in d:
                    bp.y_offset = float(d['y_mm'])
                if 'observing_band' in d and d['observing_band'] != 'None':
                    bp.band = float(d['observing_band']) * core.G3Units.GHz
                if 'polarization_angle' in d:
                    bp.pol_angle = float(d['polarization_angle']) * core.G3Units.deg
                if 'name' in d:
                    bpm[wafer+'/'+d['name']] = bp
    return bpm

class WHWPAngle(object):
    CLK_FREQ = 16e6
    N_TICK = 50000
    def __init__(self, g3c, start=None, end=None):
        istart, iend, tstart, tend = reshape_start_end(start, end)
        i0 = 0
        bolotime = []
        clk_cnts = []
        encoder_cnts = []
        cnts_at_ref = []
        clk_at_ref = []
        irig_time = []
        irig_clk = []
        for fn in g3c.g3files:
            #print 'loading', fn
            try:
                for frame in core.G3File(fn):
                    if frame.type is core.G3FrameType.Timepoint and 'DfMux' in frame:
                        bolotime1 = np.asarray([frame['EventHeader'].time], dtype=np.uint64)
                        mask = isbetween(tstart, tend,
                                         istart, iend,
                                         i0, bolotime1)
                        i0  += bolotime1.size
                        bolotime.append(bolotime1[mask])
                        continue
                    if frame.type is not core.G3FrameType.Scan:
                        del(frame)
                        continue
                    if 'whwp_clk_cnts' in frame:
                        clk_cnts.append(np.array(
                            frame['whwp_clk_cnts'],
                            dtype=np.uint32))
                        irig_time.append(np.array(
                            [x.time for x
                             in frame['whwp_irig_start_time']],
                            dtype=np.uint64))
                    else:
                        clk_cnts.append(np.array(
                            frame['whwp_encoder_clk_cnts'],
                            dtype=np.uint32))
                        irig_time.append(np.array(
                            [x.time for x
                             in frame['whwp_irig_start_times']],
                            dtype=np.uint64))
                    encoder_cnts.append(np.array(
                        frame['whwp_encoder_cnts'],
                        dtype=np.uint16))
                    cnts_at_ref.append(np.array(
                        frame['whwp_encoder_cnts_at_ref'],
                        dtype=np.uint16))
                    clk_at_ref.append(np.array(
                        frame['whwp_clk_cnts_at_ref'],
                        dtype=np.uint32))
                    irig_clk.append(np.array(
                        frame['whwp_irig_start_clk_cnts'],
                        dtype=np.uint32))
                    if 'DetectorSampleTimes' in frame:
                        bolotime1 = np.asarray(
                            [x.time for x
                             in frame['DetectorSampleTimes']],
                            dtype=np.uint64)
                        mask = isbetween(tstart, tend,
                                         istart, iend,
                                         i0, bolotime1)
                        i0  += bolotime1.size
                        bolotime.append(bolotime1[mask])
                        del(bolotime1, mask)

                    del(frame)
            except(RuntimeError):
                #print fn, 'broken!'
                pass
            del(fn)
        bolotime = np.hstack(bolotime)
        clk_cnts = np.hstack(clk_cnts)
        encoder_cnts = np.hstack(encoder_cnts)
        cnts_at_ref = np.hstack(cnts_at_ref)
        clk_at_ref = np.hstack(clk_at_ref)
        irig_time = np.hstack(irig_time)
        irig_clk = np.hstack(irig_clk)


        clk_nrev = np.cumsum(np.append(
            0, np.diff(clk_cnts.astype(np.int64)) < 0))
        irig_nrev = np.cumsum(np.append(
            0, np.diff(irig_clk.astype(np.int64)) < 0))
        clk_ref_nrev = np.cumsum(np.append(
            0, np.diff(clk_at_ref.astype(np.int64)) < 0))

        # my check
        #'''
        print ('nrev:', clk_nrev[-1], irig_nrev[-1]);
        __clk_cnts = clk_cnts;
        __irig_clk = irig_clk;
        __clk_at_ref = clk_at_ref;
        #'''

        clk_cnts = (clk_nrev << 32) + clk_cnts # int64
        irig_clk = (irig_nrev << 32) + irig_clk # int64
        clk_at_ref = (clk_ref_nrev << 32) + clk_at_ref # int64

        # my check
        #'''
        from utils import plottmp;
        plottmp(np.arange(len(clk_cnts)),[clk_cnts, __clk_cnts], ny=2, i=0, outname='clk_cnts');
        plottmp(np.arange(len(irig_clk)),[irig_clk, __irig_clk], ny=2, i=0, outname='irig_clk');
        plottmp(np.arange(len(clk_at_ref)),[clk_at_ref, __clk_at_ref], ny=2, i=0, outname='clk_at_ref');
        del __clk_cnts, __irig_clk, __clk_at_ref; 
        #'''
        del(clk_nrev, irig_nrev, clk_ref_nrev)

        #irig_ok = (irig_time > bolotime[0] - 10**10) * (irig_time < bolotime[-1] + 10**10) # margin 100 s
        irig_ok = (irig_time > bolotime[0] - 10**9) * (irig_time < bolotime[-1] + 10**9) # margin 10 s
        if not irig_ok.all():
            print('Remove strange timestamp!:', end='')
            print(' '.join([core.G3Time(x).GetFileFormatString()
                            for x in irig_time[~irig_ok]]))
            #del(x)
        if irig_ok.sum() <= 1:
            del(g3c)
            print(list(locals().keys()))
            for key in list(locals().keys()):
                if key == 'self' or key=='key':
                    continue
                setattr(self, key, locals()[key])
            #raise ValueError('Timestamp of WHWP encoder is not available!')
            print('Timestamp of WHWP encoder is not available!')
            #return
        else:
            irig_time = irig_time[irig_ok]
            irig_clk = irig_clk[irig_ok]
            irig_ok = np.diff(irig_time) > 0
            if not irig_ok.all():
                print('Remove strange timestamp!:', end='')
                print(' '.join([core.G3Time(x).GetFileFormatString()
                                for x in irig_time[~irig_ok]]))
                #del(x)
            irig_ok = np.append(irig_ok[0], irig_ok)
            if irig_ok.sum() <= 1:
                raise ValueError('Timestamp of WHWP encoder is not available!')
            irig_time = irig_time[irig_ok]
            irig_clk = irig_clk[irig_ok]

            enc_ok = slice(
                np.searchsorted(clk_cnts, irig_clk[0]),
                np.searchsorted(clk_cnts, irig_clk[-1],
                                'right'))
            clk_cnts = clk_cnts[enc_ok]
            encoder_cnts = encoder_cnts[enc_ok]

            ref_ok = slice(
                np.searchsorted(clk_at_ref, irig_clk[0]),
                np.searchsorted(clk_at_ref, irig_clk[-1],
                                'right'))
            clk_at_ref = clk_at_ref[ref_ok]
            cnts_at_ref = cnts_at_ref[ref_ok]

        clk_cnts_diff = np.diff(clk_cnts).astype(np.uint32)
        #irig_clk_diff = np.diff(irig_clk).astype(np.uint32)
        enc_cnts_diff = np.diff(encoder_cnts)

        # my check
        '''
        np.set_printoptions(edgeitems=10);
        print('clk_at_ref {}={}'.format(len(clk_at_ref), clk_at_ref));
        print('cnts_at_ref {}={}'.format(len(cnts_at_ref), cnts_at_ref));
        print('clk_cnts {}={}'.format(len(clk_cnts), clk_cnts));
        print('encoder_cnts {}={}'.format(len(encoder_cnts), encoder_cnts));

        for i in range(10) : print(cnts_at_ref[i]);
        tmp = np.diff(cnts_at_ref);
        print('diff {} = {}'.format(len(tmp),tmp));
        tmp = (tmp!=50000.)
        print('diff!=50000 {} = {}'.format(len(tmp),tmp));
        tmp = ~np.convolve([True,True], tmp);
        print('~convolve([True,Treu], diff!=50000) {} = {}'.format(len(tmp),tmp));
        tmp2 = cnts_at_ref % 50000.;
        tmp3 = np.median(tmp2[tmp]);
        print('ok {} = {}'.format(len(tmp),tmp));
        print('ref cnts in evolution {} = {}'.format(len(tmp2),tmp2));
        print('referenece = {}'.format(tmp3));

        print('diff(cnts_at_ref!=50000) {}={}'.format(len(tmp), tmp));
        plt.plot(clk_at_ref, tmp);
        plt.savefig('aho.pdf');
        plt.close();
        '''

        dclk = np.median(clk_cnts_diff)
        mask = np.isclose(clk_cnts_diff, dclk,
                          atol=60, rtol=0)
        denc = np.median(enc_cnts_diff)
        mask *= np.isclose(enc_cnts_diff, denc,
                           atol=3, rtol=0)
        del(dclk, denc)
        dclk = np.mean(clk_cnts_diff[mask])
        denc = np.mean(enc_cnts_diff[mask])
        enc_speed0 = denc/dclk
        #print 'dclk:', dclk
        #print 'denc:', denc
        #print 'roughspeed: ', \
        #    enc_speed0 / self.N_TICK * self.CLK_FREQ
        del(mask, enc_cnts_diff)

        enc_cnts_err = encoder_cnts - encoder_cnts[0] \
            - (enc_speed0 * (clk_cnts - clk_cnts[0]))
        enc_cnts_err = \
            (enc_cnts_err + 2.**15)%(2.**16) - 2.**15
        enc_cnts_err = np.unwrap(
            enc_cnts_err * (np.pi * 2 / 2**15)
            ) * (2**15 / np.pi / 2)
        enc_speed = enc_speed0 \
            + enc_cnts_err[-1] / (clk_cnts[-1] - clk_cnts[0])
        #print 'speed:', \
        #    enc_speed / self.N_TICK * self.CLK_FREQ
        enc_const = encoder_cnts[0] \
            + enc_speed * (clk_cnts - clk_cnts[0])
        enc_nrev = np.floor(
            enc_const / 2**16).astype(np.int32)
        encoder_cnts = (enc_nrev << 16) + encoder_cnts
        enc_cnts_err = encoder_cnts - enc_const
        encoder_cnts[enc_cnts_err < - 2**14] += 2**16
        encoder_cnts[enc_cnts_err > 2**14] -= 2**16
        enc_cnts_err = \
            (enc_cnts_err + 2.**15)%(2.**16) - 2.**15
        #print 'err:', enc_cnts_err.max(), enc_cnts_err.min()
        enc_speed0 *= self.CLK_FREQ / self.N_TICK
        enc_speed *= self.CLK_FREQ / self.N_TICK
        del(enc_cnts_err, enc_const, enc_nrev)

        gaploc = np.where(clk_cnts_diff > 1000)[0] + 1
        gaplen = np.round(
            (clk_cnts_diff[gaploc - 1] - dclk) / dclk /50).astype(int)*50
        del(dclk, denc, clk_cnts_diff)

        gapmask = np.ones(encoder_cnts.shape, dtype=np.bool)
        for i, l in zip(gaploc, gaplen):
            print('gap!', i, l)
            del(i,l)
        clk_cnts = fill_gaps(clk_cnts, gaploc, gaplen)
        encoder_cnts = fill_gaps(
            encoder_cnts, gaploc, gaplen)
        gapmask = fill_gaps(
            gapmask, gaploc, gaplen,
            zero_fill=True).astype(np.bool)

        ### MY IMPLEMENTATION ###
        # Select good(ok) cnts_at_ref 
        '''
        np.set_printoptions(edgeitems=10);
        cnts_at_ref_cumsum = np.cumsum( np.concatenate([[cnts_at_ref[0]],np.diff(cnts_at_ref)]) );
        print('cnts_at_ref (uint16) max ={}'.format(np.iinfo(cnts_at_ref.dtype)));
        print('cnts_at_ref {}={}'.format(len(cnts_at_ref), cnts_at_ref));
        print('cnts_at_ref_cumsum {}={}'.format(len(cnts_at_ref_cumsum), cnts_at_ref_cumsum));
        print('diff(cnts_at_ref) {}={}'.format(len(np.diff(cnts_at_ref)), np.diff(cnts_at_ref)));
        # get reference (offset) for the HWP angle
        ok_diff_at_ref     = (np.diff(cnts_at_ref)==self.N_TICK); # diff==50000 or not
        # If before or after diff is not 50000, convolve returns True. 
        # The final value is True if the both of before and after diff is 50000.
        ok_ref = ~np.convolve([True,True],~ok_diff_at_ref); 
        if np.all(ok_ref==False):
            print('WARNING!! There is no good reference signal. (# of not-ok reference: {})'.format(np.sum_nonzero(ok_BeforeAfter_ref==False)));
            pass;
        cnts_at_ref_per_rot = cnts_at_ref_cumsum%self.N_TICK ;
        reference_offset = np.median(cnts_at_ref_per_rot[ok_ref]);
        print('reference_offset = {}'.format(reference_offset));
        if (cnts_at_ref_per_rot[ok_ref] != reference_offset).any():
            count_error = cnts_at_ref_per_rot[ok_ref].max() - cnts_at_ref_per_rot[ok_ref].min()
            print('Warning: WHWP encoder may have error count by %d'%(count_error))
            pass;
        plt.plot(cnts_at_ref_per_rot[ok_ref]);
        plt.savefig('aho.pdf');
        plt.close();
        '''
        # Search for 1st reference clock counts
        clk_at_ref0 = clk_at_ref[0];
        def getOffset(clk0) :
            np.set_printoptions(edgeitems=10);
            print('clk_at_ref0 = {}'.format(clk0));
            print('clk_cnts ({}) = {}'.format(len(clk_cnts), clk_cnts));
            print('encoder_cnts ({}) = {}'.format(len(encoder_cnts), encoder_cnts));
            clk_index_after_ref0 = np.where(clk_cnts>clk0)[0][0]; # Index of clk right after clk0.
            print('clk_index_after_ref0 = {}'.format(clk_index_after_ref0));

            reference_offset = 0.;
            if clk_index_after_ref0==0 :
                print('WARNING!! There is no data before ref0.');
                pass;
            else :
                index_pair_ref0 = np.array([clk_index_after_ref0-1, clk_index_after_ref0]);
                clk_pair_ref0   = np.array([clk_cnts[index_pair_ref0[0]], clk_cnts[index_pair_ref0[1]]]);
                enc_pair_ref0   = np.array([encoder_cnts[index_pair_ref0[0]], encoder_cnts[index_pair_ref0[1]]]);
                if enc_pair_ref0[0]>enc_pair_ref0[1] :
                    print('WARNING!! There is overflow in the enc_pair_ref0 = {}'.format(enc_pair_ref0));
                    pass;
                enc_ref0 = np.interp([clk0], clk_pair_ref0, enc_pair_ref0);
                print('clk_cnts before ref0 = {}'.format(clk_pair_ref0[0]));
                print('clk_cnts after  ref0 = {}'.format(clk_pair_ref0[1]));
                print('enc_cnts before ref0 = {}'.format(enc_pair_ref0[0]));
                print('enc_cnts after  ref0 = {}'.format(enc_pair_ref0[1]));
                print('enc_ref0 = {}'.format(enc_ref0 ));
                correct_cnts_after_ref0 = (enc_pair_ref0[1] - enc_ref0) + (int)(enc_pair_ref0[1]/self.N_TICK)*self.N_TICK;
                offset_cnts_after_ref0  = correct_cnts_after_ref0 - enc_pair_ref0[1];
                print('correct_cnts_after_ref0 = {}'.format(correct_cnts_after_ref0));
                print('offset_cnts_after_ref0  = {}'.format(offset_cnts_after_ref0 ));
                reference_offset = offset_cnts_after_ref0;
                pass;
            return reference_offset;
        reference_offset = getOffset(clk_at_ref0);
        getOffset(clk_at_ref[-1]);
        ### END OF MY IMPLEMENTATION ###

        #print 'WARNING: Reconstruction of WHWP angle origin is not implemented yet.'
        angle0 = 0 # angle of the HWP corresponding to the encoder global reference mark
        enc_angle = (encoder_cnts - reference_offset + angle0) \
                    * (np.pi * 2 / self.N_TICK)

        '''
        irig_ok = (irig_time > bolotime[0] - 10**10) * (irig_time < bolotime[-1] + 10**10) # margin 100 s
        if not irig_ok.all():
            print('Remove strange timestamp!:', end='')
            print(' '.join([core.G3Time(x).GetFileFormatString()
                            for x in irig_time[~irig_ok]]))
            #del(x)
        if irig_ok.sum() <= 1:
            del(g3c)
            print(list(locals().keys()))
            for key in list(locals().keys()):
                if key == 'self' or key=='key':
                    continue
                setattr(self, key, locals()[key])
            #raise ValueError('Timestamp of WHWP encoder is not available!')
            print('Timestamp of WHWP encoder is not available!')
            return
        irig_time = irig_time[irig_ok]
        irig_clk = irig_clk[irig_ok]
        irig_ok = np.diff(irig_time) > 0
        if not irig_ok.all():
            print('Remove strange timestamp!:', end='')
            print(' '.join([core.G3Time(x).GetFileFormatString()
                            for x in irig_time[~irig_ok]]))
            #del(x)
        irig_ok = np.append(irig_ok[0], irig_ok)
        if irig_ok.sum() <= 1:
            raise ValueError('Timestamp of WHWP encoder is not available!')
        irig_time = irig_time[irig_ok]
        irig_clk = irig_clk[irig_ok]
        '''
        clk_speed = (irig_time[-1]-irig_time[0])/(irig_clk[-1]-irig_clk[0])
        enc_time = np.interp(
            clk_cnts - irig_clk[0],
            irig_clk - irig_clk[0],
            ((irig_time - irig_time[0]).astype(np.int64)
             - clk_speed * (irig_clk - irig_clk[0])))
        enc_time += clk_speed * (clk_cnts - irig_clk[0])
        enc_time = enc_time.astype(np.uint64) + irig_time[0]
        del(g3c)
        for key in list(locals().keys()):
            if key == 'self':
                continue
            setattr(self, key, locals()[key])

    def store_angle(self, g3c):
        if g3c.bolotime is None:
            bolotime = self.bolotime
        else:
            bolotime = g3c.bolotime
        if self.irig_time[0] < bolotime[0]:
            tmin = self.irig_time[0]
        else:
            tmin = bolotime[0]
        g3c.speed = ((self.enc_angle[-1] - self.enc_angle[0])
                     / (self.enc_time[-1] - self.enc_time[0])
                     * (100000000 / (np.pi * 2.)))
        enc_angle0 = (g3c.speed * (np.pi * 2. / 100000000)
                      * (self.enc_time - self.enc_time[0]))
        g3c.angle = np.interp(
            bolotime - tmin,
            self.enc_time - tmin,
            self.enc_angle - self.enc_angle[0] - enc_angle0)
        g3c.angle += (g3c.speed * (np.pi * 2. / 100000000)
                      * (bolotime - self.enc_time[0]).astype(np.int64))
        g3c.angle += self.enc_angle[0]
        g3c.bolo_time = (bolotime - bolotime[0]) / 86400.e8
        g3c.firstmjd = g3c.bolo_time[0] + 7. / 86400
        g3c.lastmjd = g3c.bolo_time[-1] - 7. / 86400
        istart = np.searchsorted(g3c.bolo_time, g3c.firstmjd)
        iend = np.searchsorted(g3c.bolo_time, g3c.lastmjd)
        g3c.scanlist = [[istart, iend - istart]]
        g3c.nt = g3c.bolo_time.size
        g3c.f = {}
        del(tmin, istart, iend)
        return
