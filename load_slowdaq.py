#!/usr/bin/env python

import os,sys
import time
import numpy as np
from slowdaq.netstring import File
from slowdaq.netarray import np_deserialize
try:
    import ujson as json
except:
    import json

try:
    strobj = basestring # for python2
except(NameError):
    strobj = str # for python3

def expand_netarray(list_na):
    output = []
    for na in list_na:
        array, chksum = np_deserialize(na)
        output.append(array)
    output = np.array(output)
    return output

def expand_dict(list_dict):
    outdict = {}
    for d in list_dict:
        subkeys = d.keys()
        for subkey in subkeys:
            if not subkey in outdict: outdict[subkey] = []
            outdict[subkey].append(d[subkey])
    return outdict

def isinstance_netarray(x):
    if isinstance(x, strobj) and x.find('adler32')>0:
        return True
    else:
        return False

def load_slow(files, Tstart=None, Tend=None, slowdir='/', sources=[], iterative=False):
    """Loading slowdaq data files and filling the data into python dictionary.

    files: list of slowdaq log files
    Tstart: starting time in unix time (seconds)
    Tend: starting time in unix time (seconds)
    slowdir: path for slowdaq data directory
    sources: list of source you need to get
    """

    def fill_dict(msgjson, slowdict):
        if not 'time' in msgjson:
            print('No time, skipping:', msgjson)
            return 
        if not 'source' in msgjson:
            print('No source, skipping:', msgjson)
            return 
        t = msgjson['time']
        if Tstart is not None and t<Tstart: return 
        if Tend is not None and t>Tend:     return 1 # return 1 to break the loop 
        source = msgjson['source'][0]
        if len(sources) and not source in sources: return 
        if not source in slowdict: slowdict[source] = {}
        
        for key in msgjson.keys():
            if not key in slowdict[source]: slowdict[source][key] = []
            slowdict[source][key].append(msgjson[key])

    t0 = time.time()
    slowdata = {}
    for fname in files:
        print('Loading', os.path.join(slowdir, fname))
        sys.stdout.flush()

        if iterative: # iterative file reading, original implementation, but very slow
            with File(os.path.join(slowdir, fname), 'r') as f:
                for msg in f:
                    msgjson = json.loads(msg)
                    if fill_dict(msgjson, slowdata)==1: break
        else:
            with File(os.path.join(slowdir, fname), 'r') as f:
                fdata = f.load()
            for msg in fdata:
                msgjson = json.loads(msg)
                if fill_dict(msgjson, slowdata)==1: break
            fdata = []

    t1 = time.time()
    # Trying to expand nested dictionaries...
    msg = 'Existing sources: '
    for src in slowdata:
        sys.stdout.flush()
        msg += src + ' '
        for key in slowdata[src].keys():
            #print '\t', key
            if isinstance_netarray(slowdata[src][key][0]):
                slowdata[src][key] = expand_netarray(slowdata[src][key])
                continue
            if isinstance(slowdata[src][key][0], dict):
                slowdata[src][key] = expand_dict(slowdata[src][key])
                for subkey in slowdata[src][key]:
                    #print '\t\t', subkey
                    if isinstance_netarray(slowdata[src][key][subkey][0]):
                        slowdata[src][key][subkey] = expand_netarray(slowdata[src][key][subkey])
                        continue
                    if isinstance(slowdata[src][key][subkey][0], dict):
                        slowdata[src][key][subkey] = expand_dict(slowdata[src][key][subkey])
                        for subsubkey in slowdata[src][key][subkey]:
                            #print '\t\t\t', subsubkey
                            if isinstance_netarray(slowdata[src][key][subkey][subsubkey][0]):
                                slowdata[src][key][subkey][subsubkey] = expand_netarray(slowdata[src][key][subkey][subsubkey])
                                continue
                            if isinstance(slowdata[src][key][subkey][subsubkey][0], dict):
                                slowdata[src][key][subkey][subsubkey] = expand_dict(slowdata[src][key][subkey][subsubkey])
                                #for subsubsubkey in slowdata[src][key][subkey][subsubkey]:
                                    #print '\t\t\t\t', subsubsubkey
    t2 = time.time()
    print(msg, '\nLoaging slowdata done. It took %f seconds'%(t2-t0))
    return slowdata

if __name__=='__main__':
    """
    import cPickle as pickle
    import sqlite3
    import socket
    root_dir = './'
    hostname = socket.gethostname()
    if hostname.find('cc.kek.jp')>=0:
        root_dir = '/group/cmb/polarbear/data/pb2a_integration/'
        print(hostname, root_dir)
    elif hostname.find('tostada1')>=0:
        root_dir = '/data/pb2/KEKdata//'
        print(hostname, root_dir)

    runOffset = '13900000'

    slowdir = os.path.join(root_dir, 'slowdaq', runOffset)

    if len(sys.argv) != 2:
        print('This requires a runID as an argument.')
        sys.exit()
    runID = sys.argv[1]

    # Fetching file information from sqlite database
    dbname = os.path.join(root_dir, 'filedb', 'db_slowdaq_%s_merged.db'%runOffset)
    if not os.path.isfile:
        print(dbname, 'not existing.')
        sys.exit()
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    cur.execute("SELECT date, fileName FROM slowdaq_merged where runID==? order by startTime", (runID,))
    output = cur.fetchall()
    cur.close()
    """
    files = sys.argv[1:]

    slowdata = load_slow(files)
