#!/usr/bin/env python
import os;
import sqlite3;
import numpy as np;

class DBreaderStimulator:

    dbfilename = '';
    sq         = None;
    cursor     = None;
    verbose    = 2;

    def __init__(self, dbfilename='', tablename='pb2a_stimulator', verbose=2):
      self.dbfilename = dbfilename;
      self.tablename  = tablename;
      print('Open DB file : {}'.format(self.dbfilename));
      print('  Table name : {}'.format(self.tablename ));
      self.sq         = sqlite3.connect(self.dbfilename);
      self.cursor     = self.sq.cursor();
      self.verbose    = verbose;
      self.cursor.execute('SELECT * FROM sqlite_master where type="table"');
      print('Input DB table structure : {}'.format(self.cursor.fetchall()[0]));
      self.cursor.execute('SELECT * FROM {}'.format(self.tablename));
      alldb = self.cursor.fetchall();
      print('Input DB first element : {}'.format(alldb[0]));
      pass;

    def getchannel(self, runID=None, channelname='') :
        self.cursor.execute('SELECT * FROM {}'.format(self.tablename));
        alldb = self.cursor.fetchall();
        dbs = [];
        for db in alldb :
            run   = int(db[0]);
            name  = db[3];
            if channelname==name :
                if (runID is None) or (runID==run) :
                    dbs.append(db);
                pass;
            pass;
        if len(dbs)>0 : return dbs;
        print('There is no matched runID / name to {} / {}.'.format(runID, channelname));
        return None;

    def getamp(self, runID=0, channelname='') :
        channel = self.getchannel(runID, channelname);
        if channel is None : return (0.,0.);
        else               : return (float(channel[0][6]), float(channel[0][7]));

    def gettau(self, runID=0, channelname='') :
        channel = self.getchannel(runID, channelname);
        if channel is None : return (0.,0.);
        else               : return (float(channel[0][4]), float(channel[0][5]));

    def getintensity(self, runID=None, channelname='', nearRunID=None, source=None) :
        channels = self.getchannel(runID, channelname);
        if self.verbose>1 : print('get channels: {}'.format(channels));
        if channels is None : 
            return None;
        else                : 
            # selection for source
            if source is not None :
                __channels = [];
                for channel in channels :
                    if channel[4]==source : __channels.append(channel);
                    pass;
                channels = __channels;
                pass;
            # select near run data
            if nearRunID is not None :
                if nearRunID>22300000 : nearRunID -= 22300000;
                while len(channels)>0 : 
                    runIDs = np.array([(int)(channel[0]) for channel in channels]);
                    diffrun= np.abs(runIDs - nearRunID);
                    nearest_index = diffrun.argmin();
                    if channels[nearest_index][6] is not None and channels[nearest_index][7] is not None:
                        channels = [channels[nearest_index]]; # select one channel
                    else :
                        del channels[nearest_index]; # remove the channel
                        pass;
                    pass;
            # check there is None data channels
            for i in range(len(channels)-1, -1, -1) :
                if channels[i][6] is None or channels[i][7] is None: del channels[i];
                pass;
            if self.verbose>1 : print('DBreaderStimulator::getintensity(): Selected channels: {}'.format(channels));
            if len(channels)>0 : return [[float(channel[6]), float(channel[7])] for channel in channels]; # return K_RJ, K_CMB
            else               : return None;


if __name__ == '__main__':
    #'''
    #db = DBreaderStimulator('./data/pb2a_stimulator_run223_20210223.db', tablename='pb2a_stimulator');
    db = DBreaderStimulator('./data/pb2a-20211004/pb2a_stim.db', tablename='pb2a_stimulator');
    print('getchannel():', db.getchannel(22300610, 'PB20.13.13_Comb01Ch01'));
    print('gettau():', db.gettau(22300610, 'PB20.13.13_Comb01Ch01'));
    print('getamp():',db.getamp(22300610, 'PB20.13.13_Comb01Ch01'));
    for i in range(1,10) :
        name = 'PB20.13.13_Comb01Ch0{}'.format(i);
        print('getamp() for {}:'.format(name), db.getamp(22300610, name));
        pass;
    pass;
    #'''

    '''
    db = DBreaderStimulator('./data/pb2a_stim_template_20210607.db', tablename='pb2a_stim_template');
    print(db.getintensity(channelname='PB20.13.13_Comb01Ch01',nearRunID=22300610,source='Jupiter'));
    for a in db.getintensity(channelname='PB20.13.13_Comb01Ch01',source='Jupiter'): print(a);
    print(np.mean(np.array(db.getintensity(channelname='PB20.13.13_Comb01Ch01',source='Jupiter'))[:,0])); # averaged jupiter [mK_RJ/amp]
    #'''

### schema for pb2a_stim_template DB (pb2a_stim_template_20210607.db) ###
# CREATE TABLE pb2a_stim_template (                                     #
# run_id TEXT,run_subid TEXT, hardware_map_readout_name TEXT,           #
# Bolo_name TEXT, source TEXT, number_freq_use TEXT,                    #
# intensity_K_RJ INTEGER, intensity_K_CMB INTEGER, status TEXT);        #
#########################################################################

