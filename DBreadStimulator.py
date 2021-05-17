#!/usr/bin/env python
import os;
import sqlite3;

class DBreaderStimulator:

    dbfilename = '';
    sq         = None;
    cursor     = None;
    verbose    = 2;

    def __init__(self, dbfilename='', verbose=2):
      self.dbfilename = dbfilename;
      print('Open DB file : {}'.format(self.dbfilename));
      self.sq         = sqlite3.connect(self.dbfilename);
      self.cursor     = self.sq.cursor();
      self.verbose    = verbose;
      self.cursor.execute('SELECT * FROM sqlite_master where type="table"');
      print('Input DB table structure : {}'.format(self.cursor.fetchall()[0]));
      self.cursor.execute('SELECT * FROM pb2a_stimulator');
      alldb = self.cursor.fetchall();
      print('Input DB first element : {}'.format(alldb[0]));
      pass;

    def getchannel(self, runID=0, channelname='') :
      self.cursor.execute('SELECT * FROM pb2a_stimulator');
      alldb = self.cursor.fetchall();
      for db in alldb :
          run   = int(db[0]);
          name  = db[3];
          if runID==run and channelname==name :
            return db;
          pass;
      print('There is no matched runID / name to {} / {}.'.format(runID, channelname));
      return None;

    def getamp(self, runID=0, channelname='') :
        channel = self.getchannel(runID, channelname);
        if channel is None : return 0;
        else               : return (float(channel[6]), float(channel[7]));

    def gettau(self, runID=0, channelname='') :
        channel = self.getchannel(runID, channelname);
        if channel is None : return 0;
        else               : return (float(channel[4]), float(channel[5]));



if __name__ == '__main__':
    db = DBreaderStimulator('./data/pb2a_stimulator_run223_20210223.db');
    print(db.getchannel(22300610, 'PB20.13.13_Comb01Ch01'));
    print(db.gettau(22300610, 'PB20.13.13_Comb01Ch01'));
    print(db.getamp(22300610, 'PB20.13.13_Comb01Ch01'));
    for i in range(1,10) :
        print(db.getamp(22300610, 'PB20.13.13_Comb01Ch0{}'.format(i)));
        pass;
    pass;
