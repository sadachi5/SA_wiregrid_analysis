#!/usr/bin/env python
import os;
import sqlite3;

class DBreader:

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
    self.cursor.execute('SELECT * FROM boloid');
    print('Input hardware map : {}'.format(self.cursor.fetchall()[0][10]));
    pass;

  def getchannel(self, channelname='') :
    self.cursor.execute('SELECT * FROM boloid');
    allchannels = self.cursor.fetchall();
    for channel in allchannels :
      name = channel[0];
      if channelname==name :
        return channel;
      pass;
    print('There is no matched name to {}.'.format(channelname));
    return None;

if __name__ == '__main__':
  db = DBreader('./data/boloid_pb2a_20210412.db');
  #print(db.getchannel('PB20.13.13_Comb01Ch01'));
  for i in range(1,10) :
    print(db.getchannel('PB20.13.13_Comb01Ch0{}'.format(i)));
    pass;
  pass;
