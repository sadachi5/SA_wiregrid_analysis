#!/usr/bin/env python
import os;
import sqlite3;

class DBreader:

  def __init__(self, dbfilename='', tablename='boloid', idcolumn='name', verbose=2):
    self.dbfilename = dbfilename;
    self.tablename  = tablename ;
    self.idcolumn   = idcolumn;
    self.verbose    = verbose;
    print('Open DB file : {}'.format(self.dbfilename));
    self.sq         = sqlite3.connect(self.dbfilename);
    self.cursor     = self.sq.cursor();
    self.cursor.execute('SELECT * FROM sqlite_master where type="table"');
    print('Input DB table structure : {}'.format(self.cursor.fetchall()[0]));
    self.verbose    = verbose;
    self.cursor.execute('SELECT {id} FROM {table} ORDER BY {id}'.format(id=self.idcolumn, table=self.tablename));
    self.allid = [ __id[0] for __id in self.cursor.fetchall()];
    if verbose>1: print('Id : {}'.format(self.allid));
    self.cursor.execute('SELECT * FROM {table} ORDER BY {id}'.format(id=self.idcolumn, table=self.tablename));
    self.alldata = self.cursor.fetchall();
    pass;

  def getdata(self, idname='') :
    for i, __id in enumerate(self.allid) :
      if self.verbose>2 :  print(__id, idname);
      if __id==idname :
        return self.alldata[i];
      pass;
    print('There is no matched name to {}.'.format(idname));
    return None;

  def getcolumn(self, column='theta_det', idname='PB20.13.13_Comb01Ch01') :
    if isinstance(idname, str) : idname = "'{}'".format(idname);
    self.cursor.execute('SELECT {} FROM {} WHERE {}={}'.format(column, self.tablename, self.idcolumn, idname));
    selectdata = self.cursor.fetchall();
    if len(selectdata)==0 : 
        print('There is no matched name to {} in {}.'.format(idname, self.idcolumn));
        return None;
    return selectdata[0][0];

  def printChannelName(self):
    for __id in self.allid :
        print(__id);
        pass;
    return 0;

  def printAll(self):
    print(self.alldata);
    return 0;
    


if __name__ == '__main__':
  #db = DBreader('./data/boloid_pb2a_20210412.db', 'boloid', 'name');
  db = DBreader('./output_ver2/db/all_mod.db', 'wiregrid', 'readout_name');
  #print(db.getdata('PB20.13.13_Comb01Ch01'));
  for i in range(1,10) :
    print(db.getdata('PB20.13.13_Comb01Ch0{}'.format(i)));
    theta = db.getcolumn('theta_det', 'PB20.13.13_Comb01Ch0{}'.format(i));
    print('theta (type:{}) = {}'.format(type(theta), theta));
    pass;
  db.printChannelName();
  db.printAll();
  pass;
