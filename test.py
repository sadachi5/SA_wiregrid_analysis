
'''
import pickle
fruits= ["apple","orange","melon"]
f = open("aho.pkl","w");
pickle.dump(fruits, f) 
'''

'''
import json
from slowdaq import netstring;

alldict = [];
with netstring.File('/home/cmb/sadachi/data/pb2a/slowdaq/22300000/20210205/slowdaq_20210205_223205_Run22300000_002.log.gz','r') as f :
    for msg in f :
        data = json.loads(msg);
        print(data);
        if not 'time' in data :
            print('No time, skipping:', data);
        else :
            print(data['time']);
            time = data['time'];
            alldict.append(data);
            pass;
        pass;
    pass;
'''


'''
import numpy as np;
from plot import loadbolo;
from utils import plottmp, printVar;
from datetime import datetime;
g3c, bolonames, time, start_mjd, end_mjd = loadbolo(loadSlow=True);

slowData = g3c.slowData;
print(slowData.keys());
print(slowData['SIM900'].keys());
print(slowData['SIM900']['SIM900'].keys());



timedata = slowData['Lakeshore151']['time'];
tempdata = slowData['Lakeshore151']['MODEL370_370A4A_T'];
ntemp = len(tempdata[0]);
s_time= [ datetime.fromtimestamp(time) for time in timedata ];
temps = [[] for k in range(ntemp)];
for i, data in enumerate(tempdata) :
    for k in range(ntemp) :
        temps[k].append(data[k]);
        pass;
    pass;
s_time = np.array(s_time);
temps  = np.array(temps );

printVar(s_time);
printVar(temps);

for k in range(k) :
    plottmp(s_time, temps[k], i=k, xtime=True);
    pass;
'''



'''
import numpy as np;
from utils import plottmp;

alpha = 0.; # deg
beta  =  20.; # deg

x = np.arange(200.)/200. * 180. * 4. - 360.;
rad = x/180.*np.pi;
y = (np.arctan(np.tan(rad)*np.cos(np.pi/180. * alpha)/np.cos(np.pi/180. * beta)) - np.arctan(np.tan(rad)))*180./np.pi * 2.;

plottmp(x,y,xlabel='angle [deg.]',ylabel='diff [deg.] by ({:.0f},{:.0f}) deg. tilts'.format(alpha,beta));



#y = (np.arctan(np.tan((rad-np.pi)/2.)) + np.pi/2.)*2.;
#y = np.arctan2(np.sin(rad),np.cos(rad));
y = np.arctan2(-np.sin(rad),-np.cos(rad))+np.pi;
plottmp(rad,y,xlabel='angle [rad.]',ylabel='acos(cos(angle)) [rad.]');
#'''


'''
import numpy as np;
import sqlite3;
import pickle;

dbname = 'output_ver1/db/PB20.13.13/ver1_PB20.13.13_Comb01Ch01';
conn = sqlite3.connect(dbname+'.db');
cursor = conn.cursor();
cursor.execute('SELECT * FROM wiregrid');
print(cursor.fetchall());
cursor.close();
conn.close();

infile = open(dbname+'.pkl', 'rb');
print( pickle.load(infile) );
#'''


'''
import subprocess
out = subprocess.getoutput('bjobs');
print('out=',out.split('\n'));
print('Njob=',len(out.split('\n'))-1);
#'''







#'''
import sqlite3

tablename = 'wiregrid'
#directory = '/home/cmb/sadachi/analysis_2021/output_ver2';
directory = '/Users/shadachi/Experiment/PB/analysis/analysis_2021/output_ver2';
filenames = [
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch22.db'.format(directory),
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch24.db'.format(directory),
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch25.db'.format(directory),
        ];

con = sqlite3.connect('aho.db')
cur = con.cursor()
sql = 'DROP TABLE if exists {}'.format(tablename);
con.execute(sql);
con.commit();

sql = "ATTACH DATABASE '%s' as filetmp" % (filenames[0])
print(sql);
con.execute(sql)
con.commit()
cur.execute("SELECT name FROM filetmp.sqlite_master WHERE type='table'");
print(cur.fetchall());

sql2 = 'SELECT  * FROM filetmp.{}'.format(tablename)
cur.execute(sql2);
print(cur.fetchall());

sql2 = "PRAGMA TABLE_INFO({})".format(tablename);
print(sql2);
cur.execute(sql2);
info=cur.fetchall();
print(info);

columndefs = [];
notPrimaryColumns = [];
for column in info :
  cid    = column[0];
  name   = column[1];
  ctype  = column[2];
  notnull= column[3];
  default= column[4];
  pk     = column[5];

  columndef = ' {} {}'.format(name, ctype);
  if notnull : columndef += ' NOT NULL';
  if not default is None : columndef += ' DEFAULT {}'.format(default);
  if pk      : columndef += ' PRIMARY KEY';
  else       : notPrimaryColumns.append(name);

  print(columndef);
  columndefs.append(columndef);
  pass;
tabledef = ','.join(columndefs);

print(tabledef);
sql = 'CREATE TABLE {}({})'.format(tablename, tabledef);
con.execute(sql);
con.commit();

sql = 'DETACH DATABASE filetmp';
con.execute(sql);
con.commit();

insertColumns = ','.join(notPrimaryColumns);
for i,filename in enumerate(filenames):
    sql = "ATTACH DATABASE '%s' as file%d" % (filename,i)
    con.execute(sql)
    con.commit();
    sql2 = 'INSERT INTO {table}({columns}) SELECT {columns} FROM file{i}.{table}'.format(table=tablename, columns=insertColumns, i=i);
    print(sql2);
    cur.execute(sql2);
    con.commit();
    pass;
print(cur.fetchall());

for i,filename in enumerate(filenames):
    sql = "DETACH DATABASE file%d" % (i)
    con.execute(sql)
    con.commit();
    pass;

print(con);
print(cur);

cur.close()
con.close()

con = sqlite3.connect('aho.db');
cur = con.cursor();
cur.execute('SELECT * FROM {}'.format(tablename));
print(cur.fetchall());
cur.close();
con.close();
#'''

