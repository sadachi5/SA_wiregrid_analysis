
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
#y = (np.arctan(np.tan(rad)*np.cos(np.pi/180. * alpha)/np.cos(np.pi/180. * beta)) - np.arctan(np.tan(rad)))*180./np.pi * 2.;
y = np.sin(rad)**2.;

plottmp(rad,y,xlabel='angle [deg.]',ylabel='diff [deg.] by ({:.0f},{:.0f}) deg. tilts'.format(alpha,beta));



#y = (np.arctan(np.tan((rad-np.pi)/2.)) + np.pi/2.)*2.;
#y = np.arctan2(np.sin(rad),np.cos(rad));
#y = np.arctan2(-np.sin(rad),-np.cos(rad))+np.pi;
#plottmp(rad,y,xlabel='angle [rad.]',ylabel='acos(cos(angle)) [rad.]');
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






# sqlite merge test
'''
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



# open pandas
'''
import pandas as pd;
import numpy as np;
from utils import colors, getPandasPickle, plottmphist;
database = 'output_ver3/db/all_pandas.pkl';
df = getPandasPickle(database);
print(df.keys());
print('plottmphist(df[\'r\'],i=0,show=True)');
"""
plottmphist(df['r'],i='',outname='all_r',xlabel=r'$r$',nbins=500,xrange=[0,5000],log=True,show=False,drawflow=True);
plottmphist(df['r'],i='',outname='all_r_zoom',xlabel=r'$r$',nbins=50,xrange=[0,50],log=True,show=False,drawflow=True);
plottmphist(df['chisqr'],i='',outname='all_chisqr',xlabel=r'$\chi ^2$',nbins=500,xrange=[0,5000],log=True,show=False,drawflow=True);
plottmphist(df['chisqr'],i='',outname='all_chisqr_zoom',xlabel=r'$\chi ^2$',nbins=50,xrange=[0,50],log=True,show=False,drawflow=True);
plottmphist(df['r'],y=df['chisqr'],i='',outname='all_r-chisqr',xlabel=r'$r$',ylabel=r'$\chi ^2$',nbins=50,xrange=[0,500],yrange=[0,500],log=True,show=False);
plottmphist(df['r'],y=df['chisqr'],i='',outname='all_r-chisqr_zoom',xlabel=r'$r$',ylabel=r'$\chi ^2$',nbins=50,xrange=[0,50],yrange=[0,50],log=True,show=False);
"""

basecut0 = 'theta_det_err<1.*{:e}/180.'.format(np.pi);
"""
df_sel0 = df.query(basecut0);
print('# of "{}" = {}/{}'.format(basecut0, len(df_sel0), len(df)));
plottmphist(df_sel0['r'],i='',outname='thetaless1_r',xlabel=r'$r$',nbins=500,xrange=[0,500],log=True,show=False,drawflow=True);
plottmphist(df_sel0['chisqr'],i='',outname='thetaless1_chisqr',xlabel=r'$\chi ^2$',nbins=500,xrange=[0,500],log=True,show=False,drawflow=True);
"""

basecut = 'r+chisqr>20. & '+basecut0;
"""
df_sel = df.query(basecut);
print('# of "{}" = {}/{}'.format(basecut, len(df_sel), len(df)));
plottmphist(df_sel['r'],i=6,nbins=500,xrange=[0,500],log=True,show=False,drawflow=True);
plottmphist(df_sel['chisqr'],i=7,nbins=500,xrange=[0,500],log=True,show=False,drawflow=True);
plottmphist(df_sel['r'],y=df_sel['chisqr'],i=8,nbins=50,xrange=[0,50],yrange=[0,50],log=True,show=False);
"""

sel2 = 'chisqr>5000';
"""
df_sel2 = df.query(sel2);
print('# of "{}" = {}/{}'.format(sel2, len(df_sel2), len(df)));
plottmphist(df_sel2['r'],y=df_sel2['chisqr'],i=9,nbins=50,log=True,show=False);
"""

sel3 = 'r>5000';
"""
df_sel3 = df.query(sel3);
print('# of "{}" = {}/{}'.format(sel3, len(df_sel3), len(df)));
plottmphist(df_sel3['r'],y=df_sel3['chisqr'],i=10,nbins=50,log=True,show=False);
"""

sel4 = 'chisqr<2 & chisqr>0. & '+basecut;
"""
df_sel4 = df.query(sel4);
print('# of "{}" = {}/{}'.format(sel4, len(df_sel4), len(df)));
pd.set_option('display.max_rows', 100)
print(df_sel4[['readout_name','theta_det_err']]);
pd.set_option('display.max_rows', 5)
"""

datas = [];
cuts = [0,0.5,1.0,1.5,2.,10.];
cuts_strs = [];
for i in range(len(cuts)) :
    if i<len(cuts)-1 :
        selection = 'theta_det_err*180./{pi}>={cut1} & theta_det_err*180./{pi}<{cut2}'.format(pi=np.pi,cut1=cuts[i],cut2=cuts[i+1]);
        cut_str   = cuts[i]+r' deg <= $\Delta \theta_{\mathrm{det}}$ < '+cuts[i+1]+' deg';
    else :
        selection = 'theta_det_err*180./{pi}>={cut}'.format(pi=np.pi,cut=cuts[i]);
        cut_str   = cuts[i]+r' deg <= $\Delta \theta_{\mathrm{det}}$';
        pass;
    datas.append(df.query(selection)['r']);
    cuts_strs.append(cut_str);
    pass;
plottmphist(df['theta_det_err']*180./np.pi,i='',outname='all_theta_det_err',xlabel=r'$\Delta \theta_{det} [deg.]$',nbins=360,xrange=[0,90.],log=True,show=False,drawflow=True);
#plottmphist(df['theta_det_err']*180./np.pi,y=df['chisqr']+df['r'],i=14,nbins=180,xrange=[0,90],yrange=[0,3600],log=True,show=False);
#plottmphist(df['theta_det_err']*180./np.pi,y=df['chisqr']+df['r'],i=15,nbins=100,log=True,show=False);
plottmphist(datas,i='',outname='r_wt_different_theta_det_err',xlabel=r'$r$',ylabel=r'Counts of bolometers',nbins=100,xrange=[0,1000.],log=True,show=False,drawflow=True,stacked=True,nHist=len(datas),label=cuts_strs);
#'''


# Check diff of theta between circle and ellipse
'''
import numpy as np;
from utils import rad_to_deg, rad_to_deg_pitopi, deg_to_rad;
theta = np.arange(0,2.*np.pi,deg_to_rad(22.5/2.));
deg   = rad_to_deg(theta);
r = 1.;
x = np.cos(theta);
y = np.sin(theta);
R = r*1.01;
X = x;
Y = R*np.sin(theta);
Theta = np.arctan2(Y,X);

diff_deg = rad_to_deg_pitopi(Theta - theta);

from matplotlib import pyplot as plt;
plt.plot(deg, diff_deg, marker='o');
plt.xlabel(r'Circle angle $\theta$ [deg.]');
plt.ylabel('Angle diff.\nbetween circle(r={}) and ellipse(R={}) [deg.]'.format(r,R));
plt.grid(True);
plt.plot([-1000,1000.],[0,0],c='k',linewidth=1);
plt.xlim(0,360);
plt.savefig('aho.png');
#'''



# Check HWP rotation fluctuation
#'''
import numpy as np;
from utils import plottmp, rad_to_deg, rad_to_deg_pitopi, deg_to_rad, between;
from matplotlib import pyplot as plt;
from Demod import Demod;
import cmath;
time_length = 20.;
tmin_demod = 5.;
tmax_demod = 15.;
sampling_rate = 100.;
time = np.linspace(0.,time_length,(int)(time_length*sampling_rate)+1);
speed = 2.;
#tau   = 0.;
tau   = 0.01;
HWP_theta = (time * speed * 2.*np.pi)%(2.*np.pi);
HWP_theta_det = ((time-tau) * speed * 2.*np.pi)%(2.*np.pi);
fluc_amp = deg_to_rad(0.5);
theta_dfluc = fluc_amp * np.sin(2.*HWP_theta_det+deg_to_rad(45.));
HWP_theta_wtfluc = HWP_theta_det + theta_dfluc;
plottmp(time,rad_to_deg(HWP_theta),xlabel='Time [sec.]',ylabel='HWP angle [deg.]',i=0,outname='HWP_angle',xlim=None, xtime=False);
plottmp(time,rad_to_deg(HWP_theta_det),xlabel='Time [sec.]',ylabel='HWP angle after delay [deg.]',i=0,outname='HWP_angle_det',xlim=None, xtime=False);
plottmp(time,rad_to_deg(theta_dfluc),xlabel='Time [sec.]',ylabel='HWP angle fluc. [deg.]',i=0,outname='HWP_angle_dfulc',xlim=None, xtime=False);
plottmp(rad_to_deg(HWP_theta),rad_to_deg_pitopi(HWP_theta-HWP_theta_wtfluc),xlabel='HWP angle [deg.]',ylabel='HWP angle - HWP angle wt fluc. [deg.]',i=0,outname='HWP_angle_vs_dfulc',xlim=None, xtime=False);


demod = Demod(time, HWP_theta);
demod_wtfluc = Demod(time, HWP_theta_wtfluc);

i_figs = 3;
j_figs = 3;
fig, axs = plt.subplots(i_figs,j_figs);
fig.set_size_inches(6*j_figs,6*i_figs);
plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

y_demods = [];
y_demods_wtfluc = [];
phis = [];
phis_wtfluc = [];

wire_angles = np.linspace(0.,180.,8+1)[0:8];
wire_amp = 1.;
for n, wire_angle in enumerate(wire_angles) :
    wire_theta = deg_to_rad(wire_angle);
    tod        = wire_amp*np.sin(4.*HWP_theta-2.*wire_theta);
    tod_wtfluc = wire_amp*np.sin(4.*HWP_theta_wtfluc-2.*wire_theta);

    y_demod = demod.demod(tod,HWP_theta,4.,narrow=True,theta_det=0.);
    y_demod_wtfluc = demod_wtfluc.demod(tod_wtfluc,HWP_theta_wtfluc,4.,narrow=True,theta_det=0.);
    time_demod, y_demod = between(time, y_demod, tmin_demod, tmax_demod);
    y_demod_wtfluc = between(time, y_demod_wtfluc, tmin_demod, tmax_demod)[1];
    y_demods.append(y_demod);
    y_demods_wtfluc.append(y_demod_wtfluc);
    phis.append(cmath.phase(np.mean(y_demod)));
    phis_wtfluc.append(cmath.phase(np.mean(y_demod_wtfluc)));

    i = (int)(n/3);
    j = n%3;
    axs[i][j].plot(time, tod, label='TOD' , c='k', linewidth=0.5, linestyle='-');
    axs[i][j].plot(time, tod_wtfluc, label='TOD wt fluc.' , c='k', linewidth=0.5, linestyle='--');
    axs[i][j].plot(time_demod, y_demod_wtfluc.real, label='real wt fluc.' , c='r', linewidth=1., linestyle='-');
    axs[i][j].plot(time_demod, y_demod_wtfluc.imag, label='imag. wt fluc.', c='r', linewidth=1., linestyle='--');
    axs[i][j].plot(time_demod, y_demod.real, label='real' , c='tab:blue', linewidth=1., linestyle='-');
    axs[i][j].plot(time_demod, y_demod.imag, label='imag.', c='tab:blue', linewidth=1., linestyle='--');
    axs[i][j].grid(True);
    axs[i][j].legend();
    axs[i][j].set_ylim(-1.2*wire_amp, 1.2*wire_amp);

    axs[2][2].scatter(np.mean(y_demod.real), np.mean(y_demod.imag), c='tab:blue', marker='o', s=10.);
    axs[2][2].scatter(np.mean(y_demod_wtfluc.real), np.mean(y_demod_wtfluc.imag), c='r', marker='o', s=10.);
    axs[2][2].grid(True);
    axs[2][2].legend();
    axs[2][2].set_xlim(-1.2*wire_amp, 1.2*wire_amp);
    axs[2][2].set_ylim(-1.2*wire_amp, 1.2*wire_amp);
    pass;
fig.savefig('tmp/HWP_sim_demod.png');
plt.close();

phis = np.array(phis);
phis_wtfluc = np.array(phis_wtfluc);
plottmp(wire_angles, rad_to_deg_pitopi(phis_wtfluc-phis), outname='diff_phi', ylabel='Angle diff. [deg.]');
#'''

