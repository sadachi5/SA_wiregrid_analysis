
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



#'''
import numpy as np;
from utils import plottmp;

alpha = 0.; # deg
beta  =  20.; # deg

x = np.arange(200.)/200. * 180.;
rad = x/180.*np.pi;
y = (np.arctan(np.tan(rad)*np.cos(np.pi/180. * alpha)/np.cos(np.pi/180. * beta)) - np.arctan(np.tan(rad)))*180./np.pi * 2.;

plottmp(x,y,xlabel='angle [deg.]',ylabel='diff [deg.] by ({:.0f},{:.0f}) deg. tilts'.format(alpha,beta));
