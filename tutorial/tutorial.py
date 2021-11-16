#!/usr/bin/env python
# coding: utf-8

# # Tutorial to retrieve TOD & HWP angle via sa_pipeline

# ### Import general libraries

# In[96]:


import os
import sys
import numpy as np
topdir = os.environ['PWD']
sys.path.append(os.path.join(os.path.dirname(topdir), '')) # Add top directory
print(topdir)
# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().magic(u'matplotlib inline')
# inline non-interactive mode
# python kernel check
print(sys.executable)
print(sys.version)


# ### Data run ID & time setting

# In[3]:


runID = 22300609
subID = 0
start = "20210205_174900";
end   = "20210205_175900";


# ### Get G3Time of start/end time

# In[4]:


from spt3g import core
start_mjd = None if start==None else core.G3Time(start).mjd;
end_mjd   = None if end  ==None else core.G3Time(end  ).mjd;

print(f'{start} = {start_mjd}')
print(f'{end} = {end_mjd}')


# ###  Import simons_array_offline_software

# In[17]:


sys.path.append(os.path.join(os.path.dirname(topdir), './library/simons_array_offline_software'))
# major pipeline libraries
import simons_array_python.sa_pipeline_inputs as sa_pi;
import simons_array_python.sa_observation as sa_ob;
import simons_array_python.sa_pipeline_filters as sa_pf;
# operator of time clip
import simons_array_python.sa_timestream_operators as sa_op;
# HWP angle calculator
import simons_array_python.sa_hwp as sa_hwp;


# ### Get detector list

# In[7]:


num_bolo = 1
#num_bolo = 10
#num_bolo = None # All bolos
all_detectors = sa_op.gen_bolo_list()[:num_bolo] if num_bolo is not None else sa_op.gen_bolo_lis() # Only 1 bolos
print(f'all_detectors {all_detectors} ({len(all_detectors)} bolos)');


# ### Initialize observation (data class)

# In[11]:


print('initialize observation')
observation_tuple = (runID,subID);
ob = sa_ob.Observation(observation_tuple)
ob.detectors = all_detectors
ob.load_metadata()


# ### Initialize pipeline & operator to populate tod_list

# In[13]:


print('dataload operation')
pi = sa_pi.InputLevel0CachedByObsID(all_detectors=all_detectors, n_per_cache=len(all_detectors), 
        load_g3=True, load_gcp=True,
        load_slowdaq=True, load_hwp=True, 
        load_dets=True, ignore_faulty_frame=True, record_frame_time=True);
op_dataload = sa_pf.OperatorDataInitializer(pi)
op_dataload.filter_obs(ob)


# ### Operator to clip the tod by time

# In[15]:


print('time clip operation')
op_timeclip = sa_pf.OperatorClipBeginEnd(begin_mjd=start_mjd, end_mjd=end_mjd)
op_timeclip.filter_obs(ob)


# ### Operator for HWP angle

# In[18]:


print('HWP angle calculation')
op_hwp = sa_hwp.HWPAngleCalculator(encoder_reference_angle=0.)
op_hwp.filter_obs(ob)


# ### Print TOD data

# In[36]:


print(ob)
tod = ob.tod_list[0];
print(tod, len(ob.tod_list))
print('keys',tod.cache.keys());
def printlist(list_data, list_name): print(f'{list_name} (size:{len(list_data)}) = {list_data}');
printlist(tod.read_times(), 'times');
printlist(tod.read('bolo_time'), 'bolo_times');
printlist(tod.read('raw_antenna_time_mjd'), 'raw_antenna_time_mjd');
printlist(tod.read('raw_az_pos'), 'raw_az_pos');
printlist(tod.read('13.13_15.90T-I'), 'TOD (13.13_15.90T-I)');
printlist(tod.read('hwp_angle'), 'HWP angle');


# ### Plot TOD data 

# In[110]:


# Get data arrays
time = tod.read('bolo_time')
antenna_time = tod.read('raw_antenna_time_mjd')
bolo_data = tod.read('13.13_15.90T-I')
az_pos = tod.read('raw_az_pos')
hwp_angle = tod.read('hwp_angle')

# Convert
hwp_angle = hwp_angle * 180./np.pi # rad --> deg
az_pos = az_pos * 180./np.pi # rad --> deg

# Make figure (Nrow)
Nrow = 3; 
fig, axs = plt.subplots(Nrow,1);
fig.set_size_inches(8*1,4*Nrow);
plt.subplots_adjust(wspace=0.5, hspace=0.7, left=0.15, right=0.95,bottom=0.20, top=0.80)
    
# Bolometer TOD
ax = axs[0]
ax.plot(time, bolo_data, label='bolometer')
ax.grid(True)
ax.set_xlabel('Date Time')
ax.set_ylabel('Bolometer ADC')
ax.set_title('Bolometer ADC')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.tick_params(axis='x',labelrotation=30,labelsize=10);

#  HWP angle
ax = axs[1]
ax.plot(time, hwp_angle, label='HWP angle')
ax.grid(True)
ax.set_xlabel('Date Time')
ax.set_ylabel('HWP angle [deg]')
ax.set_title('HWP angle [deg]')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.tick_params(axis='x',labelrotation=30,labelsize=10);

# Azimuth position
ax = axs[2]
ax.plot(antenna_time, az_pos, label='bolometer')
ax.grid(True)
ax.set_xlabel('Date Time')
ax.set_ylabel('Azimuth Position [deg]')
ax.set_title('Azimuth Position [deg]')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.tick_params(axis='x',labelrotation=30,labelsize=10);


# Zoom-in a narrow time range
def sec_to_mjd(sec): return sec * 1./(60.*60.*24) # 1 mjd = 1 day
sec_span = 10; # sec
t_start = time[0]
t_end   = time[0]+sec_to_mjd(sec_span)
for ax in axs: ax.set_xlim(t_start, t_end);


# In[ ]:




































