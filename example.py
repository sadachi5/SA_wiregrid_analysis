import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import libg3py3 as libg3
import math
from datetime import datetime

#filename='/group/cmb/polarbear/data/pb2a/g3compressed/20000000/Run20000206';
filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300610';
#boloname='PB20.13.12_Comb25Ch02';
boloname='PB20.13.13_Comb01Ch01';
#boloname=None;

g3c=libg3.G3Compressed(filename)
print( 'L16:', g3c.readout )
print( 'L17:', g3c.boloprop )
print( 'L18:', np.array(g3c.bolonames_all) )
#repr(g3c.bolo)
#repr(g3c.bolotime)
bolo=g3c.loadbolo(boloname)

time=[]
for t in g3c.bolotime:
    time.append(datetime.utcfromtimestamp(t/100000000))
    pass
y=bolo[0].real
print( time )
print( y )
