import numpy as np;
from library.simons_array_offline_software.simons_array_python.sa_common_numerical import *;
from utils import plottmp, mjd_to_second, second_to_mjd, printVar;
from scipy.signal import firwin;
from scipy.fft import fft, fftfreq;


size  = 3000;
time  = np.arange(size)/100.;
time_mjd  = second_to_mjd(time);
angle = (2.*np.pi*time * 2.) % (2.*np.pi);
y     = 100.*np.sin(2.*np.pi*time * 2.) + np.arange(size)*0.0;

dt = np.median(np.diff(time));
numtaps = size;
freq = fftfreq(numtaps, dt);
y_fft = fft(y);

plottmp(freq, np.abs(y_fft), 'Freq. [Hz]', 'y',i=0, outname='test_fft', xlim=[0,10]);
'''
nt = y.shape[-1];
printVar(nt);
numtaps = len(y);
tmpN = nt + 3 * numtaps - 1;
printVar(tmpN);
nfft = getNFFT(tmpN);
printVar(nfft);
filter_fft = fftfilter(y, nt);
printVar(filter_fft);
'''


