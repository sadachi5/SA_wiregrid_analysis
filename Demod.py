import numpy as np;
import library.simons_array_offline_software.simons_array_python.sa_common_numerical as sa_common_numerical;
from utils import plottmp, mjd_to_second, second_to_mjd;
from scipy.signal import firwin;
from scipy.fft import fft, fftfreq;

import Out;

class Demod :

    # time  : array of bolometer times in TOD [sec]
    # angle : array of HWP angles [rad]
    def __init__( self, time, angle, 
            numtaps   = 1023,
            bandwidth = 1.9,        band_modes        = [0,2,4]            ,
            bandwidth_narrow = 0.1, band_modes_narrow = [0,1,2,3,4,5,6,7,8],
            out=None, verbosity=0 ) :
        # Initialize Out
        self.m_verbosity = verbosity;
        if out==None : self.m_out = Out.Out(verbosity=verbosity);
        else         : self.m_out = out;

        # Initialize demod configuration
        # numtaps : # of FIR taps (# of filtering coefficients: N)
        # tod_size : number of sampling points in the considered TOD data
        # dt : median of time interval of the sampling in TOD
        # nyq : nyquest frequency
        # speed    : HWP rotation speed [Hz]
        self.m_numtaps  = numtaps; # should be odd
        self.m_tod_size = len(time);
        self.m_dt_ave   = self.get_dt(time);
        self.m_nyq      = (1./self.m_dt_ave) / 2. ;
        self.m_speed    = self.get_speed(time, angle, mask=None); # [Hz]
        # freq array only for plotting
        self.m_freq_ntaps = fftfreq(self.m_numtaps , self.m_dt_ave);
        self.m_freq_nlpf  = fftfreq(sa_common_numerical.getNFFT(self.m_tod_size + 3*self.m_numtaps-1), self.m_dt_ave);

        # Prepare band-pass and low-pass filters
        self.m_band         = bandwidth ; # nominal band width(+-) [Hz/speed]
        self.m_demod_modes  = band_modes; # nominal (for demod) band centers [Hz/speed] (mode*speed has [Hz] unit.)
        self.m_band_narrow  = bandwidth_narrow ; # narrow band width(+-) [Hz/speed]
        self.m_narrow_modes = band_modes_narrow; # band centers for narrow band filters [Hz/speed] (mode*speed has [Hz] unit.)
        self.m_all_modes = self.m_demod_modes + [ m for m in self.m_narrow_modes if not m in self.m_demod_modes ]; # get all modes for band pass filter

        # Print variables
        self.m_out.OUTVar(self.m_tod_size, 'm_tod_size', 0);
        self.m_out.OUTVar(self.m_dt_ave  , 'm_dt_ave  ', 0);
        self.m_out.OUTVar(self.m_nyq     , 'm_nyq     ', 0);
        self.m_out.OUTVar(self.m_speed   , 'm_speed   ', 0);
        self.m_out.OUTVar(self.m_band        , 'm_band        ', 0);
        self.m_out.OUTVar(self.m_demod_modes , 'm_demod_modes ', 0);
        self.m_out.OUTVar(self.m_band_narrow , 'm_band_narrow ', 0);
        self.m_out.OUTVar(self.m_narrow_modes, 'm_narrow_modes', 0);
        self.m_out.OUTVar(self.m_all_modes   , 'm_all_modes   ', 0);

        # Define band-pass/low-pass filters
        self.m_fbpf         = {}  ; # band pass fitlers for narrow band filters
        self.m_flpf         = None; # low  pass fitlers for narrow band filters 
        self.m_fbpf_narrow  = {}  ; # band pass filters for nominal demod
        self.m_flpf_narrow  = None; # low  pass filters for nominal demod
        self.defineFilters();

        pass;

    # time [sec]
    def get_dt(self, time) : return np.median(np.diff(time));
    
    # time [sec], angle [rad] --> rotation speed [Hz]
    def get_speed(self, time, angle, mask=None):
        if self.m_out.m_verbosity>1 : plottmp(time[:-1], np.diff(angle), 'time', 'angle diff.', outname='AngleDiff', i=0); 
        dangle = np.diff(angle) / np.pi / 2 ; # radian to rotation number
        self.m_out.OUT('dangle befor unwrap = {}'.format(dangle),1);
        dangle[dangle < -0.5] += 1; # unwrap
        self.m_out.OUT('dangle after unwrap = {}'.format(dangle),1);
        dt_array = np.diff(time); # array of dt
        self.m_out.OUTVar(dt_array,'dt_array',1);
        if mask is None :
            dm = np.ones(len(dangle), dtype=bool);
        else:
            dm = mask[1:] * mask[:-1]; # If next or previous data is bad, the current data should be masked.
            pass;
        s = dangle / dt_array; # Hz
        if self.m_out.m_verbosity>1 : plottmp(time[:-1], s, 'time', 'Rotation Speed', outname='AngleSpeed', i=0); 
        if self.m_out.m_verbosity>1 : plottmp(time[:-1], dangle, 'time', 'angle diff.', outname='AngleDiff', i=1); 
        if self.m_out.m_verbosity>1 : plottmp(time[:-1], dt_array, 'time', 'time diff.', outname='TimeDiff', i=0); 
        self.m_out.OUTVar(s,'s',1);
        self.m_out.OUTVar(dm,'dm',1);
        self.m_out.OUTVar(dm[dm==True],'dm==True before mask',1);
        s0 = np.median(s[dm]); # Median of speed data who has True in dm.
        self.m_out.OUTVar(s0,'s0',1);
        dm *= (np.abs(s - s0) < 0.1); # Mask data if the diff from the median is >= 0.1 Hz.
        self.m_out.OUTVar(dm,'dm after mask',1);
        self.m_out.OUTVar(dm[dm==True],'dm==True after mask',1);
        self.m_out.OUTVar(dangle  ,'dangle before mask',1);
        self.m_out.OUTVar(dt_array,'dt_array before mask',1);
        self.m_out.OUTVar(dangle[dm]  ,'dangle after mask',1);
        self.m_out.OUTVar(dt_array[dm],'dt_array after mask',1);
        self.m_out.OUTVar(dangle.sum()  ,'sum of dangle before mask',1);
        self.m_out.OUTVar(dt_array.sum(),'sum of dt_array before mask',1);
        self.m_out.OUTVar((dangle[dm]).sum()  ,'sum of dangle after mask',1);
        self.m_out.OUTVar((dt_array[dm]).sum(),'sum of dt_array after mask',1);
        return dangle[dm].sum() / dt_array[dm].sum(); # sum of not-masked angle data / sum of not-masked time period data
    
    def defineFilters(self) :
        for isnarrow, modes in enumerate([self.m_demod_modes, self.m_narrow_modes]):
            # isnarrow = 0 : demod_modes 
            # isnarrow = 1 : narrow_modes 
            fbpf = {};
            for mode in modes:
                if mode == 0:
                    fbpf[0] = None;
                    continue;
                if mode in fbpf:
                    continue;
                b = self.m_band_narrow if isnarrow else self.m_band; # band width
                pass_zero = (self.m_speed * mode - b) <= 0.; # check if the band lower edge is < 0.
                # Get bandpass filter in time-space
                self.m_out.OUT('band center = {} / band width = {}'.format(mode, b), 1);
                bpf = sa_common_numerical.firwinc(self.m_numtaps,
                              [self.m_speed * (mode - b),  # lower band edge [Hz]
                               self.m_speed * (mode + b)], # upper band edge [Hz]
                              nyq=self.m_nyq, pass_zero=pass_zero)
                if self.m_out.m_verbosity>1 : plottmp(self.m_freq_ntaps, np.abs(bpf), 'index', 'bpf', outname='bpfnarrow' if isnarrow else 'bpf', i=mode, xlim=[0,15]);
                # Get bandpass filter in freq-space
                fbpf[mode] = sa_common_numerical.fftfilter(bpf, self.m_tod_size)
                if self.m_out.m_verbosity>1 : plottmp(self.m_freq_nlpf, np.abs(fbpf[mode]), 'index', 'fbpf[mode]', outname='fbpfnarrow' if isnarrow else 'fbpf', i=mode, xlim=[0,15]);
                pass;
            b    = self.m_band_narrow if isnarrow else self.m_band; # band width
            lpf  = firwin(self.m_numtaps, self.m_speed * b, nyq=self.m_nyq, pass_zero=True); # firwin : scipy function / speed * b : band upper threshould = band width / pass_zero : reverse the band
            flpf = sa_common_numerical.fftfilter(lpf, self.m_tod_size);
            if isnarrow: # narrow_modes
                self.m_fbpf_narrow = fbpf; # band pass fitlers for narrow band filters
                self.m_flpf_narrow = flpf; # low  pass fitlers for narrow band filters
            else: # demod_modes
                self.m_fbpf = fbpf; # band pass filters for nominal demod
                self.m_flpf = flpf; # low  pass filters for nominal demod
                pass;
            pass;
        return 0;
    
    # y     : array of detector ADC outputs
    # angle : array of HWP angles [rad]
    # mode  : center of filtering band center [Hz]
    # narrow: use narrow band width or not (default: wide band width)
    def demod(self, y, angle, mode, narrow=False, doBpf=True, doLpf=True) :
        self.m_out.OUT('demod mode={} / isnarrow={}'.format(mode, narrow));
        # Select low pass fitler
        flpf = self.m_flpf_narrow if narrow else self.m_flpf;
        # No bandpass
        if mode == 0 :
            return sa_common_numerical.convolvefilter(y, self.m_numtaps, flpf).real;
            pass;
        # Select band pass fitler
        fbpf = self.m_fbpf_narrow[mode] if narrow else self.m_fbpf[mode]
        # Get exp term
        expangle = {};
        for modetmp in self.m_all_modes:
            expangle[modetmp] = np.exp(-1j * angle * modetmp);
            pass;
        e = expangle[mode];
        if self.m_out.m_verbosity>1 : plottmp(range(len(e)), e.real, 'Index', 'exp real [{}] {}'.format(mode, 'narrow' if narrow else 'demod'), outname='Exp{}{}'.format(mode, 'narrow' if narrow else 'demod'), i=0); 
        
        # Get demod result
        # If you use fbpf/flpf instead of bpf/lpf, 
        # we need just filter size for bpf/lpf.
        self.m_out.OUTVar(fbpf,'fbpf',1);
        self.m_out.OUTVar(flpf,'flpf',1);
        u = sa_common_numerical.demod(y, e, 
                bpf =self.m_numtaps if doBpf else None, lpf =self.m_numtaps if doLpf else None, 
                fbpf=fbpf           if doBpf else None, flpf=flpf           if doLpf else None); 
        return u;

    def __del__(self) :
        pass;

    pass; # End of class Demod()


if __name__ == '__main__' :
    from utils import printVar;
    size  = 3000;
    time  = np.arange(size)/100.;
    time_mjd  = second_to_mjd(time);
    angle = (2.*np.pi*time * 2.) % (2.*np.pi); # 2Hz
    #y     = 100.*np.sin(angle * 4.) + 300. + np.arange(size)*1.0; # 4f (8Hz) + baseline fluctuation
    y     = 100.*np.sin(2.*np.pi*time * 8.) + 300. + np.arange(size)*1.0; # 4Hz + baseline fluctuation
    printVar(y);
    plottmp(time, y, 'time [sec]', 'y', outname='y', xlim=[0,5]);
    plottmp(time, angle, 'angle' , 'angle', outname='angle', xlim=[0,5]);

    # plot fft
    def plotfft(time, y, outname='test_fft', i=0, xlim=None, label='y') :
        if len(y)%2 == 0 : 
            y_tmp = y[0:-1];
            time_tmp = time[0:-1];
            pass;
        dt = np.median(np.diff(time_tmp));
        size = len(y_tmp);
        printVar(size);
        freq = fftfreq(size, dt);
        window = np.hamming(size);
        y_fft = fft(y_tmp*window);
        plottmp(freq, np.abs(y_fft), 'Freq. [Hz]', label ,i=i, outname=outname, xlim=xlim);
        return 0;


    plotfft(time, y    , 'fft_before_demod', i=0, xlim=[0,15], label='y before demod');

    demod = Demod(time, angle, verbosity=2, band_modes=[0,2,4], band_modes_narrow=[0,2,4]);
    y_demod0 = demod.demod(y,angle,0.,narrow=True); # only low pass filter
    y_demod2 = demod.demod(y,angle,2.,narrow=True);
    y_demod4 = demod.demod(y,angle,4.,narrow=True);
    printVar(y_demod0);
    printVar(y_demod2);
    printVar(y_demod4);
    printVar(y_demod0[(int)(len(y)/2)]);
    printVar(y_demod2[(int)(len(y)/2)]);
    printVar(y_demod4[(int)(len(y)/2)]);
    plottmp(time, y_demod0.real, 'time [sec]', 'y_demod0.real', outname='demody0.real');
    plottmp(time, y_demod2.real, 'time [sec]', 'y_demod2.real', outname='demody2.real');
    plottmp(time, y_demod4.real, 'time [sec]', 'y_demod4.real', outname='demody4.real');
    plottmp(time, y_demod0.imag, 'time [sec]', 'y_demod0.imag', outname='demody0.imag');
    plottmp(time, y_demod2.imag, 'time [sec]', 'y_demod2.imag', outname='demody2.imag');
    plottmp(time, y_demod4.imag, 'time [sec]', 'y_demod4.imag', outname='demody4.imag');
    plottmp(time, np.abs(y_demod0), 'time [sec]', '|y_demod0|', outname='demody0.abs');
    plottmp(time, np.abs(y_demod2), 'time [sec]', '|y_demod2|', outname='demody2.abs');
    plottmp(time, np.abs(y_demod4), 'time [sec]', '|y_demod4|', outname='demody4.abs');

    plotfft(time, y_demod2, 'fft_after_demod2', i=0, xlim=[0,15], label='y after demod2');
    plotfft(time, y_demod4, 'fft_after_demod4', i=0, xlim=[0,15], label='y after demod4');

    pass;
