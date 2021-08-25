import os, sys
import argparse;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle
import math
from scipy.optimize import curve_fit
from scipy.fftpack import fft, fftfreq;

import Out;
from OneAngleData import OneAngleData 
from Demod import Demod
from utils import mjd_to_second, theta0to2pi, rad_to_deg, between, rms, saveFig, colors;

from DBreaderStimulator import DBreaderStimulator;
from DBreader import DBreader;


def plotOneTOD(axs, time, y, color, label='', title='') :
    axs.plot(time,y,label=label, marker='o', markersize=0.5,linestyle='',color=color)
    # Plot cosmetic
    axs.set_title(title);
    axs.set_xlabel('Time [sec.]');
    axs.set_ylabel('ADC output');
    axs.grid(True);
    axs.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
    return 0;

def plotOneFFT(axs, time, y, color, label=None, title='', time_lim=None, freq_lim=None, addlegend=False, doNorm=False, normFreq=8.) :
    # time cut on x-axis
    if time_lim!=None :
        time_cut, y_cut = between(time, y, time_lim[0], time_lim[1]);
    else :
        time_cut = time;
        y_cut    = y   ;
        pass;
    freq  = fftfreq(len(time_cut), np.median(np.diff(time_cut)));
    # time cut on y-axis
    y_fft = fft(y_cut, len(y_cut));
    if freq_lim!=None :
        freq_cut, y_fft_cut = between(freq, y_fft, freq_lim[0], freq_lim[1]);
    else :
        freq_cut = freq;
        y_fft_cut= y_fft;
        pass;
    # normalize at normFreq
    if doNorm :
        normFreqI = np.abs(freq_cut - normFreq).argmin(); # get index of the freq. closest to normFreq
        norm_y    = 1./y_fft_cut[normFreqI]; 
        y_fft_cut = y_fft_cut * norm_y ;
        pass;

    if not label is None : label = '{} [mag. ]'.format(label);
    axs.plot(freq_cut,np.abs(y_fft_cut),label=label, linestyle='-',linewidth=0.5,color=color)
    #axs.plot(freq_cut,y_fft_cut.real   ,label='{} [real ]'.format(label), linestyle='--',color=color)
    #axs.plot(freq_cut,y_fft_cut.imag   ,label='{} [imag.]'.format(label), linestyle=':',color=color)
    # Plot cosmetic
    axs.set_title(title);
    axs.set_xlabel('Frequency [Hz]');
    axs.set_ylabel('PSD normalized @ {} Hz'.format(normFreq) if doNorm else 'PSD');
    axs.grid(True);
    axs.set_yscale('log');
    if addlegend : axs.legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.05,0.1),
            framealpha = 1,frameon = False,fontsize = 10,title='',borderaxespad=0.);
    return 0;

def plotDemodTOD(axs, time, y_demod, mode, title='' ) : 
    axs.plot(time,np.abs(y_demod),label='Magnitude', marker='o', markersize=0.5,linestyle='',color=colors[0])
    axs.plot(time,y_demod.real,label='Real part', marker='o', markersize=0.5,linestyle='',color=colors[1])
    axs.plot(time,y_demod.imag,label='Imag. part', marker='o', markersize=0.5,linestyle='',color=colors[2])
    # Plot cosmetic
    axs.set_title(title);
    axs.set_xlabel('Time [sec.]');
    axs.set_ylabel('ADC output');
    axs.grid(True);
    axs.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
    return 0;

def plotDemodFigure(time, y_demods, y_demods_narrow, 
       band_modes, band_modes_narrow, bandwidth, bandwidth_narrow, 
       title,
       outdir, outname, out, ext,
       time_lim=None,
       ) :
    # Demodulation plots for one angleData
    nModes = max(len(band_modes),len(band_modes_narrow));
    figDemod, axsDemod = plt.subplots(nModes,2);
    figDemod.tight_layout(rect=[0,0,1,0.96])
    figDemod.set_size_inches(12,8);
    figDemod.suptitle(title)

    time_cut = [];
    y_demods_cut = {};
    y_demods_narrow_cut = {};
    if time_lim != None :
        for mode, y in y_demods.items() :
            time_cut, y_cut = between(time, y, time_lim[0], time_lim[1]);
            y_demods_cut[mode] = y_cut;
            pass;
        for mode, y in y_demods_narrow.items() :
            time_cut, y_cut = between(time, y, time_lim[0], time_lim[1]);
            y_demods_narrow_cut[mode] = y_cut;
            pass;
    else :
        time_cut = time;
        y_demods_cut = y_demods;
        y_demods_narrow_cut = y_demods_narrow;
        pass;


    for m in range(nModes) :
        # Draw TOD x: time / y: real part of output demodulated by Xf (nominal band filter)
        if len(band_modes)>m : 
            mode = band_modes[m];
            plotDemodTOD(axsDemod[m,0], time_cut, y_demods_cut[mode], mode=mode,
                    title = 'Demodulated by {}f (bandwidth={})'.format(mode, bandwidth), 
                    );
            pass;
        # Draw TOD x: time / y: real part of output demodulated by Xf (narrow band filter)
        if len(band_modes_narrow)>m : 
            mode = band_modes_narrow[m];
            plotDemodTOD(axsDemod[m,1], time_cut, y_demods_narrow_cut[mode], mode=mode,
                    title = 'Demodulated by {}f (bandwidth={})'.format(mode, bandwidth_narrow), 
                    );
            pass;
        pass; # End of loop over modes

    if (not os.path.isdir(outdir)) : 
        out.WARNING('There is no output directory: {}'.format(outdir));
        out.WARNING('--> Make the directory.');
        os.makedirs(outdir);
        pass;
 
    outpath = '{}/{}'.format(outdir, outname);
    out.OUT('Saving plot ({}...)'.format(outpath),0);
    saveFig(figDemod, outpath, ext);
    plt.close(figDemod);

    return 0;
 

def plotAll(angleDataList, db_theta=None, outdir='aho', outname='aho', pickledir='aho', out=None, ext='pdf', verbosity=0) :
    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;

    # set output path
    outpath = outdir+'/'+outname;

    bolonames = angleDataList[0]['data'].m_bolonames;
    out.OUT('bolonames = {}'.format(bolonames),0);

    # initialize Demod class
    demods = [];
    band_modes = [0,2,4];
    band_modes_narrow = [0,1,2,4];
    bandwidth = 1.9;
    bandwidth_narrow = 0.1;
    for j, angledata in enumerate(angleDataList) :
        data  = angledata['data'];
        time  = data.m_bolotime_array[0] * 1.e-8;
        #angle = data.m_whwp_angle/360. * 2. * np.pi; # deg. --> rad.
        angle = data.m_whwp_angle; # [rad.]
        demod = Demod(time,angle,
                band_modes=band_modes,band_modes_narrow=band_modes_narrow,
                bandwidth =bandwidth ,bandwidth_narrow =bandwidth_narrow,
                out=out);
        demods.append(demod);
        pass;

    # loop over bolonames to make plots
    for i, boloname in enumerate(bolonames):

        fig, axs = plt.subplots(4,1);
        fig.tight_layout(rect=[0,0,0.8,0.96])
        plt.subplots_adjust(wspace=0.2, hspace=1.0)
        fig.set_size_inches(14,8);
        fig.suptitle('{}: {}'.format(outname,boloname))

        figFFT, axsFFT = plt.subplots(4,1);
        figFFT.tight_layout(rect=[0,0,0.75,0.96]);
        figFFT.set_size_inches(12,8);
        figFFT.suptitle('{}: {}'.format(outname,boloname))

        figFFTnorm, axsFFTnorm = plt.subplots(4,1);
        figFFTnorm.tight_layout(rect=[0,0,0.75,0.96]);
        figFFTnorm.set_size_inches(12,8);
        figFFTnorm.suptitle('{}: {}'.format(outname,boloname))

        # demod results
        demod_results = [];
        # theta_det for demod
        theta_det =  0.;
        if not db_theta is None :
            out.OUT('db_theta = {}'.format(db_theta),0);
            theta_det = db_theta.getcolumn('theta_det', boloname);
            pass;
    
        for j, angledata in enumerate(angleDataList) :
            data = angledata['data']; # OneAngleData
            wireangle = angledata['angle'];
            wirelabel = 'angleI={}: wire {}deg.'.format(j,wireangle),
            cal     = angledata['cal'];
            cal_err = angledata['cal_err'];


            demod_result = {};

            time = data.m_bolotime_array[i] * 1.e-8;
            time = time - time[0];
            y          = data.m_y_array[i];
            whwp_angle = data.m_whwp_angle;
            angle      = whwp_angle; # [rad.]
            out.OUT('data = {}'.format(data), -1);
            out.OUT('whwp_angle [rad.] (size={}) = {}'.format(len(whwp_angle), whwp_angle), -1);
            out.OUT('theta_det for demod = {}'.format(theta_det), 0);
            out.OUT('y (size={}) = {}'.format(len(y),y), -1);
            out.OUT('time (size={}) = {}'.format(len(time),time), -1);
            out.OUT('diff(time) (size={}) = {}'.format(len(time),np.diff(time)), -1);
         
            # average subtraction
            ave = np.average(y);
            y_subave = y - ave;
            # linear fit of bolo output
            linearfunc=np.poly1d(np.polyfit(time,y,1))
            y_subDC = y - linearfunc(time);
            # Demod
            demod = demods[j];
            numtaps = demod.m_numtaps;
            y_demods        = {};
            y_demods_narrow = {};
            for mode in band_modes :
                out.OUT('demod mode={} (nominal) for angleI={} (wire {}deg.)'.format(mode,j,wireangle),0);
                y_demods[mode] = demod.demod(y,angle,mode,narrow=False,theta_det=theta_det);
                #y_demods[mode] = demod.demod(y,angle,mode,narrow=False,doBpf=False,doLpf=False,theta_det=theta_det);
                pass;
            for mode in band_modes_narrow :
                out.OUT('demod mode={} (narrow) for angleI={} (wire {}deg.)'.format(mode,j,wireangle),0);
                y_demods_narrow[mode] = demod.demod(y,angle,mode,narrow=True,theta_det=theta_det);
                #y_demods_narrow[mode] = demod.demod(y,angle,mode,narrow=True,doBpf=False,doLpf=False,theta_det=theta_det);
                pass;
         
            ### Drawing WHWP angle + TOD figure for all of the angles ###
            # Draw x: WHWP angle / y: output
            axs[0].plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subave,label='Raw data - Ave.({:.1e}) [angleI={}]'.format(ave,j), marker='x', markersize=0.5,linestyle='',color=colors[j])
            #axs[0].plot(rad_to_deg(theta0to2pi(whwp_angle)),y_subDC,label='Raw data - poly(1) [angleI={}]'.format(j), marker='v', markersize=0.5,linestyle='',color=colors[j])
            # Plot cosmetic
            axs[0].set_title('WHWP angle');
            axs[0].set_xlabel('WHWP angle [deg.]');
            #axs[0].set_title('HWP angle');
            #axs[0].set_xlabel('HWP angle [deg.]');
            axs[0].set_ylabel('ADC output');
            """
            # For SHORT width plot
            axs[0].set_xlim(0.,90.);
            if j==0 :
                bounds = list(axs[0].get_position().bounds);
                print('bounds = ', bounds);
                bounds[2] = bounds[2]*0.25;
                axs[0].set_position(bounds);
                pass;
            """

            axs[0].grid(True);
            axs[0].legend(mode = 'expand',loc='upper left',bbox_to_anchor=(1.05,0.1),framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);

            # Draw TOD x: bolotime / y: output
            time_cut, y_subave_cut = between(time, y_subave, 5., 15.);
            plotOneTOD(axs[1], time_cut, y_subave_cut, 
                    color = colors[j], 
                    label = wirelabel,
                    title = 'Raw TOD (subtracted by average)',
                    );

            time_cut, y_demods4_cut = between(time, y_demods[4], 5., 15.);
            # Draw TOD x: bolotime / y: magnitude of output demodulated by 4f
            plotOneTOD(axs[2], time_cut, np.abs(y_demods4_cut), 
                    color = colors[j], 
                    label = wirelabel,
                    title = 'Demodulated by 4f (bandwidth={}) [mag.]'.format(bandwidth),
                    );
            plotOneTOD(axs[3], time_cut, y_demods4_cut.real, 
                    color = colors[j], 
                    label = wirelabel,
                    title = 'Demodulated by 4f (bandwidth={}) [real]'.format(bandwidth),
                    );

            ### End of WHWP angle + TOD figure for all of the angles ###


            ### Drawing FFT figure for all of the angles ###
            # FFT of raw TOD
            plotOneFFT(axsFFT[0], time, y_subave, color=colors[j], 
                    label=wirelabel, 
                    title='Raw TOD (subtracted by average)', 
                    time_lim=None, freq_lim=[0.,10.], addlegend=True if j==len(angleDataList)-1 else False);
            # FFT of demodulated by 1f (narrow bandwidth)
            timemin = time[0];
            timemax = time[-1];
            if len(time) > numtaps * 2 :
                timemin = time[numtaps];
                timemax = time[-numtaps];
                pass;
            plotOneFFT(axsFFT[1], time, y_demods_narrow[1], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 1f (bandwidth={})'.format(bandwidth_narrow), 
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);j
            plotOneFFT(axsFFT[2], time, y_demods_narrow[2], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 2f (bandwidth={})'.format(bandwidth_narrow), 
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);
            plotOneFFT(axsFFT[3], time, y_demods_narrow[4], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 4f (bandwidth={})'.format(bandwidth_narrow), 
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);
            ### End of FFT figure for all of the angles ###

            ### Drawing FFT figure for all of the angles [NORMALIZED] ###
            # FFT of raw TOD
            plotOneFFT(axsFFTnorm[0], time, y_subave, color=colors[j], 
                    label=wirelabel, 
                    title='Raw TOD (subtracted by average)', 
                    doNorm = True, normFreq = 8.,
                    time_lim=None, freq_lim=[0.,10.], addlegend=True if j==len(angleDataList)-1 else False);
            # FFT of demodulated by 1f (narrow bandwidth)
            timemin = time[0];
            timemax = time[-1];
            if len(time) > numtaps * 2 :
                timemin = time[numtaps];
                timemax = time[-numtaps];
                pass;
            plotOneFFT(axsFFTnorm[1], time, y_demods_narrow[1], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 1f (bandwidth={})'.format(bandwidth_narrow), 
                    doNorm = True, normFreq = 8.,
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);
            plotOneFFT(axsFFTnorm[2], time, y_demods_narrow[2], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 2f (bandwidth={})'.format(bandwidth_narrow), 
                    doNorm = True, normFreq = 8.,
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);
            plotOneFFT(axsFFTnorm[3], time, y_demods_narrow[4], color=colors[j], 
                    #label=wirelabel, 
                    title='Demodulated by 4f (bandwidth={})'.format(bandwidth_narrow), 
                    doNorm = True, normFreq = 8.,
                    time_lim=[timemin,timemax], freq_lim=[0.,10.]);
            ### End of FFT figure for all of the angles [NORMALIZED] ###


            ### Drawing demoded TOD figure for one angle ###
            # No time cut
            plotDemodFigure(time, y_demods, y_demods_narrow,
                    band_modes, band_modes_narrow, bandwidth, bandwidth_narrow,
                    title = '{}: {} [angleI={}]'.format(outname,boloname,j),
                    outdir=outdir, outname='{}{}_demod_angle{}'.format(outname,boloname,j), 
                    out=out, ext=ext, time_lim=None);
            # Time cut [numtaps, tod_size-numtaps]
            if len(time) > numtaps * 2 :
                timemin = time[numtaps];
                timemax = time[-numtaps];
                plotDemodFigure(time, y_demods, y_demods_narrow, 
                        band_modes, band_modes_narrow, bandwidth, bandwidth_narrow,
                        title = '{}: {} [angleI={}]'.format(outname,boloname,j),
                        outdir=outdir, outname='{}{}_demod_angle_timecut{}'.format(outname,boloname,j), 
                        out=out, ext=ext, time_lim=[timemin, timemax]);
                pass;
            ### End of demoded TOD figure for one angle ###


            if len(time) > numtaps * 2 :
                timemin = time[numtaps];
                timemax = time[-numtaps];
                time_cut, y_demods4_narrow_cut = between(time, y_demods_narrow[4], timemin, timemax);

                real     = np.median(y_demods4_narrow_cut.real);
                real_err = rms(y_demods4_narrow_cut.real-real);
                imag     = np.median(y_demods4_narrow_cut.imag);
                imag_err = rms(y_demods4_narrow_cut.imag-imag);
                mag      = np.median(np.abs(y_demods4_narrow_cut));
                mag_err  = rms(np.abs(y_demods4_narrow_cut)-mag);

                demod_result['real'    ] = real     ;
                demod_result['real_err'] = real_err ;
                demod_result['imag'    ] = imag     ;
                demod_result['imag_err'] = imag_err ;
                demod_result['mag'     ] = mag      ;
                demod_result['mag_err' ] = mag_err  ;

                demod_result['real_mK'    ] = real    * cal ;
                demod_result['real_mK_err'] = np.sqrt(np.power(real*cal_err,2.) + np.power(real_err*cal,2.)) ; # include calibration error
                demod_result['imag_mK'    ] = imag    * cal ;
                demod_result['imag_mK_err'] = np.sqrt(np.power(imag*cal_err,2.) + np.power(imag_err*cal,2.)) ; # include calibration error
                demod_result['mag_mK'     ] = mag     * cal ;
                demod_result['mag_mK_err' ] = np.sqrt(np.power(mag*cal_err,2.) + np.power(mag_err*cal,2.)) ; # include calibration error

                demod_results.append(demod_result);
                pass;

            pass; # End of loop over OneAngleDatas
       
        out.OUT('Saving plot ({}_{}...) for [{}]:{}'.format(outpath, boloname, i, boloname),0);
        saveFig(fig, '{}{}'.format(outpath,boloname), ext);
        plt.close(fig);

        out.OUT('Saving plot ({}FFT_{}...) for [{}]:{}'.format(outpath, boloname, i, boloname),0);
        saveFig(figFFT, '{}FFT_{}'.format(outpath,boloname), ext);
        plt.close(figFFT);

        out.OUT('Saving plot ({}FFTnorm_{}...) for [{}]:{}'.format(outpath, boloname, i, boloname),0);
        saveFig(figFFTnorm, '{}FFTnorm_{}'.format(outpath,boloname), ext);
        plt.close(figFFTnorm);


        if len(demod_results)>0 :
            reals     = [result['real'] for result in demod_results ];
            reals_err = [result['real_err'] for result in demod_results ];
            imags     = [result['imag'] for result in demod_results ];
            imags_err = [result['imag_err'] for result in demod_results ];
            mags      = [result['mag'] for result in demod_results ];
            mags_err  = [result['mag_err'] for result in demod_results ];
            reals_mK     = [result['real_mK'] for result in demod_results ];
            reals_mK_err = [result['real_mK_err'] for result in demod_results ];
            imags_mK     = [result['imag_mK'] for result in demod_results ];
            imags_mK_err = [result['imag_mK_err'] for result in demod_results ];
            mags_mK      = [result['mag_mK'] for result in demod_results ];
            mags_mK_err  = [result['mag_mK_err'] for result in demod_results ];
            out.OUTVar(reals    ,0);
            out.OUTVar(reals_err,0);
            out.OUTVar(reals_mK ,0);
            out.OUTVar(reals_mK_err,0);

            figres, axsres = plt.subplots(1,1);
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            figres.set_size_inches(6,6);
            axsres.errorbar(reals,imags,xerr=reals_err,yerr=imags_err,linestyle='',marker='o',markersize=1.,capsize=2.);
            for k in  range(len(reals)) :
                axsres.text(reals[k]+mags[k]/10., imags[k]+mags[k]/10., '{} deg.\n({:.1f},{:.1f})'.format(angleDataList[k]['angle'], reals[k],imags[k]));
                pass;
            axsres.set_xlabel('Real');
            axsres.set_ylabel('Imag.');
            axsres.grid(True);
            out.OUT('Saving plot ({}{}_demod.png) for [{}]:{}'.format(outpath, boloname, i, boloname),0);
            saveFig(figres, '{}{}_demod'.format(outpath,boloname), ext);

            outfile = open('{}/{}{}.pkl'.format(pickledir,outname,boloname), 'wb');
            # copy angleDataList except data key
            tmpDataList = [];
            for angleData in angleDataList :
                tmpData = {};
                for key, value in angleData.items() :
                    if key=='data' : continue;
                    else :
                        tmpData[key] = value;
                        pass;
                    pass;
                tmpDataList.append(tmpData);
                pass;
            pickle.dump(tmpDataList,outfile);
            pickle.dump(reals    ,outfile);
            pickle.dump(reals_err,outfile);
            pickle.dump(imags    ,outfile);
            pickle.dump(imags_err,outfile);
            pickle.dump(mags     ,outfile);
            pickle.dump(mags_err ,outfile);
            pickle.dump(reals_mK    ,outfile);
            pickle.dump(reals_mK_err,outfile);
            pickle.dump(imags_mK    ,outfile);
            pickle.dump(imags_mK_err,outfile);
            pickle.dump(mags_mK     ,outfile);
            pickle.dump(mags_mK_err ,outfile);

            pass;
        
        plt.close();

        pass; # End of loop over bolometers
         

    return 0;



    


def main(boloname, filename='', 
         outdir='aho', outname='aho', pickledir='aho', loadpickledir='aho',
         #theta_det_db=['output_ver2/db/all_mod.db','wiregrid','readout_name'], 
         theta_det_db=None, 
         loaddata=True, out0=None, ext='pdf', verbosity=0 ) :

    # initialize Out
    if out0==None : out = Out.Out(verbosity=verbosity);
    else          : out = out0;

    out.OUT('boloname = {}'.format(boloname), 0);

    # amp is half height of the modulation. It is not the full height.
    db_stim = DBreaderStimulator('./data/pb2a_stimulator_run223_20210223.db', verbose=verbosity);
    stimulator_amp  = [db_stim.getamp(22300607, boloname), db_stim.getamp(22300610, boloname),]; # [ADC counts] Run22300607, Run22300610
    out.OUT(stimulator_amp,-2);
    if stimulator_amp[0][0]==0. or stimulator_amp[1][0]==0. :
        out.WARNING('There is no matched stimulator amplitude data for {}'.format(boloname));
        out.WARNING('The output is {}'.format(stimulator_amp));
        pass;
    out.OUT('stimulator amp (run 22300607) = {} +- {}'.format(stimulator_amp[0][0], stimulator_amp[0][1]), 0);
    out.OUT('stimulator amp (run 22300610) = {} +- {}'.format(stimulator_amp[1][0], stimulator_amp[1][1]), 0);
    # Get stimulator temperature
    # Old version
    #stimulator_temp = 52. ; # 52. [mK_RJ/amp[ADC counts]] @ 90GHz, 103. [mK_RJ/amp[ADC counts]] @ 150GHz
    # New version: Get the number from stimulator template DB
    db_stim_temp = DBreaderStimulator('./data/pb2a_stim_template_20210607.db', tablename='pb2a_stim_template');
    #temps = db_stim_temp.getintensity(channelname=boloname,nearRunID=22300610,source='Jupiter'); # [K_RJ/amp, K_CMB/amp]
    #stimulator_temp = temps[0] * 1000.; # near run jupiter [mK_RJ/amp]
    temps = db_stim_temp.getintensity(channelname=boloname,nearRunID=None,source='Jupiter'); # [K_RJ/amp, K_CMB/amp]
    if temps is None :
        out.WARNING('There is no matched stimulator temperature data for {}/ Jupiter'.format(boloname));
        out.WARNING('The output is {}'.format(temps));
        out.WARNING(' --> Gain calibration constant is set to 0.');
        stimulator_temp = 0.;
    else :
        stimulator_temp = np.mean(np.array(temps)[:,0]) * 1000.; # averaged jupiter [mK_RJ/amp]
        pass;


    # DB for theta_det calibration
    db_theta = None;
    if (not theta_det_db is None) and len(theta_det_db)>2 :
        out.OUT('Read theta_det form DB in {}'.format(theta_det_db[0]),0);
        db_theta = DBreader(theta_det_db[0], theta_det_db[1], theta_det_db[2], verbose=verbosity);
        pass;

    # set calibration constant and its error from ADC output to mK_RJ
    cal     = stimulator_temp/stimulator_amp[1][0]  if stimulator_amp[1][0]>0. else 0.; # chose sitmulator run after calibration
    cal_err = cal/stimulator_amp[1][0] * stimulator_amp[1][1] if stimulator_amp[1][0]>0. else 0.; # chose stimulator run after calibration 
    out.OUT('calibration constant (ADC->mK_RJ) = {:.3f} +- {:.3f}'.format(cal,cal_err),0);

    # ver0
    if 'ver0' in outdir :
        # all angle data but angle values are not correct and angle=0, 22.5 data are bad.
        angleDataList = [
            {'angle':  0. , 'start':'20210205_173930', 'end':'20210205_174430', 'outname':'A-1_-22.5deg'    },
            {'angle': 22.5, 'start':'20210205_174930', 'end':'20210205_175600', 'outname':'A0_0deg'    },
            {'angle': 45  , 'start':'20210205_180230', 'end':'20210205_180400', 'outname':'A1_22.5deg' },
            {'angle': 67.5, 'start':'20210205_180510', 'end':'20210205_180530', 'outname':'A2_45deg'   },
            {'angle': 90. , 'start':'20210205_180640', 'end':'20210205_180700', 'outname':'A3_67.5deg' },
            {'angle':112.5, 'start':'20210205_180800', 'end':'20210205_180820', 'outname':'A4_90deg'   },
            {'angle':135. , 'start':'20210205_181000', 'end':'20210205_181130', 'outname':'A5_112.5deg'},
            {'angle':157.5, 'start':'20210205_181320', 'end':'20210205_181340', 'outname':'A6_135deg'  },
            {'angle':180. , 'start':'20210205_181557', 'end':'20210205_181617', 'outname':'A7_157.5deg'},
            {'angle':202.5, 'start':'20210205_181750', 'end':'20210205_181810', 'outname':'A8_180deg'  },
            {'angle':225. , 'start':'20210205_182000', 'end':'20210205_182400', 'outname':'A9_202.5deg'},
        ];
    # one of ver1--5
    elif  any([(ver in outdir) for ver in ['ver1','ver2','ver3','ver4','ver5']])  :
        # 0--157.5deg, 8 angles (No 180 deg)
        angleDataList = [
            {'angle':  0  , 'start':'20210205_180230', 'end':'20210205_180400', 'outname':'A1_0deg'    },
            {'angle': 22.5, 'start':'20210205_180510', 'end':'20210205_180530', 'outname':'A2_22.5deg' },
            {'angle': 45. , 'start':'20210205_180640', 'end':'20210205_180700', 'outname':'A3_45deg'   },
            {'angle': 67.5, 'start':'20210205_180800', 'end':'20210205_180820', 'outname':'A4_67.5deg' },
            {'angle': 90. , 'start':'20210205_181000', 'end':'20210205_181130', 'outname':'A5_90deg'   },
            {'angle':112.5, 'start':'20210205_181320', 'end':'20210205_181340', 'outname':'A6_112.5deg'},
            {'angle':135. , 'start':'20210205_181557', 'end':'20210205_181617', 'outname':'A7_135deg'  },
            {'angle':157.5, 'start':'20210205_181750', 'end':'20210205_181810', 'outname':'A8_157.5deg'},
        ];
    # otherwise 
    else :
        # 0--180deg, 9 angles
        angleDataList = [
            {'angle':  0  , 'start':'20210205_180230', 'end':'20210205_180400', 'outname':'A1_0deg'    },
            {'angle': 22.5, 'start':'20210205_180510', 'end':'20210205_180530', 'outname':'A2_22.5deg' },
            {'angle': 45. , 'start':'20210205_180640', 'end':'20210205_180700', 'outname':'A3_45deg'   },
            {'angle': 67.5, 'start':'20210205_180800', 'end':'20210205_180820', 'outname':'A4_67.5deg' },
            {'angle': 90. , 'start':'20210205_181000', 'end':'20210205_181130', 'outname':'A5_90deg'   },
            {'angle':112.5, 'start':'20210205_181320', 'end':'20210205_181340', 'outname':'A6_112.5deg'},
            {'angle':135. , 'start':'20210205_181557', 'end':'20210205_181617', 'outname':'A7_135deg'  },
            {'angle':157.5, 'start':'20210205_181750', 'end':'20210205_181810', 'outname':'A8_157.5deg'},
            #{'angle':180. , 'start':'20210205_182000', 'end':'20210205_182400', 'outname':'A9_180deg'  },
            {'angle':180. , 'start':'20210205_182000', 'end':'20210205_182020', 'outname':'A9_180deg'  },
        ];

        pass;


    # retrieve each angle data
    out.OUT('load data from pickle file = {}'.format(not loaddata), 0);
    out.OUT('input pickle file dir      = {}'.format(loadpickledir), 0);
    for i, angleData in enumerate(angleDataList) :
        out.OUTVar(boloname,0);
        oneAngleData = OneAngleData(
                    filename = filename             , boloname  = boloname          ,
                    start    = angleData['start']   , end       = angleData['end']  ,
                    outname  = angleData['outname'] , outdir    = pickledir+'/'+boloname,
                    loadpickledir = loadpickledir+'/'+boloname, 
                    loaddata = loaddata             , out       = out               ,
                );
        angleDataList[i]['data'] = oneAngleData;
        angleDataList[i]['cal'    ] = cal;
        angleDataList[i]['cal_err'] = cal_err;
        pass;

    if boloname!='' : plotAll(angleDataList, db_theta=db_theta, outdir=outdir, outname=outname, pickledir=pickledir, out=out, ext=ext);

    return 0;



if __name__=='__main__' :

    verbose = 1;
    filename='/group/cmb/polarbear/data/pb2a/g3compressed/22300000_v05/Run22300609';
    #boloname='PB20.13.13_Comb01Ch01';
    boloname='PB20.13.13_Comb01Ch02';
    outname ='ver1_';
    outdir  ='plot_ver1';
    pickledir = 'plot'; # write directory for demod data
    loadpickledir = ''; # read directory for TOD data retrieved from g3 file (If it is empty, it will be 'pickledir')
    ext     ='png';

    parser = argparse.ArgumentParser();
    parser.add_argument('-b', '--boloname', default=boloname, help='boloname (default: {}): If boloname=="", make data and stop job. (no making plot or demodulation) '.format(boloname));
    parser.add_argument('-f', '--filename', default=filename, help='input g3 filename (default: {})'.format(filename));
    parser.add_argument('-d', '--outdir', default=outdir, help='output directory for the plots (default: {})'.format(outdir));
    parser.add_argument('-o', '--outname', default=outname, help='output filename (default: {})'.format(outname));
    parser.add_argument('-p', '--pickledir', default=pickledir, help='write directory for pickle files to save the data retreived from g3 file (default: {})'.format(pickledir));
    parser.add_argument('-l', '--loadpickledir', default=pickledir, help='read directory for pickle files to load the data retreived from g3 file (default=pickledir)');
    parser.add_argument('-L', '--loadpickle', dest='loaddata', action='store_false', default=True, 
            help='Whether load pickle file or not. If not load it, it will load raw data file. (default: False)');
    parser.add_argument('-e', '--extension', default=ext, help='Output file extensions for figure: You can set multiple extensions by "," e.g. "pdf,png". (default: {})'.format(ext));
    parser.add_argument('--anglecalib', default=None, help='Input database for angle calibration of theta_det e.g. "<DB filename>,<tablename>,<columnname for bolo>". (default: None)');
    parser.add_argument('-v', '--verbose', default=verbose, type=int, help='verbosity level: A larger number means more printings. (default: {})'.format(verbose));
    args = parser.parse_args();

    if args.loadpickledir=='' : loadpickledir = args.pickledir;
    else                      : loadpickledir = args.loadpickledir;
    theta_det_db = None;
    if not args.anglecalib is None : theta_det_db = args.anglecalib.split(',');

    out = Out.Out(args.verbose);
    out.OUT('loaddata = {}'.format(args.loaddata),1)

    main(boloname=args.boloname, filename=args.filename, 
            outdir=args.outdir, outname=args.outname, pickledir=args.pickledir, loadpickledir=loadpickledir,
            theta_det_db = theta_det_db,
            loaddata=args.loaddata, out0=out, ext=args.extension);
    pass;


