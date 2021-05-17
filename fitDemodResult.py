import os, sys
import argparse;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle
import math
import copy
from scipy.optimize import curve_fit
from scipy.fftpack import fft, fftfreq;

import Out;
from utils import *;

from minuitfit import *;

def main(picklefile, outdir, outname, excludeAngle=[], refAngle=0., usemK=False, out0=None, verbosity=0, ext='png') :

    # initialize Out
    if out0==None : out = Out.Out(verbosity=verbosity);
    else          : out = out0;
    # output filepath
    outpath = '{}/{}'.format(outdir, outname);

    pklfile = open(picklefile, 'rb');
    angleDataList = pickle.load(pklfile);
    reals_org     = pickle.load(pklfile);
    reals_err_org = pickle.load(pklfile);
    imags_org     = pickle.load(pklfile);
    imags_err_org = pickle.load(pklfile);
    mags_org      = pickle.load(pklfile);
    mags_err_org  = pickle.load(pklfile);
    reals_mK_org     = pickle.load(pklfile);
    reals_mK_err_org = pickle.load(pklfile);
    imags_mK_org     = pickle.load(pklfile);
    imags_mK_err_org = pickle.load(pklfile);
    mags_mK_org      = pickle.load(pklfile);
    mags_mK_err_org  = pickle.load(pklfile);

    def fitfunc(pars, x, y) :
        return (x-pars[0])**2. + pars[3]*(y-pars[1])**2. - pars[2]**2.;

    def createfitsquare(func, x,y,xerr,yerr) :
        def fitsquare(*pars) :
            diffsquare = func(pars, x, y);
            square = np.sum( np.abs(diffsquare) / (np.power(xerr,2.)+np.power(np.multiply(yerr,pars[3]),2.)) );
            #print(square);
            return square;
        return fitsquare;

    err = 0.01;
    errdef = 1;
    init_pars  = [0., 0., 5000., 1.];
    limit_pars = [[-1000.,1000.], [-1000.,1000.], [1000., 5000.], [0.8, 1.2]];
    error_pars = [1.e-5,1.e-5,1.e-5,1.e-5];
    #fix_pars   = [True,True,False,True];
    fix_pars   = [False,False,False,False];

    # Prepare data
    if usemK :
        reals     = reals_mK_org;
        reals_err = reals_mK_err_org;
        imags     = imags_mK_org;
        imags_err = imags_mK_err_org;
    else :
        reals     = reals_org;
        reals_err = reals_err_org;
        imags     = imags_org;
        imags_err = imags_err_org;
        pass;
    x_fit       = copy.deepcopy(reals    );
    xerr_fit    = copy.deepcopy(reals_err);
    y_fit       = copy.deepcopy(imags    );
    yerr_fit    = copy.deepcopy(imags_err);
    
    # calcualte rough estimate of the paramters
    x0_tmp = np.mean(x_fit);
    y0_tmp = np.mean(x_fit);
    i_0deg = 0;
    i_90deg= 0;
    for k, angleData in enumerate(angleDataList) :
        if angleData['angle']== 0 : i_0deg = k;
        if angleData['angle']==90 : i_90deg= k;
        pass;
    rtheta_tmp = calculateRTheta(reals[i_0deg]-reals[i_90deg], imags[i_0deg]-imags[i_90deg]);
    r_tmp  = rtheta_tmp[0]/2.;
    # set initial values
    init_pars[0] = x0_tmp;
    init_pars[1] = y0_tmp;
    init_pars[2] = r_tmp ;
    # set limits on parameters
    limit_pars[0][0] = x0_tmp - r_tmp*0.2;
    limit_pars[0][1] = x0_tmp + r_tmp*0.2;
    limit_pars[1][0] = y0_tmp - r_tmp*0.2;
    limit_pars[1][1] = y0_tmp + r_tmp*0.2;
    limit_pars[2][0] = r_tmp*0.8;
    limit_pars[2][1] = r_tmp*1.2;
    #limit_pars[2][0] = r_tmp*0.6;
    #limit_pars[2][1] = r_tmp*1.4;
    #limit_pars[2][0] = r_tmp*0.1;
    #limit_pars[2][1] = r_tmp*10.;

    # Check excludeAngle
    exclude_indices = [];
    for ex_angle in excludeAngle :
        for k, angleData in enumerate(angleDataList) :
            angleData  = angleDataList[k];
            if angleData['angle'] == ex_angle : exclude_indices.append(k);
            pass;
        pass;
    exclude_indices.sort(reverse=True); # sorted by reversed order
    # Exclude angles
    out.OUTVar(exclude_indices,'',-1);
    for k in exclude_indices :
        del x_fit[k], xerr_fit[k], y_fit[k], yerr_fit[k];
        pass;
    out.OUT('real [x] (no cut)         = {}'.format(reals),-1);
    out.OUT('x_fit (after excludeAngle) = {}'.format(x_fit),-1);

    fitsquare = createfitsquare(fitfunc, x_fit, y_fit, xerr_fit, yerr_fit);
    result, minuit = minuitminosfit(fitsquare, init=init_pars, fix=fix_pars, limit=limit_pars, error=error_pars, errordef=errdef, precision=1.e-10, verbosity=2);
    pars = result[0];
    r    = pars[2];
    errs = result[3];
    rerr = errs[2];

    # Print fit results
    parlabels = ['Center of real', 'Center of imag.', 'Raidus', 'Coefficient on imag. term'];
    printResult(result, out=out,
            parlabels = parlabels,
            );
    x0    = pars[0];
    x0err = errs[0];
    y0    = pars[1];
    y0err = errs[1];
    (r0, r0_err), (theta0, theta0_err) = calculateRTheta(x0, y0, x0err, y0err);
    theta0_deg = rad_to_deg(theta0);
    theta0_err_deg = rad_to_deg(theta0_err);
    out.OUT('(x0,y0) = ({:8.3f}, {:8.3f}) +- ({:5.3f}, {:5.3f})'.format(x0,y0,x0err,y0err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('r0 (center)', r0, r0_err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('theta0 (center) [rad.]', theta0, theta0_err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('theta0 (center) [deg.]', rad_to_deg(theta0), rad_to_deg(theta0_err)),0);

    # Draw minuit contours
    # x0 v.s. y0
    drawcontour2(minuit, fix_pars, parIDs=[0,1], parLabels=['x0','y0'],
            center=[pars[0],pars[1]], dxdy=[r*0.05,r*0.05],
        outname='{}_{}'.format(outpath,'contour_x0-y0'), out=out);
    # x0 v.s. r
    drawcontour2(minuit, fix_pars, parIDs=[0,2], parLabels=['x0','r'], 
            center=[pars[0],pars[2]], dxdy=[r*0.05, r*0.05],
        outname='{}_{}'.format(outpath,'contour_x0-r'), out=out);
    # y0 v.s. r
    drawcontour2(minuit, fix_pars, parIDs=[1,2], parLabels=['y0','r'], 
            center=[pars[1],pars[2]],  dxdy=[r*0.05, r*0.05],
        outname='{}_{}'.format(outpath,'contour_y0-r'), out=out);
    del minuit;

    # Calculate & print each angle of each wire data
    fitRThetas = [];
    for k, angleData in enumerate(angleDataList) :
        x1    = reals[k];
        x1err = reals_err[k];
        y1    = imags[k];
        y1err = imags_err[k];
        (r1, r1_err), (theta1, theta1_err) = calculateRTheta(x1-x0, y1-y0, np.sqrt(x1err**2.+x0err**2.), np.sqrt(y1err**2.+y0err**2.));
        theta1_deg     = rad_to_deg(theta1);
        theta1_err_deg = rad_to_deg(theta1_err);
        fitRThetas.append({'wireangle':angleData['angle'], 'r':[r1,r1_err], 'theta':[theta1,theta1_err], 'theta_deg':[theta1_deg, theta1_err_deg]});
        pass;
    
    # Make funciton of the fitted circle
    delta = 1.;
    xrange = np.arange(-r*1.2,r*1.2,delta);
    yrange = np.arange(-r*1.2,r*1.2,delta);
    X,Y = np.meshgrid(xrange, yrange);
    Z = (X-pars[0])**2. + pars[3]*(Y-pars[1])**2. - pars[2]**2.;

    # Draw plot
    figres, axsres = plt.subplots(2,1);
    figres.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=1, hspace=1, left=0.15, right=0.85,bottom=0.15, top=0.85)
    axsScale = [2./3., 1./3.]; # scale against total figure size for each axes
    marginW = 0.17;
    marginH = 0.17;
    fillspaceW = 1.-marginW*2.;
    fillspaceH = 1.-marginH*2.;
    marginasymmW = 0.03;
    marginasymmH = 0.00;
    marginasymmH2= 0.16;
    axsres[0].set_position([(marginW+marginasymmW),1.-sum(axsScale[:0+1])+axsScale[0]*(marginH+marginasymmH),fillspaceW,axsScale[0]*fillspaceH]); # left, bottom, width, height
    axsres[1].set_position([(marginW+marginasymmW),1.-sum(axsScale[:1+1])+axsScale[1]*(marginH+marginasymmH2),fillspaceW,axsScale[1]*fillspaceH]); # left, bottom, width, height
    figres.set_size_inches(6,9);
    # Draw data points
    axsres[0].errorbar(reals,imags,xerr=reals_err,yerr=imags_err,linestyle='',marker='o',markersize=1.,capsize=2.);
    shiftx=r*0.04;
    shifty=r*0.04;
    fitTheta0 = None;
    for k in  range(len(reals)) :
        fitTheta = fitRThetas[k]['theta_deg'];
        if fitRThetas[k]['wireangle'] == refAngle : fitTheta0 = fitTheta;
        
        x_tmp = reals[k];
        y_tmp = imags[k]+shifty;
        #if k==0 : x_tmp = reals[k]-0.*shiftx;
        #if k==0 : y_tmp = imags[k]+4.*shifty;
        #if k==1 : x_tmp = reals[k]+0.*shiftx   ;
        #if k==1 : y_tmp = imags[k]-5.*shifty;
        #if k==10 : x_tmp = reals[k]+0.*shiftx   ;
        #if k==10 : y_tmp = imags[k]-5.*shifty;
        #axsres[0].text(x_tmp, y_tmp, '{} deg.\n({:.1f} , {:.1f})\n'.format(angleDataList[k]['angle'], reals[k],imags[k])+r'$\theta$'+' = {:.2f} +- {:.2f} deg.'.format(fitTheta[0],fitTheta[1]), color='tab:blue');
        axsres[0].text(x_tmp, y_tmp, '{} deg.\n'.format(angleDataList[k]['angle'])+r'$\theta$'+' = {:.2f} +- {:.2f} deg.'.format(fitTheta[0],fitTheta[1]), fontsize=10, color='tab:blue');
        pass;
    # Draw center
    axsres[0].errorbar([0.,x0],[0.,y0],xerr=[0.,x0err],yerr=[0.,y0err],linestyle='-',marker='o',markersize=1.,capsize=2.,color='tab:brown');
    axsres[0].text(x0+shiftx, y0+0.5*shifty, 'Center: \n({:.2f}+-{:.2f} , {:.2f}+-{:.2f})'.format(x0,x0err,y0,y0err),color='tab:brown');
    axsres[0].text(shiftx, -4.*shifty, r'$r_0$={:.2f}+-{:.2f}'.format(r0,r0_err)+'\n'+r'$\theta_0=${:.2f}+-{:.2f} deg.'.format(theta0_deg,theta0_err_deg),fontsize=10,color='tab:brown');
    # Draw circle
    axsres[0].contour(X,Y,Z,[0],colors='r',linewidths=0.5);
    axsres[0].text(-4800,-4000, 'Radius: \n{:.2f}+-{:.2f}'.format(r,rerr),color='tab:red');
    # Cosmetic
    xlabel = r'Real [$mK_\mathrm{RJ}$]' if usemK else 'Real [ADC]';
    ylabel = r'Imag. [$mK_\mathrm{RJ}$]' if usemK else 'Imag. [ADC]';
    axsres[0].set_title(outname);
    axsres[0].set_xlabel(xlabel,fontsize=16);
    axsres[0].set_ylabel(ylabel,fontsize=16);
    axsres[0].set_xlim(-5000,5000);
    axsres[0].set_ylim(-5000,5000);
    axsres[0].tick_params(labelsize=12);
    axsres[0].grid(True);

    # Draw ideal radial lines at each 45 deg
    for k in range(8) :
        theta_ideal = deg_to_rad(-k*45. + fitTheta0[0]);
        x_line = x0+r*np.cos(theta_ideal);
        y_line = y0+r*np.sin(theta_ideal);
        # Draw
        axsres[0].plot([x0,x_line],[y0,y_line],linestyle=':',linewidth=0.8,color='tab:red');
        pass;


    # Angle plot
    # Calculate the d_theta
    theta_wires = [];
    d_thetas = [];
    d_theta_errs = [];
    for k in  range(len(reals)) :
        fitTheta = fitRThetas[k]['theta_deg'];
        theta_wires.append(fitRThetas[k]['wireangle']);
        out.OUT('k={} wire angle = {:.1f}, fit angle = {:.1f}, fit angle (0deg) = {:.1f}'.format(k, theta_wires[-1], fitTheta[0], fitTheta0[0]),0);
        d_fittheta = fitTheta0[0] - (fitTheta[0] if (fitTheta[0]<=fitTheta0[0]) else (fitTheta[0]-360.));
        out.OUT('k={} d_fittheta = {:.1f}'.format(k, d_fittheta),0);
        d_theta = (theta_wires[-1]-refAngle)*2. - d_fittheta;
        d_thetas.append(d_theta);
        d_theta_errs.append(fitTheta[1]);
        pass;
    theta_wires = np.array(theta_wires);
    d_thetas    = np.array(d_thetas);
    # Draw
    axsres[1].errorbar(theta_wires,d_thetas,None,d_theta_errs,linestyle='-',marker='o',markersize=1.,capsize=2.,color='tab:blue');
    # Cosmetic
    xlabel = r'$\theta_{wire}$ [deg.]';
    #ylabel = r'$2\theta_{wire}$ - ' +'\n' +r'$\Delta (\theta_{fit}(\theta_{wire})$,$\theta_{fit}(0))$ [deg.]';
    ylabel = r'$2\theta_{wire}$ - $\Delta\theta_{fit}(\theta_{wire},0)$ [deg.]';
    # Draw 0 line
    axsres[1].plot([0.,180.],[0,0],linestyle='-',linewidth=0.5,marker='',color='k');
    axsres[1].set_xlabel(xlabel,fontsize=16);
    axsres[1].set_ylabel(ylabel,fontsize=16);
    axsres[1].set_xlim(refAngle,refAngle+180.);
    axsres[1].set_ylim(min(d_thetas[theta_wires>=refAngle]),max(d_thetas[theta_wires>=refAngle]));
    axsres[1].tick_params(labelsize=12);
    axsres[1].grid(True);
     


    # Save plot
    out.OUT('Saving plot ({}_demodfitresult...)'.format(outpath),0);
    saveFig(figres, '{}_demodfitresult'.format(outpath), ext);

    return 0;

 


if __name__=='__main__' :

    boloname = 'PB20.13.13_Comb01Ch01';
    #boloname = 'PB20.13.13_Comb01Ch02';

    if len(sys.argv)>1 :
        boloname = sys.argv[1];
        print('boloname = ', boloname);
        pass;


    '''
    picklefile  = 'plot_ver0/ver0_PB20.13.13_Comb01Ch01_demodresult.pkl';
    outname     = 'ver0';
    outdir      = 'plot_ver0';
    excludeAngle= [0.,22.5] ;
    refAngle    = 45.; # deg
    #'''

    #'''
    picklefile  = 'plot_ver1/ver1_{}_demodresult.pkl'.format(boloname);
    outname     = 'ver1_{}'.format(boloname);
    outdir      = 'plot_ver1';
    excludeAngle= [] ;
    refAngle    = 0.; # deg
    #'''

    ext         = 'png';
    main(picklefile, usemK=True, refAngle=refAngle, outdir=outdir, outname=outname, excludeAngle=excludeAngle, ext=ext, verbosity=2);
    pass;

