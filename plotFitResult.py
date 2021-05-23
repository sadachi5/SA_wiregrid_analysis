import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import Out;
from utils import *;

def plotFitResult(func, pars, chisqr, x, y, xerr, yerr, wireangles, outpath, outsuffix='tmp', ext='png', errs=None, usemK=False) :
    # Retrieve parameters
    x0 = pars[0];
    y0 = pars[1];
    a  = pars[2];
    b  = pars[3];
    alpha= pars[4];
    x0err = errs[0] if not errs is None else 0.;
    y0err = errs[1] if not errs is None else 0.;
    aerr  = errs[2] if not errs is None else 0.;
    berr  = errs[3] if not errs is None else 0.;
    alphaerr= errs[4] if not errs is None else 0.;
    r   = (a+b)/2.;
    rerr= np.sqrt((aerr**2. + berr**2.)/4.);
    
    # Calculate differences
    thetas      = [];
    theta_errs  = [];
    drs         = []; # fitted_r - data_r
    dr_errs     = [];
    dthetas     = []; # wire_angle*2 - fitted_theta
    dtheta_errs = [];
    """
    printVar(x);
    printVar(y);
    printVar(xerr);
    printVar(yerr);
    printVar(wireangles);
    """
    for _x, _y, _xerr, _yerr, _wireangle in zip(x, y, xerr, yerr, wireangles) :
        _X = _x - x0;
        _Y = _y - y0;
        _Xerr = np.sqrt( _xerr**2. + x0err**2.);
        _Yerr = np.sqrt( _yerr**2. + y0err**2.);
        (_r, _rerr), (_theta, _thetaerr) = calculateRThetaEllipse(
                _X, _Y, alpha, a, b, 
                _Xerr, _Yerr, alphaerr, aerr, berr);
        thetas.append(_theta);
        theta_errs.append(_thetaerr);
        _dr    = _r - r;
        _drerr = np.sqrt( _rerr**2. + rerr**2. );
        drs    .append(_dr);
        dr_errs.append(_drerr);
        _dfit_theta = theta0to2pi( (thetas[0]+2.*np.pi) - _theta );
        #printVar(_dfit_theta);
        _dtheta     = _dfit_theta - _wireangle * 2.;
        _dthetaerr  = _thetaerr;
        dthetas     .append(_dtheta);
        dtheta_errs .append(_dthetaerr);
        pass;
    printVar(drs);
    printVar(dthetas);
 
    # Make funciton of the fitted circle
    delta = 1.;
    xrange = np.arange(-r*1.2,r*1.2,delta);
    yrange = np.arange(-r*1.2,r*1.2,delta);
    X,Y = np.meshgrid(xrange, yrange);
    Z = func(X,Y,pars);

    # Setup figure
    fig, axs = plt.subplots(3,1);
    fig.tight_layout(rect=[0,0,1,1]);
    fig.set_size_inches(6,12);
    plt.subplots_adjust(wspace=1, hspace=1, left=0.15, right=0.85,bottom=0.15, top=0.85)
    axsScale = [1./2., 1./4., 1./4.]; # scale against total figure size for each axes
    marginW = 0.17;
    marginH = 0.17;
    fillspaceW = 1.-marginW*2.;
    fillspaceH = 1.-marginH*2.;
    marginasymmW = 0.03;
    marginasymmH = 0.00;
    marginasymmH2= 0.16;
    axs[0].set_position([(marginW+marginasymmW),1.-sum(axsScale[:0+1])+axsScale[0]*(marginH+marginasymmH),fillspaceW,axsScale[0]*fillspaceH]); # left, bottom, width, height
    axs[1].set_position([(marginW+marginasymmW),1.-sum(axsScale[:1+1])+axsScale[1]*(marginH+marginasymmH2),fillspaceW,axsScale[1]*fillspaceH]); # left, bottom, width, height
    axs[2].set_position([(marginW+marginasymmW),1.-sum(axsScale[:2+1])+axsScale[2]*(marginH+marginasymmH2),fillspaceW,axsScale[2]*fillspaceH]); # left, bottom, width, height

    # Top plot
    # Draw data points
    axs[0].errorbar(x,y,xerr=xerr,yerr=yerr,linestyle='',marker='o',markersize=1.,capsize=2.);
    # Draw center
    axs[0].errorbar([0.,pars[0]],[0.,pars[1]],xerr=[0.,0.],yerr=[0.,0.],linestyle='-',marker='o',markersize=1.,capsize=2.,color='tab:brown');
    # Draw fitted circle
    axs[0].contour(X,Y,Z,[0],colors='r',linewidths=0.5);
    # Cosmetic
    xlabel = r'Real [$mK_\mathrm{RJ}$]' if usemK else 'Real [ADC]';
    ylabel = r'Imag. [$mK_\mathrm{RJ}$]' if usemK else 'Imag. [ADC]';
    axs[0].set_title('$\chi ^2$={}: {}'.format(chisqr, outsuffix));
    axs[0].set_xlabel(xlabel,fontsize=16);
    axs[0].set_ylabel(ylabel,fontsize=16);
    axs[0].set_xlim(-5000,5000);
    axs[0].set_ylim(-5000,5000);
    axs[0].tick_params(labelsize=12);
    axs[0].grid(True);
    # Draw ideal radial lines at each 45 deg
    # reference to i=0 wire angle
    for k in range(8) :
        _theta_ideal = -k*np.pi/4. + thetas[0];
        _X = a*np.cos(_theta_ideal-alpha)
        _Y = b*np.sin(_theta_ideal-alpha)
        _x = np.cos(alpha)*_X - np.sin(alpha)*_Y;
        _y = np.sin(alpha)*_X + np.cos(alpha)*_Y;
        _x_line = x0+_x;
        _y_line = y0+_y;
        # Draw
        axs[0].plot([x0,_x_line],[y0,_y_line],linestyle=':',linewidth=0.8,color='tab:red');
        pass;

    # Middle plot
    # Draw data points
    axs[1].errorbar(rad_to_deg_0to2pi(thetas),drs,xerr=rad_to_deg_0to2pi(theta_errs),yerr=dr_errs,linestyle='',marker='o',markersize=1.,capsize=2.);
    # Draw 0 line
    axs[1].plot([0.,360.],[0,0],linestyle='-',linewidth=1.0,marker='',color='k');
    # Cosmetic
    xlabel = r'$\theta$ in HWP complex [deg.]';
    ylabel = r'$r_{data} - r_{fit}$ '+(r'[$mK_\mathrm{RJ}$]' if usemK else '[ADC unit]');
    axs[1].set_xlabel(xlabel,fontsize=16);
    axs[1].set_ylabel(ylabel,fontsize=16);
    axs[1].tick_params(labelsize=12);
    axs[1].set_xlim(0,360.);
    axs[1].grid(True);
 
    # Bottom plot
    # Draw data points
    axs[2].errorbar(rad_to_deg_0to2pi(wireangles),rad_to_deg_pitopi(dthetas),xerr=None,yerr=rad_to_deg_pitopi(dtheta_errs),linestyle='',marker='o',markersize=1.,capsize=2.);
    # Draw 0 line
    axs[2].plot([0.,360.],[0,0],linestyle='-',linewidth=1.0,marker='',color='k');
    # Cosmetic
    xlabel = r'Wire angle $\theta_{\mathrm{Wire}}$ [deg.]';
    ylabel = r'$\theta_{\mathrm{HWP complex}} - 2\theta_{wire}$ - const. [deg.]';
    axs[2].set_xlabel(xlabel,fontsize=16);
    axs[2].set_ylabel(ylabel,fontsize=16);
    axs[2].tick_params(labelsize=12);
    axs[2].set_xlim(0,180.);
    axs[2].grid(True);


    outfigname = '{}_{}'.format(outpath,outsuffix);
    print('Saving figure : {}.{}'.format(outfigname,ext));
    saveFig(fig, outfigname, ext);
    return;


