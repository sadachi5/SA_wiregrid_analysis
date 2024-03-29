import os, sys
import argparse
import gc
import traceback
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
import sqlite3

import Out;
from utils import *;
from plotFitResult import plotFitResult;

from minuitfit import *;
import lmfit;
from LMfit import LMfit, getParamsValues;


def main(picklefile, boloname, outdir, outname, excludeAngle=[], refAngle=0., usemK=False, out0=None, verbosity=0, ext='png', initalpha_frac=None, nLMfit=1, batchmode=False, drawExcludeAngle=False, fineCircle=False, fitCircle=False, limUnit=2500) :
    print('##### fitDemodResult for {} ########################'.format(boloname));

    # initialize Out
    if out0==None : out = Out.Out(verbosity=verbosity);
    else          : out = out0;
    # output filepath
    wafername = getwafername(boloname);
    outpath   = '{}/plot/{}/{}/{}'.format(outdir, wafername, boloname, outname);
    outdbdir  = '{}/db/{}'.format(outdir, wafername);
    outdbpath = '{}/db/{}/{}'.format(outdir, wafername, outname);
    out.OUT('outpath = {}'.format(outpath),0);
    out.OUT('input picklefile = {}'.format(picklefile),0);

    if not os.path.isfile(picklefile) :
        out.ERROR('No {}'.format(picklefile));
        return -1;
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

    """
    def fitfunc(x,y, pars) :
        # Ax*x + By*y + C*xy = 1 (A=1/(r'*r'))
        # pars[0] : x0
        # pars[1] : y0
        # pars[2] : r'*r' (=1/A)
        # pars[3] : 1/B
        # pars[4] : C
        return (x-pars[0])**2. + (pars[2]/pars[3])*(y-pars[1])**2. + (pars[2]*pars[4])*(x-pars[0])*(y-pars[1]) - pars[2];
    """
    
    def fitfunc(x,y, pars) :
        # x*x + B/A*y*y + C/A*xy = 1/A (A=1/(r'*r'))
        # A = (cos(alpha)/a)**2 + (sin(alpha)/b)**2
        # B = (sin(alpha)/a)**2 + (cos(alpha)/b)**2
        # C = (1/(a*a) - 1/(b*b))*2.*sin(alpha)*cos(alpha)
        x0 = pars[0];
        y0 = pars[1];
        x1 = x-pars[0];
        y1 = y-pars[1];
        theta = np.arctan2(y,x);
        a  = pars[2];
        b  = pars[3];
        alpha = pars[4];
        A = np.power(np.cos(alpha)/a, 2.) + np.power(np.sin(alpha)/b, 2.);
        B = np.power(np.sin(alpha)/a, 2.) + np.power(np.cos(alpha)/b, 2.);
        C = np.multiply(  np.divide(1, np.power(a,2.)) - np.divide(1, np.power(b,2.)) ,  2.*np.sin(alpha)*np.cos(alpha) );
        X_fit = a*np.cos(theta-alpha);
        Y_fit = b*np.sin(theta-alpha);
        cosa  = np.cos(alpha);
        sina  = np.sin(alpha);
        x_fit = cosa*X_fit - sina*Y_fit;
        y_fit = sina*X_fit + cosa*Y_fit;
        r_fit = np.sqrt(np.power(x_fit,2.) + np.power(y_fit,2.));
        r1    = np.sqrt(np.power(x1,2.) + np.power(y1,2.));
        diff0   = (x1**2. + B/A*y1**2. + C/A*x1*y1 - 1./A);
        diff1   = (r_fit - r1)*np.abs(r_fit - r1);
        #print('0th term: {}'.format(x1**2.));
        #print('1st term: {}'.format(B/A*y1**2.));
        #print('2nd term: {}'.format(C/A*x1*y1));
        #print('r^2: {}'.format(1/A));
        #print('x1**2. + B/A*y1**2. + C/A*x1*y1 - 1./A: {}'.format(diff0));
        #print('|r_fit - r1|*(r_fit - r1) : {}'.format(diff1));
        #return diff0;
        return diff1;

    def createfitsquare(func, x,y,xerr,yerr, absame=False) :
        def fitsquare(*pars) :
            new_pars = [ par for par in pars ];
            if absame : new_pars[3]=new_pars[2]; # b=a for fitCircle
            diffsquare = func(x, y, new_pars);
            #err_square = np.power(xerr,2.)+np.power(np.multiply(yerr,pars[3]),2.);
            #err_square = np.power(xerr,2.)+np.power(np.multiply(yerr,pars[2]/pars[3]),2.); # NOTE: Should be improved.
            err_square = np.power(xerr,2.)+np.power(yerr,2.); # NOTE: Should be improved.
            square = np.sum( np.abs(diffsquare) / (err_square) );
            print('diff^2 = ',diffsquare);
            print('err^2  = ',err_square);
            print('(diff/err)^2  = ',np.abs(diffsquare)/err_square);
            print('square = ',square);
            return square;
        return fitsquare;


    def createfitsquareLM(fitsquare) :
        def fitsquareLM(par0,par1,par2,par3,par4) :
            return fitsquare([par0,par1,par2,par3,par4]);
        return fitsquareLM;

    err = 0.01;
    errdef = 1;
    # pars: x0, y0, a, b, alpha
    init_pars  = [0., 0., 5000., 5000., 0. if initalpha_frac is None else np.pi*initalpha_frac ];
    limit_pars = [[-2000.,2000.], [-2000.,2000.], [0., 10000.], [0., 10000.], [-np.pi/2., np.pi/2.]];
    error_pars = [1.e-5,1.e-5,1.e-5,1.e-5,1.e-5];
    #fix_pars   = [True,True,False,True,True];
    #fix_pars   = [False,False,False,False,False];
    if fitCircle :
        fix_pars   = [False,False,False,True,True];
    else : # Ellipse fit
        fix_pars   = [False,False,False,False,False if initalpha_frac is None else True];
        pass;

    # Prepare data
    if usemK and not (np.all(np.array(reals_mK_org) == 0.)) :
        reals     = reals_mK_org;
        reals_err = reals_mK_err_org;
        imags     = imags_mK_org;
        imags_err = imags_mK_err_org;
    else :
        usemK = False;
        reals     = reals_org;
        reals_err = reals_err_org;
        imags     = imags_org;
        imags_err = imags_err_org;
        pass;

    x_fit       = copy.deepcopy(reals    );
    xerr_fit    = copy.deepcopy(reals_err);
    y_fit       = copy.deepcopy(imags    );
    yerr_fit    = copy.deepcopy(imags_err);
    # Check excludeAngle
    exclude_indices = [];
    wireangles_rad = []; # rad
    for k, angleData in enumerate(angleDataList) :
        # WARNING!! (take care about if it is rad. or deg.)
        angleDataList[k]['angle_deg'] = angleData['angle']; # deg
        angleDataList[k]['angle_rad'] = deg_to_rad(angleData['angle']); # rad
        if angleData['angle_deg'] in excludeAngle : # deg
            exclude_indices.append(k);
        else :
            wireangles_rad.append(deg_to_rad(angleData['angle_deg'])); # deg
            pass;
        pass;
    exclude_indices.sort(reverse=True); # sorted by reversed order
    # Exclude angles
    out.OUTVar(exclude_indices,-1);
    out.OUTVar(wireangles_rad, 0);
    for k in exclude_indices :
        del x_fit[k], xerr_fit[k], y_fit[k], yerr_fit[k];
        pass;
    out.OUT('real [x] (no cut)         = {}'.format(reals),-1);
    out.OUT('x_fit (after excludeAngle) = {}'.format(x_fit),-1);

    # convert to numpy array
    x_fit       = np.array(x_fit    );
    xerr_fit    = np.array(xerr_fit );
    y_fit       = np.array(y_fit    );
    yerr_fit    = np.array(yerr_fit );
    
    # calcualte rough estimate of the paramters
    x0_tmp = np.mean(x_fit);
    y0_tmp = np.mean(x_fit);
    i_0deg = 0;
    i_90deg= 0;
    for k, angleData in enumerate(angleDataList) :
        if angleData['angle_deg']== 0 : i_0deg = k; # deg
        if angleData['angle_deg']==90 : i_90deg= k; # deg
        pass;
    rtheta_tmp = calculateRTheta(reals[i_0deg]-reals[i_90deg], imags[i_0deg]-imags[i_90deg]);
    r_tmp  = rtheta_tmp[0]/2.;
    # set initial values
    init_pars[0] = x0_tmp if not fix_pars[0] else init_pars[0];
    init_pars[1] = y0_tmp if not fix_pars[1] else init_pars[1];
    init_pars[2] = r_tmp  if not fix_pars[2] else init_pars[2];
    init_pars[3] = r_tmp  if not fix_pars[3] else init_pars[3];
    init_pars[4] = init_pars[4] if not fix_pars[4] else init_pars[4];
    # set limits on parameters
    limit_pars[0][0] = x0_tmp - r_tmp;
    limit_pars[0][1] = x0_tmp + r_tmp;
    limit_pars[1][0] = y0_tmp - r_tmp;
    limit_pars[1][1] = y0_tmp + r_tmp;
    limit_pars[2][0] = r_tmp*0.0;
    limit_pars[2][1] = r_tmp*10.0;
    limit_pars[3][0] = r_tmp*0.0;
    limit_pars[3][1] = r_tmp*10.0;
    #limit_pars[4][0] = 0.;
    #limit_pars[4][1] = 1000.;

    # fit functions
    #fitsquare   = createfitsquare(fitfunc, x_fit, y_fit, xerr_fit, yerr_fit);
    #fitsquareLM = createfitsquareLM(fitsquare);
    # lmfit (to retrieve correct parameters)
    def resid(params, x, y, xerr=None, yerr=None):
        par0 = params['par0'].value
        par1 = params['par1'].value
        par2 = params['par2'].value
        par3 = params['par3'].value
        par4 = params['par4'].value
        if xerr is None and yerr is None :
            return np.sqrt(np.abs( fitfunc(x,y,[par0,par1,par2,par3,par4]) ));
        else :
            if xerr is None : xerr = np.zeros(len(x));
            if yerr is None : yerr = np.zeros(len(y));
            err2 = np.power(xerr,2.) + np.power(yerr,2.);
            return np.sqrt( np.abs(fitfunc(x,y,[par0,par1,par2,par3,par4])) / err2 );
            

    params = lmfit.Parameters()
    # LMfit: all free
    params.add('par0', init_pars[0], min=limit_pars[0][0], max=limit_pars[0][1], vary=not fix_pars[0])
    params.add('par1', init_pars[1], min=limit_pars[1][0], max=limit_pars[1][1], vary=not fix_pars[1])
    params.add('par2', init_pars[2], min=limit_pars[2][0], max=limit_pars[2][1], vary=not fix_pars[2])
    params.add('par3', init_pars[3], min=limit_pars[3][0], max=limit_pars[3][1], vary=not fix_pars[3])
    params.add('par4', init_pars[4], min=limit_pars[4][0], max=limit_pars[4][1], vary=not fix_pars[4])
    result = lmfit.minimize(resid, params, args=(x_fit, y_fit, xerr_fit, yerr_fit), method='leastsq')
    lmfit.report_fit(result)
    params = result.params;
    chisqr = result.chisqr;
    j=0;
    if not batchmode : plotFitResult(fitfunc, getParamsValues(params), chisqr, x_fit, y_fit, xerr_fit, yerr_fit, wireangles_rad, outpath, outsuffix='lmfit{}'.format(j), ext=ext, usemK=usemK);

    # LMfit: fixed b=a / alpha=0
    rtmp = params['par2'].value;
    params['par3'].set(expr='par2')
    params['par4'].set(value=init_pars[4], min=limit_pars[4][0], max=limit_pars[4][1], vary=False)
    result = lmfit.minimize(resid, params, args=(x_fit, y_fit, xerr_fit, yerr_fit), method='leastsq')
    lmfit.report_fit(result)
    params = result.params;
    chisqr = result.chisqr;
    j+=1;
    if not batchmode : plotFitResult(fitfunc, getParamsValues(params), chisqr, x_fit, y_fit, xerr_fit, yerr_fit, wireangles_rad, outpath, outsuffix='lmfit{}'.format(j), ext=ext, usemK=usemK);


    # repeat lmfit with all free parameters
    params['par3'].set(value=rtmp if not fix_pars[3] else init_pars[3], min=rtmp*0.9, max=rtmp*1.1, expr=None, vary=not fix_pars[3]);
    params['par4'].set(value=init_pars[4], min=limit_pars[4][0], max=limit_pars[4][1], expr=None, vary=not fix_pars[4]);
    for i in range(nLMfit) :
        result = lmfit.minimize(resid, params, args=(x_fit, y_fit, xerr_fit, yerr_fit), method='leastsq')
        out.OUT("### Fit using leastsq i={} ###".format(i),0)
        lmfit.report_fit(result)
        out.OUT('before params = {}'.format(params),0);
        params = result.params;
        out.OUT('after params = {}'.format(params),0);
        chisqr = result.chisqr;
        j+=1;
        if not batchmode : plotFitResult(fitfunc, getParamsValues(params), chisqr, x_fit, y_fit, xerr_fit, yerr_fit, wireangles_rad, outpath, outsuffix='lmfit{}'.format(j), ext=ext, usemK=usemK);

        #result2 = lmfit.minimize(resid, params, args=(x_fit, y_fit, xerr_fit, yerr_fit), method='differential_evolution')
        #out.OUT("\n# Fit using differential_evolution:",0)
        #lmfit.report_fit(result2)
        pass;

    # take over initial parameters from lmfit to minuit fit
    lmfit_result = result;
    for i, par in enumerate(lmfit_result.params.values()) :
        init_pars[i] = par.value;
        pass;
    out.OUT('init_pars for minuit = {}'.format(init_pars),0);

    # minuit fit (to calculate parameter errors)
    out.OUT('x_fit: {}'.format(x_fit),0);
    out.OUT('y_fit: {}'.format(y_fit),0);
    out.OUT('xerr_fit: {}'.format(xerr_fit),0);
    out.OUT('yerr_fit: {}'.format(yerr_fit),0);
    fitsquare   = createfitsquare(fitfunc, x_fit, y_fit, xerr_fit, yerr_fit, absame=fitCircle);
    result, minuit = minuitminosfit(fitsquare, init=init_pars, fix=fix_pars, limit=limit_pars, error=error_pars, errordef=errdef, precision=1.e-10, verbosity=2);
    out.OUT('result of minuit: {}'.format(result),0);
    pars = result[0];
    errs = result[3];
    chisqr = result[1];
    ndf = len(x_fit)-np.sum(~np.array(fix_pars));
    out.OUT('pars = {}'.format(*pars),0);
    out.OUT('chisquare calculated by myself = {}'.format( fitsquare(*pars) ));

    # Print fit results
    #parlabels = ['Center of real', 'Center of imag.', 'Pseudo-raidus^2', 'Coefficient on y^2 term', 'Coefficient on xy term'];
    parlabels = ['Center of real', 'Center of imag.', 'a (semi-major axis radius)', 'b (semi-minor axis radius)', 'Rotation of ellipse axis (alpha)'];
    printResult(result, out=out,
            parlabels = parlabels,
            );
    # Retrieve parameters
    if fitCircle :
        pars[3] = pars[2];
        errs[3] = errs[2];
        pass;
    r    = (pars[2]+pars[3])/2.;
    #r    = 1./2. * np.sqrt( (1.+pars[2]/pars[3]+2.*np.sqrt(pars[2]/pars[3]-4.*pars[4]**2.*pars[2]**2.) )/(1./pars[3]-4.*pars[4]**2.*pars[2]) );
    rerr = np.sqrt( errs[2]**2. + errs[3]**2. )/2.; # NOTE: Should be implemented
    a     = pars[2];
    aerr  = errs[2];
    b     = pars[3];
    berr  = errs[3];
    alpha    = pars[4]; # [rad.]
    #alpha     = 1./2. * np.arctan( pars[4]*pars[2] /(1.-pars[2]/pars[3]) ); # [rad.]
    alphaerr = errs[4]; # [rad.] # NOTE should be implemented
    x0    = pars[0];
    x0err = errs[0];
    y0    = pars[1];
    y0err = errs[1];
    (r0, r0_err), (theta0, theta0_err) = calculateRTheta(x0, y0, x0err, y0err);
    theta0_deg = rad_to_deg_0to2pi(theta0);
    theta0_err_deg = rad_to_deg_0to2pi(theta0_err);

    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('r', r, rerr),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('alpha [deg.]'+('(No meaning)' if fitCircle else ''), rad_to_deg_0to2pi(alpha), rad_to_deg_0to2pi(alphaerr)),0);
    out.OUT('(x0,y0) = ({:8.3f}, {:8.3f}) +- ({:5.3f}, {:5.3f})'.format(x0,y0,x0err,y0err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('r0 (center)', r0, r0_err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('theta0 (center) [rad.]', theta0, theta0_err),0);
    out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format('theta0 (center) [deg.]', rad_to_deg_0to2pi(theta0), rad_to_deg_0to2pi(theta0_err)),0);
    out.OUT('{:30s} = {:.2f}/{:d} = {:.2f}'.format(r'$\chi$/ndf', chisqr, ndf, chisqr/ndf),0);

    # Draw minuit contours
    """
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
    """
    del minuit;

    # Calculate & print each angle of each wire data
    fitRThetas = [];
    for k, angleData in enumerate(angleDataList) :
        x1    = reals[k];
        x1err = reals_err[k];
        y1    = imags[k];
        y1err = imags_err[k];
        #(r1, r1_err), (theta1, theta1_err) = calculateRTheta(x1-x0, y1-y0, np.sqrt(x1err**2.+x0err**2.), np.sqrt(y1err**2.+y0err**2.));
        (r1, r1_err), (theta1, theta1_err) = calculateRThetaEllipse(x1-x0, y1-y0, alpha, a, b, np.sqrt(x1err**2.+x0err**2.), np.sqrt(y1err**2.+y0err**2.), alphaerr, aerr, berr);
        out.OUT('k={} wire angle = {:.1f}, (x,y) = ({:.1f},{:.1f}), (r,theta) = ({:.1f},{:.1f} [deg.])'.format(k, angleData['angle_deg'], x1-x0,y1-y0,r1,rad_to_deg(theta1)),0);
        theta1_deg     = rad_to_deg_0to2pi(theta1);     # rad
        theta1_err_deg = rad_to_deg_0to2pi(theta1_err); # rad
        wireangle0     = theta0to2pi(2.*angleData['angle_rad'] + theta0to2pi(theta1))/2.; # rad
        wireangle0_err = theta1_err/2.; # rad
        theta_wire0    = theta0to2pi(theta1 + angleData['angle_rad']*2.); # rad
        theta_wire0_err= theta1_err; # rad
        fitRThetas.append({'wireangle_deg':angleData['angle_deg'], 'wireangle':angleData['angle_rad'], 'r':[r1,r1_err], 'theta':[theta1,theta1_err], 'theta_deg':[theta1_deg, theta1_err_deg], 'wireangle0':[wireangle0, wireangle0_err], 'theta_wire0':[theta_wire0, theta_wire0_err]});
        out.OUT('k={} wire angle = {:.1f}, wireangle_at_0 = {:.1f}, theta_of_wire0 = {:.1f}'.format(k, angleData['angle_deg'], wireangle0, theta_wire0),0);
        pass;
    # Calclate thetawire (wire angle) at theta=0 deg in the complex plane
    mean_wireangle0     = np.mean([ rtheta['wireangle0'][0] for rtheta in fitRThetas ]); # rad
    mean_wireangle0_err = np.mean([ rtheta['wireangle0'][1] for rtheta in fitRThetas ]); # rad
    mean_theta_wire0     = np.mean([ rtheta['theta_wire0'][0] for rtheta in fitRThetas ]); # rad
    mean_theta_wire0_err = np.mean([ rtheta['theta_wire0'][1] for rtheta in fitRThetas ]); # rad
    out.OUT('wire angle @ 0deg = {:.2f}+-{:.2f}, theta_of_wire0 = {:.2f}+-{:.2f}'.format(mean_wireangle0,mean_wireangle0_err,mean_theta_wire0,mean_theta_wire0_err),0);

    
    # Make funciton of the fitted circle
    xymax = np.max( [np.max(np.abs(reals)), np.max(np.abs(imags))] ); # max of |x| or |y|
    xylim = ((int)(xymax/limUnit)+1)*limUnit;
    delta = xylim*2*1.2/2000;
    xrange = np.arange(-xylim*1.2,xylim*1.2,delta);
    yrange = np.arange(-xylim*1.2,xylim*1.2,delta);
    X,Y = np.meshgrid(xrange, yrange);
    #Z = (X-pars[0])**2. + pars[2]/pars[3]*(Y-pars[1])**2. + pars[2]*pars[4]*(X-pars[0])*(Y-pars[1]) - pars[2];
    Z = fitfunc(X,Y,pars);

    ## Draw circle plot ##
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
    if fineCircle : figres.set_size_inches(6*4,9*4);
    else          : figres.set_size_inches(6,9);
    # Draw data points
    drawIndex = [];
    if drawExcludeAngle :
        drawIndex = np.arrange(len(reals));
    else :
        drawIndex = np.array([ i for i in range(len(reals)) if i not in exclude_indices]);
        pass;
    axsres[0].errorbar(
            np.array(reals)[drawIndex]          ,np.array(imags)[drawIndex],
            xerr=np.array(reals_err)[drawIndex] ,yerr=np.array(imags_err)[drawIndex],
            linestyle='',marker='o',markersize=2.,capsize=0.);
    shiftx=r*0.04;
    shifty=r*0.04;
    fitTheta0 = None;
    # Draw center
    axsres[0].errorbar([0.,x0],[0.,y0],xerr=[0.,x0err],yerr=[0.,y0err],linestyle='-',marker='o',markersize=2.,capsize=0.,color='tab:brown');
    # Cosmetic
    xlabel = r'Real [$mK_\mathrm{RJ}$]' if usemK else 'Real [ADC]';
    ylabel = r'Imag. [$mK_\mathrm{RJ}$]' if usemK else 'Imag. [ADC]';
    axsres[0].set_title(outname);
    axsres[0].set_xlabel(xlabel,fontsize=16);
    axsres[0].set_ylabel(ylabel,fontsize=16);
    axsres[0].set_xlim(-xylim,xylim);
    axsres[0].set_ylim(-xylim,xylim);
    if fineCircle : 
        axsres[0].tick_params(labelsize=6);
        axsres[0].set_xticks(np.arange(-xylim,xylim,20));
        axsres[0].set_yticks(np.arange(-xylim,xylim,20));
    else :
        axsres[0].tick_params(labelsize=12);
        pass;
    axsres[0].grid(True);

    # Draw circle
    axsres[0].contour(X,Y,Z,[0],colors='r',linewidths=1.0 if not fineCircle else 0.5);

    ## Draw angle diff. plot ##
    # Calculate the d_theta
    theta_wires  = []; # deg
    d_thetas     = []; # deg
    d_theta_errs = []; # deg
    # Get reference theta (fitTheta0)
    for k in range(len(reals)):
        if fitRThetas[k]['wireangle_deg'] == refAngle : fitTheta0 = fitRThetas[k]['theta_deg']; # deg
        pass;
    for k in range(len(reals))  :
        fitTheta = fitRThetas[k]['theta_deg']; # deg
        theta_wires.append(fitRThetas[k]['wireangle_deg']); # deg
        out.OUT('k={} wire angle = {:.1f}, fit angle = {:.1f}, fit angle (0deg) = {:.1f}'.format(k, theta_wires[-1], fitTheta[0], fitTheta0[0]),0);
        #d_fittheta = fitTheta0[0] - (fitTheta[0] if (fitTheta[0]<=fitTheta0[0]) else (fitTheta[0]-360.)); # deg
        d_fittheta = rad_to_deg_0to2pi(deg_to_rad(fitTheta0[0]) - deg_to_rad(fitTheta[0])); # deg
        out.OUT('k={} d_fittheta = {:.1f}'.format(k, d_fittheta),0);
        d_theta =  deg180to180(d_fittheta - (theta_wires[-1]-refAngle)*2.); # deg
        d_thetas.append(d_theta); # deg
        d_theta_errs.append(fitTheta[1]); # deg
        pass;
    theta_wires = np.array(theta_wires); 
    d_thetas    = np.array(d_thetas);
    # Draw
    axsres[1].errorbar(theta_wires,d_thetas,None,d_theta_errs,linestyle='-',marker='o',markersize=1.,capsize=2.,color='tab:blue');
    #axsres[1].errorbar(np.array(theta_wires)[drawIndex],np.array(d_thetas)[drawIndex],None,d_theta_errs,linestyle='-',marker='o',markersize=1.,capsize=2.,color='tab:blue');
    # Cosmetic
    xlabel = r'$\theta_{wire}$ [deg.]';
    #ylabel = r'$2\theta_{wire}$ - ' +'\n' +r'$\Delta (\theta_{fit}(\theta_{wire})$,$\theta_{fit}(0))$ [deg.]';
    ylabel = r'$\Delta\theta_{fit}(\theta_{wire},0)$ - $2\theta_{wire}$ [deg.]';
    # Draw 0 line
    axsres[1].plot([0.,180.],[0,0],linestyle='-',linewidth=0.5,marker='',color='k');
    axsres[1].set_xlabel(xlabel,fontsize=16);
    axsres[1].set_ylabel(ylabel,fontsize=16);
    axsres[1].set_xlim(refAngle,refAngle+180.);
    axsres[1].set_ylim(min(d_thetas[theta_wires>=refAngle]),max(d_thetas[theta_wires>=refAngle]));
    axsres[1].tick_params(labelsize=12);
    axsres[1].grid(True);

    ## Save figure without text
    saveFig(figres, '{}_fitresult_notext{}'.format(outpath, '' if not fineCircle else '_fineplot'), ext);

    ## Add text in circle plot ##
    # Write center info
    axsres[0].text(x0+shiftx, y0+0.5*shifty, 'Center: \n({:.2f}+-{:.2f} , {:.2f}+-{:.2f})'.format(x0,x0err,y0,y0err),color='tab:brown');
    axsres[0].text(shiftx, -4.*shifty, r'$r_0$={:.2f}+-{:.2f}'.format(r0,r0_err)+'\n'+r'$\theta_0=${:.2f}+-{:.2f} deg.'.format(theta0_deg,theta0_err_deg),fontsize=10,color='tab:brown');

    # Write circle info
    axsres[0].text(-xylim*0.95,-xylim*0.80, 'Radius: \n{:.2f}+-{:.2f}'.format(r,rerr),color='tab:red');
    axsres[0].text(-xylim*0.95,-xylim*0.90, r'$\chi/ndf$'+' = {:.2f}/{:d} = {:.2f}'.format(chisqr, ndf, chisqr/ndf),color='tab:red');

    # Write data point info
    for k in  drawIndex :
        fitTheta = fitRThetas[k]['theta_deg']; # deg
        
        x_tmp = reals[k];
        y_tmp = imags[k]+shifty;
        #if k==0 : x_tmp = reals[k]-0.*shiftx;
        #if k==0 : y_tmp = imags[k]+4.*shifty;
        #if k==1 : x_tmp = reals[k]+0.*shiftx   ;
        #if k==1 : y_tmp = imags[k]-5.*shifty;
        #if k==10 : x_tmp = reals[k]+0.*shiftx   ;
        #if k==10 : y_tmp = imags[k]-5.*shifty;
        #axsres[0].text(x_tmp, y_tmp, '{} deg.\n({:.1f} , {:.1f})\n'.format(angleDataList[k]['angle'], reals[k],imags[k])+r'$\theta$'+' = {:.2f} +- {:.2f} deg.'.format(fitTheta[0],fitTheta[1]), color='tab:blue');
        axsres[0].text(x_tmp, y_tmp, '{} deg.\n'.format(angleDataList[k]['angle_deg'])+r'$\theta$'+' = {:.2f} +- {:.2f} deg.'.format(fitTheta[0],fitTheta[1]), fontsize=10, color='tab:blue');
        pass;

    # Draw ideal radial lines at each 45 deg
    for k in range(8) :
        theta_ideal = deg_to_rad(-k*45. + fitTheta0[0]); # deg
        x_line = x0+r*np.cos(theta_ideal);
        y_line = y0+r*np.sin(theta_ideal);
        # Draw
        axsres[0].plot([x0,x_line],[y0,y_line],linestyle=':',linewidth=0.8,color='tab:red');
        pass;
     

    # Save plot
    out.OUT('Saving plot ({}_fitresult)...'.format(outpath),0);
    saveFig(figres, '{}_fitresult{}'.format(outpath, '' if not fineCircle else '_fineplot'), ext);

    # Save database
    columns = [
           ['id'            , ' INTEGER PRIMARY KEY AUTOINCREMENT', 0 ],
           #['boloname'      , 'NUM'    , boloname      ], # ver2
           ['readout_name'  , 'TEXT'    , boloname      ], # ver3
           ['lmfit_x0'      , 'REAL'    , init_pars[0]  ],
           ['lmfit_y0'      , 'REAL'    , init_pars[1]  ],
           ['lmfit_a'       , 'REAL'    , init_pars[2]  ],
           ['lmfit_b'       , 'REAL'    , init_pars[3]  ],
           ['lmfit_alpha'   , 'REAL'    , init_pars[4]  ],
           ['lmfit_chisqr'  , 'REAL'    , lmfit_result.chisqr  ],
           ['x0'            , 'REAL'    , x0         ],
           ['y0'            , 'REAL'    , y0         ],
           ['theta0'        , 'REAL'    , theta0     ],
           ['a'             , 'REAL'    , a          ],
           ['b'             , 'REAL'    , b          ],
           ['r'             , 'REAL'    , r          ],
           ['alpha'         , 'REAL'    , alpha      ],
           ['x0_err'        , 'REAL'    , x0err      ],
           ['y0_err'        , 'REAL'    , y0err      ],
           ['theta0_err'    , 'REAL'    , theta0_err ],
           ['a_err'         , 'REAL'    , aerr      ],
           ['b_err'         , 'REAL'    , berr      ],
           ['r_err'         , 'REAL'    , rerr       ],
           ['alpha_err'     , 'REAL'    , alphaerr   ],
           ['chisqr'        , 'REAL'    , chisqr     ],
           ['wireangle0'    , 'REAL'    , mean_wireangle0      ],
           ['wireangle0_err', 'REAL'    , mean_wireangle0_err  ],
           ['theta_wire0'   , 'REAL'    , mean_theta_wire0     ],
           ['theta_wire0_err','REAL'    , mean_theta_wire0_err ],
           ['theta_det'     , 'REAL'    , theta0to2pi(2.*np.pi-mean_theta_wire0)/2.    ], # ver3
           ['theta_det_err' , 'REAL'    , mean_theta_wire0_err/2.], # ver3
           ];
    # add info for each wire angles
    for k, angleData in enumerate(angleDataList) :
        deg  = '{}'.format(angleData['angle_deg']);
        x    = reals[k];
        xerr = reals_err[k];
        y    = imags[k];
        yerr = imags_err[k];
        columns.append(['[x_{}deg]'    .format(deg), 'REAL', x   ]);
        columns.append(['[x_err_{}deg]'.format(deg), 'REAL', xerr]);
        columns.append(['[y_{}deg]'    .format(deg), 'REAL', y   ]);
        columns.append(['[y_err_{}deg]'.format(deg), 'REAL', yerr]);
        out.OUT('Add columns of x/x_err/y/y_err_{}deg'.format(deg),0);
        pass;


    out.OUT('Saving database ({}.db)...'.format(outdbpath),0);
    tablename = 'wiregrid';
    if not os.path.isdir(outdbdir) : os.makedirs(outdbdir);
    conn  = sqlite3.connect(outdbpath+'.db', isolation_level=None);
    cursor= conn.cursor(); 
    cursor.execute('DROP TABLE if exists {}'.format(tablename));
    column_names = [];
    column_types = [];
    column_values= [];
    column_configs = [];
    for column in columns :
        column_names  .append(column[0]);
        column_types  .append(column[1]);
        # ver2
        #column_values .append('"{}"'.format(column[2]) if isinstance(column[2],str) \
        # ver3
        column_values .append('{}'.format(column[2]) if isinstance(column[2],str) \
                else ( '{:d}'.format(column[2]) if isinstance(column[2],int) else '{:e}'.format(column[2]) ) );
        column_configs.append('{} {}'.format(column[0],column[1]));
        pass;
    execute_string = \
        'CREATE TABLE {}('.format(tablename)\
        + ','.join(column_configs) +')';
    out.OUT('sqlite3 table setting :',0);
    out.OUT(' {}'.format(execute_string),0);
    cursor.execute(execute_string);
    execute_string = 'INSERT INTO {} VALUES ({})'.format( tablename, ','.join(['?' for c in column_names]))
    out.OUTVar(execute_string,0);
    out.OUTVar(column_values,0);
    cursor.executemany(execute_string, [column_values] );
    conn.close();

    out.OUT('Saving database into a pickle file ({}.pkl)...'.format(outdbpath),0);
    outfile = open('{}.pkl'.format(outdbpath), 'wb');
    pickle.dump(columns,outfile);
    outfile.close();

    # delete objects
    pklfile.close();
    del pklfile;
    del angleDataList
    del reals_org     
    del reals_err_org 
    del imags_org     
    del imags_err_org 
    del mags_org      
    del mags_err_org  
    del reals_mK_org     
    del reals_mK_err_org 
    del imags_mK_org     
    del imags_mK_err_org 
    del mags_mK_org      
    del mags_mK_err_org  
    del fitfunc
    del createfitsquare
    del createfitsquareLM
    del x_fit, xerr_fit, y_fit, yerr_fit;
    del rtheta_tmp;
    del resid;
    del params;
    del result;
    del lmfit_result;
    del fitsquare;
    del fitRThetas;
    del xrange, yrange, X, Y, Z;
    del figres, axsres 
    del theta_wires, d_thetas, d_theta_errs;
    del columns;
    del cursor, conn, outfile;

    gc.collect();

    return 0;

 


if __name__=='__main__' :

    boloname = 'PB20.13.13_Comb01Ch01';
    #boloname = 'PB20.13.13_Comb01Ch02';
    picklename = None;
    pickledir  = 'plot_ver1'
    pickleprefix = 'ver1_';
    picklesuffix = '_demodresult';
    outdir      = 'output_ver1';
    outprefix = 'ver1_';
    outsuffix = '';
    outname   = None;
    ext         = 'png';


    initalpha_frac = None;
    excludeAngle= [] ;
    refAngle    = 0.; # deg

    batchmode = False;
    verbosity = 1;
    
    parser = argparse.ArgumentParser();
    parser.add_argument('-b', '--boloname', default=boloname, help='boloname (default: {})'.format(boloname));
    parser.add_argument('-p', '--pickledir', default=pickledir, help='write&read directory for pickle files to save the data retreived from g3 file (default: {})'.format(pickledir));
    parser.add_argument('--picklename', default=picklename, help='pickle filename (default: {})'.format(picklename));
    parser.add_argument('--pickleprefix', default=pickleprefix, help='pickle filename prefix: This will be ignored if picklename is specified. (default: {})'.format(pickleprefix));
    parser.add_argument('--picklesuffix', default=picklesuffix, help='pickle filename suffix: This will be ignored if picklename is specified. (default: {})'.format(picklesuffix));
    parser.add_argument('-d', '--outdir', default=outdir, help='output directory for the all (default: {})'.format(outdir));
    parser.add_argument('-o', '--outname', default=outname, help='output filename: This will be overwrite outprefix/outsuffix (default: {})'.format(outname));
    parser.add_argument('--outprefix', default=outprefix, help='output filename prefix (default: {})'.format(outprefix));
    parser.add_argument('--outsuffix', default=outsuffix, help='output filename suffix (default: {})'.format(outsuffix));
    parser.add_argument('-e', '--ext', default=ext, help='Output file extensions for figure: You can set multiple extensions by "," e.g. "pdf,png". (default: {})'.format(ext));
    parser.add_argument('--notbatch', dest='batchmode', action='store_false', default=True, help='Unset batchmode(Less output for batch job)');
    parser.add_argument('--init-alpha', dest='initalpha_frac', default=initalpha_frac, help='Fixed value of alpha(alpha will be fixed in the fit)');
    parser.add_argument('--refAngle', default=refAngle, help='Reference wire angle for the angle calibration');
    parser.add_argument('--excludeAngle', default=None, help='Exclued wire angles in the fit. Multiple angles can be specified by joining \",\". (ex. 0,45) default=\"\"');
    parser.add_argument('--limUnit', default=2500, type=int, help='x/y plot range minimum unit (default 2500)');
    parser.add_argument('--fineCircle', default=False, action='store_true', help='Make a fine circle plot (default=False)');
    parser.add_argument('--fitCircle', default=False, action='store_true', help='Fit with a circle or ellipse (default=False)');
    parser.add_argument('-v', '--verbosity', default=verbosity, type=int, help='verbosity level: A larger number means more printings. (default: {})'.format(verbosity));

    args = parser.parse_args();

    # List arguments
    bolonames = args.boloname.split(',');
    Nbolo     = len(bolonames);
    pickledirs = args.pickledir.split(',');
    if args.picklename is None :
        pickleprefixs= args.pickleprefix.split(',');
        picklesuffixs= args.picklesuffix.split(',');
        Ntmp = len(pickleprefixs);
        if Ntmp<Nbolo: pickleprefixs.extend([pickleprefixs[-1]]*(Nbolo-Ntmp));
        Ntmp = len(picklesuffixs);
        if Ntmp<Nbolo: picklesuffixs.extend([picklesuffixs[-1]]*(Nbolo-Ntmp));
        picklenames = [];
        for n, boloname in enumerate(bolonames) :
            picklenames.append('{}{}{}.pkl'.format(pickleprefixs[n], boloname, picklesuffixs[n]));
            pass;
    else :
        picklenames= args.picklename.split(',');
        pass;
    outdirs   = args.outdir.split(',');
    if args.outname is None :
        outprefixs= args.outprefix.split(',');
        outsuffixs= args.outsuffix.split(',');
        Ntmp = len(outprefixs);
        if Ntmp<Nbolo: outprefixs.extend([outprefixs[-1]]*(Nbolo-Ntmp));
        Ntmp = len(outsuffixs);
        if Ntmp<Nbolo: outsuffixs.extend([outsuffixs[-1]]*(Nbolo-Ntmp));
        outnames = [];
        for n, boloname in enumerate(bolonames) :
            outnames.append('{}{}{}'.format(outprefixs[n], boloname, outsuffixs[n]));
            pass;
    else :
        outnames = args.outname.split(',');
        pass;
    initalpha_fracs = args.initalpha_frac;
    initalpha_fracs = [None] if args.initalpha_frac is None else [ (float)(tmp) for tmp in initalpha_frac.split(',')] ;

    # Single-value arguments
    ext      = args.ext;
    refAngle = args.refAngle;
    excludeAngle = [] if args.excludeAngle is None else [ float(angle) for angle in args.excludeAngle.split(',') ];

    batchmode= args.batchmode;
    verbosity= args.verbosity;

    # Check size of list arguments
    Ntmp = len(pickledirs);
    if Ntmp<Nbolo: pickledirs.extend([pickledirs[-1]]*(Nbolo-Ntmp));
    Ntmp = len(picklenames);
    if Ntmp<Nbolo: picklenames.extend([picklenames[-1]]*(Nbolo-Ntmp));
    Ntmp = len(outdirs);
    if Ntmp<Nbolo: outdirs.extend([outdirs[-1]]*(Nbolo-Ntmp));
    Ntmp = len(outnames);
    if Ntmp<Nbolo: outnames.extend([outnames[-1]]*(Nbolo-Ntmp));
    Ntmp = len(initalpha_fracs);
    if Ntmp<Nbolo: initalpha_fracs.extend([initalpha_fracs[-1]]*(Nbolo-Ntmp));


    # Loop over bolonames
    for n, boloname in enumerate(bolonames) :
        try : 
            main(pickledirs[n]+'/'+picklenames[n], boloname=boloname, usemK=True, refAngle=refAngle, outdir=outdirs[n], outname=outnames[n], excludeAngle=excludeAngle, ext=ext, verbosity=verbosity, initalpha_frac=initalpha_fracs[n], fineCircle=args.fineCircle, batchmode=batchmode, fitCircle=args.fitCircle, limUnit=args.limUnit);
        except Exception as e:
            print('####################################################################################');
            print('ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!');
            print(' ####Error#### for {}: {}'.format(boloname, e));
            print('ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!ERROR!');
            print('####################################################################################');
            print(traceback.format_exc())
            pass;
        pass;
    pass;

