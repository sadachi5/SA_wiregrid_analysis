
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan','tab:gray','red','royalblue','turquoise','darkolivegreen', 'magenta', 'blue', 'green'];

## plottmp() ##
import os;
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
def plottmp(x,y,xlabel='x',ylabel='y',i=0,outname='aho',xlim=None, xtime=False) :
    tmpdir = 'tmp';
    if not os.path.isdir(tmpdir): os.mkdir(tmpdir);
    plt.title(outname);
    plt.plot(x, y, linestyle='', marker='o', markersize=1.);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.tight_layout(rect=[0,0,1,0.96])
    if xtime :
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.tick_params(axis='x',labelrotation=40,labelsize=8);
        pass;
    plt.grid();
    if not xlim==None : plt.xlim(xlim);
    outname = '{}/{}{}.pdf'.format(tmpdir,outname,i);
    print('save {}'.format(outname));
    plt.savefig(outname);
    plt.close();
    return;
## end of plottmp() ##


## get_var_name() ##
import inspect;
def get_var_name(var, back_vars=None):
    name = '';
    if back_vars==None : 
        back_vars = inspect.currentframe().f_back.f_globals;
        back_vars.update(inspect.currentframe().f_back.f_globals);
        pass;
    for k,v in back_vars.items():
        if id(v) == id(var):
            name=k;
            pass;
        pass;
    return name;
## end of get_var_name() ##

## printVar() ##
def printVar(var) :
    back_vars = inspect.currentframe().f_back.f_globals;
    back_vars.update(inspect.currentframe().f_back.f_locals);
    #print(back_vars.keys());
    name = get_var_name(var,back_vars); 
    print('{:10s} = {}'.format(name,var));
    return 0;
## end of printVar() ##


## saveFig() ##
def saveFig(fig, outpath, ext) :
    outdir = '/'.join(outpath.split('/')[:-1] );
    if not os.path.isdir(outdir) :
        os.makedirs(outdir);
        pass;
    for exttmp in ext.split(',') :
        fig.savefig('{}.{}'.format(outpath,exttmp));
        pass;
    return 0;
## end of saveFig() ##


# MJD to second
def mjd_to_second(mjd) : return mjd * 86400;
# second to MJD
def second_to_mjd(second) : return second * (1./86400.);

# theta range --> [-pi, pi]
def thetapitopi(theta) :
    return np.arctan2(np.sin(theta),np.cos(theta));
# theta range --> [0, 2pi]
def theta0to2pi(theta) :
    return np.arctan2(-np.sin(theta),-np.cos(theta))+np.pi;

# radian to degrees
def rad_to_deg(rad) : return np.multiply(rad, 180./(np.pi)) ; # [deg.]
# --> [0, 360] deg
def rad_to_deg_0to2pi(rad) : return np.multiply( theta0to2pi(rad), 180./(np.pi) ); # [deg.]
# --> [-180, 180] deg
def rad_to_deg_pitopi(rad) : return np.multiply( thetapitopi(rad), 180./(np.pi) ); # [deg.]

# degree to radian
def deg_to_rad(deg) : return np.multiply(deg, np.pi/180.);

# Create new array of x and y between xmin & xmax  (xmin<=x<=xmax)
# x and y have the same size
def between(x,y,xmin,xmax) :
    if len(x) != len(y) :
        print("utils:between() ERROR!! x and y don't have the same size.");
        return -1;
    xnew = [];
    ynew = [];
    for i,x0 in enumerate(x) :
        if x0>=xmin and x0<=xmax :
            xnew.append(x0);
            ynew.append(y[i]);
            pass;
        pass;
    return np.array(xnew),np.array(ynew);

# Calculate (r,theta) of (x,y) and their error in a circle
import math;
def calculateRTheta(x,y,xerr=None,yerr=None) :
    r     = np.sqrt( np.power(x,2.) + np.power(y,2.) );
    theta = math.atan2(y,x); # atan2 returns [-pi,pi]
    if xerr is None and yerr is None :
        return  [r, theta];
    else :
        if xerr is None : xerr = 0.;
        if yerr is None : yerr = 0.;
        dr_dx = x/r;
        dr_dy = y/r;
        rerr  = np.sqrt( np.power(dr_dx*xerr,2.) + np.power(dr_dy*yerr,2.) );
        dtheta_dx = -y/(x*x + y*y);
        dtheta_dy = x/(x*x + y*y);
        thetaerr  = np.sqrt( 
                np.power(dtheta_dx*xerr, 2.) +
                np.power(dtheta_dy*yerr, 2.) );
        pass;
    return [[r, rerr], [theta, thetaerr]];
# Calculate (r,theta) of (x,y) and their error in an ellipse
import math;
def calculateRThetaEllipse(x,y,alpha,a,b,xerr=None,yerr=None,alphaerr=None,aerr=None,berr=None) :
    r     = np.sqrt( np.power(x,2.) + np.power(y,2.) );
    X     =  x*np.cos(alpha) + y * np.sin(alpha);
    Y     = -x*np.sin(alpha) + y * np.cos(alpha);
    theta_XY = math.atan2(a*Y, b*X); # atan2 returns [-pi,pi]
    theta    = theta_XY + alpha;
    if xerr is None and \
       yerr is None and \
       alphaerr is None and \
       aerr is None and \
       berr is None:
        return  [r, theta];
    else :
        if xerr is None : xerr = 0.;
        if yerr is None : yerr = 0.;
        if alphaerr is None : alphaerr = 0.;
        if aerr is None : aerr = 0.;
        if berr is None : berr = 0.;
        dr_dx = x/r;
        dr_dy = y/r;
        rerr  = np.sqrt( np.power(dr_dx*xerr,2.) + np.power(dr_dy*yerr,2.) );

        X2  = np.power(X, 2.);
        Y2  = np.power(Y, 2.);
        #dX2 = np.power(np.cos(alpha)*xerr,2.) + np.power(np.sin(alpha)*yerr,2.) + np.power(Y*alphaerr,2.);
        #dY2 = np.power(np.sin(alpha)*xerr,2.) + np.power(np.cos(alpha)*yerr,2.) + np.power(X*alphaerr,2.);
        dX2 = np.power(np.cos(alpha)*xerr,2.) + np.power(np.sin(alpha)*yerr,2.);
        dY2 = np.power(np.sin(alpha)*xerr,2.) + np.power(np.cos(alpha)*yerr,2.);
        D   = a/b * Y/X;
        a2  = np.power(a, 2.);
        b2  = np.power(b, 2.);
        D2  = np.power(D, 2.);
        dThetaSubAlpha2 = D2/np.power(1.+D2, 2.) * ( np.power(aerr,2.)/a2 + np.power(berr,2.)/b2 + dX2/X2 + dY2/Y2 );
        dtheta_alpha = 1. - 1./(1.+D2) * a/b * np.power(r,2.) / np.power(x*np.cos(alpha) + y*np.sin(alpha), 2.);
        #print('dtheta_alpha = {}'.format(dtheta_alpha));
        dtheta_from_alpha2 = np.power( dtheta_alpha * alphaerr, 2. );
        #dTheta2         = np.power(alphaerr,2.) + dThetaSubAlpha2;
        dTheta2         = dThetaSubAlpha2 + dtheta_from_alpha2;
        #dTheta2         = dThetaSubAlpha2;
        thetaerr  = np.sqrt(dTheta2);
        pass;
    return [[r, rerr], [theta, thetaerr]];


# rms
def rms(x) :
    return np.sqrt(np.sum(np.power(x,2.))/(float)(len(x)));


def getwafername(boloname) :
    return boloname.split('_')[0];
