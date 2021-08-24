
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan','tab:gray','red','royalblue','turquoise','darkolivegreen', 'magenta', 'blue', 'green']*5;

## plottmp() ##
import os;
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
def plottmp(x,y,xlabel='x',ylabel='y',i=0,outname='aho',xlim=None, xtime=False, ny=1) :
    tmpdir = 'tmp';
    if not os.path.isdir(tmpdir): os.mkdir(tmpdir);
    if ny>1: ys = y;
    else   : ys = [y];
    plt.title(outname);
    for n, __y in enumerate(ys) :
        plt.plot(x, __y, linestyle='', marker='o', markersize=1., color=colors[n]);
        pass;
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

## plottmphist() ##
import os;
import numpy as np
import copy
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
def plottmphist(x0,nbins=100,y=None,xlabel='x',ylabel='Counts',i=0,outname='aho',xrange=None,yrange=None,xlim=None, ylim=None, xtime=False,show=False,log=False,drawflow=True, stacked=False, nHist=1, label='', outdir = 'tmp') :
    if not os.path.isdir(outdir): os.mkdir(outdir);
    plt.title('');
    x = copy.deepcopy(x0);
    if nHist==1 : print('x min = {} / max = {}'.format(min(x), max(x)));
    if drawflow :
        if not xrange is None :
            if nHist == 1:
                i_overflow  = (x>xrange[1]);
                i_underflow = (x<xrange[0]);
                x[i_overflow]  = xrange[1];
                x[i_underflow] = xrange[0];
            else :
                for n in range(nHist) :
                    i_overflow  = (x[n]>xrange[1]);
                    i_underflow = (x[n]<xrange[0]);
                    x[n][i_overflow]  = xrange[1];
                    x[n][i_underflow] = xrange[0];
                pass;
            pass;
        pass;
    if not y is None : print('y min = {} / max = {}'.format(min(y), max(y)));
    if y is None : 
        hist = plt.hist  (x,    bins=nbins, range=xrange, histtype='stepfilled', linestyle='-', linewidth=0.5, edgecolor='k', stacked=stacked, alpha=0.5, color = colors[1:1+nHist], label=label, align='mid', orientation='vertical', log=log);
    else         : 
        ranges = None if xrange is None or yrange is None else [xrange,yrange];
        hist = plt.hist2d(x, y, range=ranges,bins=nbins, norm=matplotlib.colors.LogNorm() if log else matplotlib.colors.Normalize());

    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.tight_layout(rect=[0,0,1,0.96])
    if xtime :
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.tick_params(axis='x',labelrotation=40,labelsize=8);
        pass;
    plt.grid();
    if label!='' : plt.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 8,title='',borderaxespad=0.);
    if not xlim==None : plt.xlim(xlim);
    if not ylim==None : plt.ylim(ylim);

    if not y is None :
        plt.colorbar(hist[3])
        pass;
    if show : plt.show();
    outname = '{}/{}{}.pdf'.format(outdir,outname,i);
    print('save {}'.format(outname));
    plt.savefig(outname);
    plt.close();
    return;
## end of plottmphist() ##



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


## getPandasPickle() ##
# filename : pickle filename (*.pkl)
def getPandasPickle(filename) :
    import pandas as pd;
    df=None;

    hostname = os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ.keys() else '';
    if hostname.endswith('kek.jp') :
        df = pd.read_pickle(filename);
    else :
        import pickle5;
        with open(filename, 'rb') as f :
            df = pickle5.load(f);
            pass;
        pass;
 
    return df;
## end of getPandaPickle() ##

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
# theta range --> [0, pi] or [upper-pi, upper] 
def theta0topi(theta,upper=np.pi) :
    theta2 = np.multiply(theta, 2.);
    theta  = np.multiply(theta0to2pi(theta2),0.5);
    theta  = np.where(theta>=upper, theta-np.pi, theta);
    return theta;


# radian to degrees
def rad_to_deg(rad) : return np.multiply(rad, 180./(np.pi)) ; # [deg.]
# --> [0, 360] deg
def rad_to_deg_0to2pi(rad) : return np.multiply( theta0to2pi(rad), 180./(np.pi) ); # [deg.]
# --> [-180, 180] deg
def rad_to_deg_pitopi(rad) : return np.multiply( thetapitopi(rad), 180./(np.pi) ); # [deg.]

# degree to radian
def deg_to_rad(deg) : return np.multiply(deg, np.pi/180.);

# degree range --> [0, 180] or [upper-pi, upper]
def deg0to180(deg,upper=180.) :
    theta = deg_to_rad(deg);
    upper_rad = deg_to_rad(upper);
    theta = theta0topi(theta, upper_rad); # [upper-pi, upper]
    deg_new = rad_to_deg(theta);
    return deg_new;

# degree range --> [-90, 90]
def deg90to90(deg) :
    deg_new = deg0to180(deg, 90.);
    return deg_new;


# diff. between two angles (rad.) [0, pi/2] or [0, pi]
def diff_angle(rad1,rad2,upper90deg=True) :
    diff = np.arccos(np.cos(rad1-rad2));
    if upper90deg : 
        diff = theta0topi(diff);
        diff = np.where(diff>=np.pi/2., np.pi-diff, diff);
    return diff;

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
    #theta = math.atan2(y,x); # atan2 returns [-pi,pi]
    theta = np.arctan2(y,x); # atan2 returns [-pi,pi]
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
