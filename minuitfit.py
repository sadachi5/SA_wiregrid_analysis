from iminuit import Minuit;
import Out;
import numpy as np;
import time;
import matplotlib.pyplot as plt;
from utils import colors;

def drawcontour(
    minuit,
    fix,
    parIDs=[1,2],
    #cls=[0.01,0.1,0.6827,0.9545],
    cls=[0.680],
    numpoints=100,
    center=None,
    outname='aho', 
    out = None,
    verbosity = 0,
    ) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;

    if fix[parIDs[0]] or fix[parIDs[1]] :
        out.WARNING('The contour parameters are fixed. Could not draw contours! --> Skip!!');
        return -1;
    
    contours = [];
    parname1 = 'x{}'.format(parIDs[0]);
    parname2 = 'x{}'.format(parIDs[1]);
    for cl in cls :
        points = minuit.mncontour(parname1,parname2,size=numpoints,cl=cl);
        out.OUT('contour points = {}'.format(points),-2);
        x=[];
        y=[];
        for point in points :
            x.append(point[0]);
            y.append(point[1]);
            pass;
        contours.append([x,y]);
        pass;

    for i, cl in enumerate(cls) :
        plt.plot(contours[i][0],contours[i][1],color=colors[i],label='$CL$ = {}'.format(cl));
        pass;
    if center!=None :
        plt.scatter([center[0]],[center[1]],marker='*',label='center',color='k');
        plt.text(center[0],center[1],'(x,y)=({:.4},{:.3})'.format(center[0],center[1]));
        pass;
    plt.legend();
    plt.grid();
    plt.savefig('{}.png'.format(outname));
    plt.close();
    return 0;

def drawcontour2(
    minuit,
    fix,
    parIDs=[1,2],
    parLabels=None,
    numpoints=100,
    center=None,
    dxdy=None, #[dx,dy] (needs "center")
    outname='aho', 
    out = None,
    verbosity = 0,
    ) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;

    if fix[parIDs[0]] or fix[parIDs[1]] :
        out.WARNING('The contour parameters are fixed. Could not draw contours! --> Skip!!');
        return -1;
    
    contours = [];
    parname1 = 'x{}'.format(parIDs[0]);
    parname2 = 'x{}'.format(parIDs[1]);
    bound = 2.;
    if (not dxdy is None) and (not center is None) :
        bound = [[center[0]-dxdy[0],center[0]+dxdy[0]], 
                 [center[1]-dxdy[1],center[1]+dxdy[1]]];
        pass;
    x, y, fval = minuit.contour(parname1,parname2,size=numpoints,bound=bound);

    cont = plt.contour(x,y,fval);
    cont.clabel(fmt='%1.1f', fontsize=12);
    if not center is None :
        plt.scatter([center[0]],[center[1]],marker='*',label='center',color='k');
        plt.text(center[0],center[1],'(x,y)=({:.4},{:.3})'.format(center[0],center[1]));
        pass;
    xlabel = parname1 if parLabels is None else parLabels[0];
    ylabel = parname2 if parLabels is None else parLabels[1];
    plt.xlabel(xlabel,fontsize=16);
    plt.ylabel(ylabel,fontsize=16);
    plt.grid();
    plt.savefig('{}.png'.format(outname));
    plt.close();
    return 0;




def minuitminosfit(
    fitfunc,
    init,
    fix,
    limit,
    error,
    errordef = 1,
    precision=1.e-5,
    out=None,
    verbosity=0
    ) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;
 
    nPar = len(init);
    minuit = Minuit(
        fitfunc, 
        *init);
    out.OUT('minuit :\n{}'.format(minuit),-1);
    names = minuit.parameters;
    minuit.errordef = Minuit.LEAST_SQUARES;
    for i in range(nPar) :
        minuit.fixed[i]=fix[i];
        minuit.limits[i]=limit[i];
        pass;
    
    minuit.migrad();
    minuit.hesse();
    minuit.minos();
    values = minuit.values;
    errs = minuit.errors;
    fmin = minuit.fval;
    nfcn = minuit.nfcn; 
    out.OUT('values = {}'.format(values), -2);
    out.OUT('errs = {}'.format(errs), -2);
    out.OUT('fmin = {}'.format(fmin), -2);
    out.OUT('nfcn = {}'.format(nfcn), -2);
    out.OUT('covariance :\n{}'.format(minuit.covariance), -1);
    out.OUT('is_valid   = {}'.format(minuit.valid   ), -1);
    out.OUT('is_accurate= {}'.format(minuit.accurate), -1);
    out.OUT('nfcn = {}'.format(nfcn), -2);
 
    for i in range(nPar) :
        out.OUT('par{} = {}'.format(i, values[i]), -2);
        pass;
 
    result = [values, fmin, nfcn, errs];
 
    #del fitfunc, nPar, names, fmin, minuit;
    del fitfunc, nPar, names, fmin;
    return result, minuit;


def createfitfunc(func, x, y, err=None) : 

    def fitfunc(pars) : 
        return func(pars,x,y,err);
 
    return fitfunc;


def createfitsquare(x, y, func, nPar, err=None, errx=None, n_fix_pars=0, out=None, verbosity=0) :

    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;
 
    # initialize error
    #print(err);
    if err == None and len(err)==1 :
        out.WARNING('err = None --> err  = [1,1,...]');
        err = [ 1 for i in range(len(x)) ];
        pass;
 
    def fitsquare(*pars) :
        #print('x',x);
        #print('len(x)',len(x));
        #print('y',y);
        #print('len(y)',len(y));
        diff = y - func(pars, x, y);
        errsquare = err*err if errx==None else err*err + errx*errx ;
        square = np.sum( np.power(diff,2.) / errsquare );
        #print(square);
        return square;
        ## !NOTE!: You should consider which calculation is correct!
        #nof  = len(x) - len(pars)-n_fix_pars-1;
        #return np.sum( np.power( diff / err, 2.) )/nof;
 
    return fitsquare;


def truncateX(x, data, xlim=[None, None], err=None) :
    xmin = min(x);
    xmax = max(x);
    if xlim[0]==None : xlim[0] = xmin;
    if xlim[1]==None : xlim[1] = xmax;
    x_truncate    = [];
    data_truncate = [];
    err_truncate  = [];
    for i in range(len(x)):
        if x[i] < xlim[0] or x[i] > xlim[1] : continue;
        x_truncate   .append(x   [i]);
        data_truncate.append(data[i]);
        if len(err)>i : err_truncate .append(err [i]);
        pass;
    return np.array(x_truncate), np.array(data_truncate), np.array(err_truncate);

def printResult(result, parlabels=None, out=None,  verbosity=0) :
    # initialize Out
    if out==None : out = Out.Out(verbosity=verbosity);
    else         : out = out;

    #result = [values, fmin, nfcn, errs];
    out.OUT('fmin (minimized value) = {}'.format(result[1]),0);
    out.OUT('nfcn (# of calls)      = {}'.format(result[2]),0);
    errs    = result[3];
    for n in range(len(result[0])) :
        if parlabels is None : parlabel = 'par[{}]'.format(n);
        else                 : parlabel = parlabels[n];
        out.OUT('{:30s} = {:8.3f} +- {:5.3f}'.format(parlabel, result[0][n],errs[n]),0);
        pass;
    return 0;

if __name__=='__main__' :
    t = np.linspace(0.,200.,2000);
    theta = t/100.;
    err = 0.01;
    errdef = 1;
    y = np.sin(2.*np.pi*theta) + 0.2 + np.random.randn()*err;
    #y = np.sin(2.*np.pi*theta) + 0.2;
    y_err = np.full(len(t), err);

    def fitfunc(pars,x,y) :
        return np.multiply(np.sin(np.multiply(x, 2.*np.pi*pars[1]) + pars[2]),  pars[0])  + pars[3];

    init_pars  = [1., 0.01, 0., 0.];
    limit_pars = [[-10.,10.], [-10.,10.], [-np.pi, np.pi], [-10.,10.]];
    error_pars = [1.e-5,1.e-5,1.e-5,1.e-5];
    #fix_pars   = [False,True,True,True];
    fix_pars   = [False,False,False,False];

    t_truncate, y_truncate, err_truncate = truncateX(t, y, [50., 150.], y_err);

    fitsquare = createfitsquare(t_truncate, y_truncate, fitfunc, len(init_pars), err_truncate);
    result    = minuitminosfit(fitsquare, init=init_pars, fix=fix_pars, limit=limit_pars, error=error_pars, errordef=errdef, precision=1.e-10, verbosity=2);
    errs      = result[3];

    printResult(result);

    t_fitrange = np.linspace(50., 150., 1000);
    fitted_y   = fitfunc(result[0], t_fitrange, None);

    plt.errorbar(t, y, y_err);
    plt.plot(t_fitrange, fitted_y, color='r');
    plt.savefig('aho.png');

    pass;
