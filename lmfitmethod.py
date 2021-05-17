import Out;
import numpy as np;
import lmfit;

class LMfit :

    def __init__(self, fitfunc, init, fix, limit, out=None, verbosity=0) :

        # initialize Out
        if out is None : self.out = Out.Out(verbosity=verbosity);
        else           : self.out = out;

        self.fitfunc = fitfunc;
        self.nPar    = len(init);
        self.init    = init;
        self.fix     = fix;
        self.limit   = limit;

        self.model   = lmfit.Model(fitfunc);

        self.pars    = self.model.make_params();
        self.out.OUT('pars before setting = {}'.format(self.pars), -1);
        for i, name in enumerate(self.model.param_names) :
            self.out.OUT('{} : pars[{}] = {} ({}--{}) Fixed={}'.format(i, name,self.init[i], self.limit[i][0], self.limit[i][1], self.fix[i]),0), 
            self.pars[name].set(
                    value = self.init[i],
                    min   = self.limit[i][0],
                    max   = self.limit[i][1],
                    vary  = not(self.fix[i])
                    );
            pass;
        self.result = None;
        self.out.OUT('pars after  setting = {}'.format(self.pars), -1);

        pass;

    def dofit(self, x, y, yerr) :
        self.result = self.model.fit(x=x, data=y, weights=yerr**(-1.0), params=self.pars, method='leastsq');
        self.out.OUT(self.result.fit_report(),-1);
        #self.result = self.model.fit(x=x, data=y, weights=yerr**(-1.0), params=self.result.params, method='leastsq');
        #self.out.OUT(self.result.fit_report(),-1);
        self.result.fit();
        self.out.OUT(self.result.fit_report(),-1);
        for name, par in self.result.params.items() :
            value = par.value;
            if name=='par0' : self.result.params[name].set( value = value*0.4);
            pass;
        self.result.fit();
        self.out.OUT(self.result.fit_report(),-1);
        return self.result;

    def printResult(self) :
        self.out.OUT('## Fit Results ##',0);
        self.out.OUT(self.result.fit_report(),0);
        self.out.OUT('## Confidence Intervals ##',0);
        self.out.OUT(self.result.ci_report(),0);
        return 0;

    def plotFitResult(self) :
        if self.result is None : 
            out.WARNING('The fit have not been done yet.');
            return -1;
        fig, gridspec = self.result.plot(data_kws={'markersize':5});
        return fig, gridspec;


    pass;



if __name__=='__main__' :
    from minuitfit import truncateX, createfitsquare;
    t = np.linspace(0.,100.,1000);
    theta = t/100.;
    err = 0.01;
    y = 10.*np.sin(2.*np.pi*theta) + 0.2 + np.random.randn()*err;
    #y = np.sin(2.*np.pi*theta) + 0.2;
    y_err = np.full(len(t), err);

    def fitfunc(x,par0,par1,par2,par3) :
        return np.multiply(np.sin(np.multiply(x, 2.*np.pi*par1) + par2),  par0)  + par3;

    init_pars  = [0., 0.01, 0., 0.];
    limit_pars = [[0.,100.], [-10.,10.], [-np.pi, np.pi], [-10.,10.]];
    #fix_pars   = [False,True,True,True];
    fix_pars   = [False,False,False,False];

    t_truncate, y_truncate, err_truncate = truncateX(t, y, [0., 100.], y_err);

    print(' y    (size={}) = {}'.format(len(y_truncate), y_truncate));
    print(' yerr (size={}) = {}'.format(len(err_truncate), err_truncate));
    fit    = LMfit(fitfunc, init=init_pars, fix=fix_pars, limit=limit_pars, verbosity=2);
    result = fit.dofit(t_truncate, y_truncate, err_truncate);
    print('result.params = {}'.format(result.params));
    for key, par in result.params.items() :
        print('result.params[{}] = {}'.format(key, par));
        print('result.params[{}] = {}'.format(key, par.value));
        pass;

    #t_fitrange = np.linspace(50., 150., 1000);
    #fitted_y   = fitfunc(result[0], t_fitrange, None);

    #plt.errorbar(t, y, y_err);
    #plt.plot(t_fitrange, fitted_y, color='r');
    #plt.savefig('aho.png');

    fig, gridspec = fit.plotFitResult();
    fig.savefig('aho.png');



    pass;


