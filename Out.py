#!/usr/bin/env python
import os, sys;
import inspect;
import datetime;
import resource;
import numpy as np;

from utils import get_var_name;

class Out :
    m_verbosity = 0; # -1: Only ERROR / 0: +WARNING / 1: +USUAL OUT(0) / 2: +OUT(-1)
    m_outfile   = None;

    def __init__(self, verbosity=0, outfile=None) :
        self.m_verbosity = verbosity;
        self.m_outfile   = outfile  ;
        return;

    def setverbosity(self,verbosity):
        self.m_verbosity = verbosity;
        return 0;

    ## OUT() ##
    def OUT(self,comment,verbosityLevel=0,*args) :
        if verbosityLevel+self.m_verbosity>0 :
            mem      = resource.getrusage(resource.RUSAGE_SELF)[2];
            funcname = inspect.currentframe().f_back.f_code.co_name;
            filename = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1];
            if funcname=='<module>': 
                try : funcname = inspect.currentframe().f_back.f_globals['__name__'];
                except : funcname = 'Unknown';
            print("{} | {}() | {:.0f}MB | {} | {}".format(filename, funcname, mem/1024./1024., comment, args if len(args)>0 else ''));
            if self.m_outfile!=None : print("{} | {}() | {:.0f}MB | {} | {}".format(filename, funcname, mem/1024./1024., comment, args if len(args)>0 else ''), file=self.m_outfile);
            pass;
        return 0;
    ## end of OUT() ##

    ## OUTVar() ##
    def OUTVar(self,var,verbosityLevel=0,varname='',*args) :
        if verbosityLevel+self.m_verbosity>0 :
            mem      = resource.getrusage(resource.RUSAGE_SELF)[2];
            funcname = inspect.currentframe().f_back.f_code.co_name;
            filename = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1];
            back_vars = inspect.currentframe().f_back.f_globals;
            back_vars.update(inspect.currentframe().f_back.f_locals);
            if varname=='' : varname  = get_var_name(var,back_vars);
            if isinstance(var, (np.ndarray, list)) : varname = '{} (size:{})'.format(varname, len(var));
            if funcname=='<module>': 
                try : funcname = inspect.currentframe().f_back.f_globals['__name__'];
                except : funcname = 'Unknown';
            print("{} | {}() | {:.0f}MB | {:10s} = {} | {}".format(filename, funcname, mem/1024./1024., varname, var, args if len(args)>0 else ''));
            if self.m_outfile!=None : print("{} | {}() | {:.0f}MB | {:10s} = {} | {}".format(filename, funcname, mem/1024./1024., varname, var, args if len(args)>0 else ''), file=self.m_outfile);
            pass;
        return 0;
    ## end of OUT() ##



    ## WARNING() ##
    def WARNING(self,comment, *args) :
        verbosityLevel = 1;
        if verbosityLevel+self.m_verbosity>0 :
            mem      = resource.getrusage(resource.RUSAGE_SELF)[2];
            funcname = inspect.currentframe().f_back.f_code.co_name;
            filename = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1];
            if funcname=='<module>': 
                try : funcname = inspect.currentframe().f_back.f_globals['__name__'];
                except : funcname = 'Unknown';
            print("{} | {}() | {:.0f}MB | WARNING!! | {}".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '');
            if self.m_outfile!=None : print("{} | {}() | {:.0f}MB | WARNING!! | {}".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '', file=self.m_outfile);
            pass;
        return 0;
    ## end of WARNING() ##

    ## ERROR() ##
    def ERROR(self,comment,*args) :
        mem      = resource.getrusage(resource.RUSAGE_SELF)[2];
        funcname = inspect.currentframe().f_back.f_code.co_name;
        filename = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1];
        if funcname=='<module>': 
            try : funcname = inspect.currentframe().f_back.f_globals['__name__'];
            except : funcname = 'Unknown';
        print("{} |     {}() | {:.0f}MB | ERROR!! | {}".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '');
        if self.m_outfile!=None : print("{} |     {}() | {:.0f}MB | ERROR!! | {}".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '', file=self.m_outfile);
        return 0;
    ## end of ERROR() ##

    ## MEMOUT() ##
    def MEMOUT(self,verbosityLevel=0) :
        if verbosityLevel+self.m_verbosity>0 :
            mem      = resource.getrusage(resource.RUSAGE_SELF)[2];
            funcname = inspect.currentframe().f_back.f_code.co_name;
            filename = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1];
            if funcname=='<module>': 
                try : funcname = inspect.currentframe().f_back.f_globals['__name__'];
                except : funcname = 'Unknown';
            print("{} | {}() | {:f.0}MB".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '');
            if self.m_outfile!=None : print("{} | {}() | {:f.0}MB".format(filename, funcname, mem/1024./1024., comment), args if len(args)>0 else '');
            pass;
        return 0;
    ## end of MEMOUT() ##





