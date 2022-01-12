#!/bin/python
import os, sys;
import numpy as np;
import sqlite3, pandas;
import copy;
from utils import deg90to90, theta0topi, colors, printVar;
from matplotlib import pyplot as plt;

# 1. merge with DBs by primarycolumn
# 2. compare the columns in each DBs in varnames
# 3. make mislabel DB from varname1!=varname2
# 4. return dfs, dfmerge, dfmislabel;
#       dfs     : input DBs
#       dfmerge : merged DB
#       dfmislabel: mislabeled bolometer DB

def compare_db(
    dbnames, 
    tablenames,
    columns,
    selections,
    dropNan,
    suffixes,
    varnames,
    primarycolumn='readout_name',
    outname='aho',
    doPlotAll=False,
    ):

    dfs = [];
    dfmerge = None;
    for i, dbname in enumerate(dbnames) :
        tablename = tablenames[i];
        column    = columns[i];
        selection = selections[i];
        conn = sqlite3.connect(dbname);
        if not os.path.isfile(dbname) :
            print('ERROR! There is no file: {}'.format(dbname));
            pass;
        sqlite_cmd = 'SELECT {} {} FROM {} {}'.format(
            primarycolumn+',' if column!='*' else '', column, tablename, ('where '+selection) if not selection=='' else ''  )
        print('sqlite retrieving command = {}'.format(sqlite_cmd));
        df=pandas.read_sql_query(sqlite_cmd, conn);
        print('Size of {}th database = {} ({})'.format(i, len(df), dbname));
        if dropNan[i] : 
            print('Size of {}th database before dropNan = {} ({})'.format(i, len(df), dbname));
            df=df.dropna(subset=[varnames[i]]);
            print('Size of {}th database after dropNan = {} ({})'.format(i, len(df), dbname));
            pass;
        dfs.append(copy.deepcopy(df));
        #print(df.duplicated([primarycolumn]));
        #print('# of duplicated = {}'.format(sum(df.duplicated([primarycolumn]))));
        if i==0 :
            dfmerge = copy.deepcopy(df);
            #dfmerge = dfmerge.add_suffix(suffixes[i]);
            #dfmerge = dfmerge.rename(columns={primarycolumn+suffixes[0]: primarycolumn});
        else :
            dfmerge = pandas.merge(dfmerge, df, suffixes=['',suffixes[i]], how='left', on=primarycolumn);
            pass;
        pass;

    print(dfmerge);



    # Plots
    n_row = max(2, (int)(len(varnames)/3+1))
    n_column = 3;
    fig, axs = plt.subplots(n_row,n_column);
    fig.set_size_inches(n_column*4,n_row*4);
    fig.tight_layout(rect=[0,0,1,1]);
    fig2, axs2 = plt.subplots(len(varnames),1);
    fig2.set_size_inches(16, len(varnames)*3);
    fig2.tight_layout(rect=[0,0,1,1]);
    fig.subplots_adjust(wspace=0.5, hspace=0.4, left=0.30, right=0.95,bottom=0.30, top=0.80)
    fig2.subplots_adjust(wspace=0., hspace=0.6, left=0.30, right=0.95,bottom=0.30, top=0.80)
    for n, varname in enumerate(varnames):
        ax = axs[(int)(n/3)][n%3]
        ax2 = axs2[n]
        varname1 = varname;
        varname2 = varname+suffixes[1]
        v1 = dfmerge[varname1];
        v2 = dfmerge[varname2];

        # Different or Same histogram
        y = (v1==v2);
        y[v2.isnull()] = -1; # Nan data is replaced to -1.
        hist, bins, paches = ax.hist(y,bins=3,range=(-1.5,1.5),histtype='stepfilled', 
                align='mid', orientation='vertical',log=False,linewidth=0.5, linestyle='-', edgecolor='k');
 
        ax.grid(True);
        for i, count in enumerate(hist):
            ax.text(i-1, count*1., str(count), horizontalalignment='center', fontsize=10)
            pass;
        ax.set_xticks([-1.,0.,1.])
        ax.set_xticklabels(['Null', 'Different','Same'])
        ax.set_title(f'{varname1}',fontsize=10);
        ax.tick_params(labelsize=10);

        # Diff plot
        if doPlotAll:
            y2 = (v1==v2)
            y2[v2.isnull()] = -1; # Nan data is replaced to -1.
            isDiff = False
            if isinstance(v1[0], float): 
                isDiff = True;
            print(f'isDiff = {isDiff} (v1[0] = {v1[0]})')
            if isDiff: 
                y2 = deg90to90(v2 - v1);
                y2[v2.isnull()] = -90; # Nan data is replaced to -1.
            ax2.plot(range(len(y)), y2, linestyle='', color='k', marker='o', markersize=0.5, markerfacecolor='k');
            ax2.set_title(f'{varname1}',fontsize=10);
            ax2.set_xlabel(f'{primarycolumn} index'.replace('_', ' '),fontsize=10);
            ax2.set_ylabel(f'Difference' if isDiff else '',fontsize=10);
            # if varname1 == 'pol_angle': ax2.set_ylim(-20,20) # for pol_angle
            if not isDiff:
                ax2.set_yticks([-1.,0.,1.])
                ax2.set_yticklabels(['Null','Different','Same'])
            ax2.tick_params(labelsize=10);
            pass;
        pass;
    fig.savefig(outname+'.png');
    if doPlotAll: fig2.savefig(outname+'2.png');


    # Show 
    print('*************************');
    pandas.set_option('display.max_columns', 50)
    print(' check difference between {} and {}'.format(varname1,varname2));
    #dfmislabel = dfmerge.query('{}!={}'.format(varname1,varname2))[~(dfmerge[varname1].isnull())];
    dfmislabel0 = dfmerge.query('{}!={}'.format(varname1,varname2))
    # Drop NaN bolometers in varname1 & primarycolumn in DB1 (Bolometers in DB1 is not able to be compared.)
    dfmislabel = dfmislabel0.dropna(subset=[varname1,primarycolumn]);
    print(dfmislabel);
    pandas.set_option('display.max_columns', 5)
    print('Size of merged DB   = {}'.format(len(dfmerge)));
    print('Size of NaN DB in {} or {}   = {}'.format(varname1, primarycolumn, len(dfmislabel0)-len(dfmislabel)));
    print('Size of mislabel    = {}'.format(len(dfmislabel)));
    print('Size of correct label DB = merged DB - mislabel - NaN DB = {}'.format(len(dfmerge)-len(dfmislabel0)));

    dfmislabel.to_csv(outname+'.csv', header=True, index=True);


    return dfs, dfmerge, dfmislabel;




if __name__=='__main__' :
    dbnames =[
            'data/pb2a-20210205/pb2a_mapping.db',
            'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            #'output_ver4/db/all_pandas.db',
            ];
    tablenames=[
            'pb2a_focalplane',
            'pb2a_focalplane',
            #'wiregrid',
            ];
    columns=[
            'pol_angle,pixel_type,bolo_name,pixel_name,bolo_type,band,pixel_handedness,det_offset_x,det_offset_y',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    suffixes=[
            '',
            '_1',
            #'_1',
            ];
    varnames=[
            #'pol_angle',
            #'pixel_type',
            #'det_offset_x',
            #'bolo_name',
            'pixel_name',
            ];
    if len(varnames)==1 : varnames = [varnames[0] for i in range(len(dbnames))];
    selections=[
            "hardware_map_commit_hash='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            "hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            #'',
            ];
    dropNan=[
            True,
            True,
            #True,
            ];


    compare_db(
            dbnames    = dbnames,
            tablenames = tablenames,
            columns    = columns,
            varnames   = varnames,
            selections = selections,
            suffixes   = suffixes,
            dropNan    = dropNan,
            primarycolumn='readout_name',
            outname='aho.png',
            );

    pass;
