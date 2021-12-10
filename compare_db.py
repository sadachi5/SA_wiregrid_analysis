#!/bin/python
import os, sys;
import numpy as np;
import sqlite3, pandas;
import copy;
from utils import theta0topi, colors, printVar;
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



    fig, axs = plt.subplots(1,2);
    fig.set_size_inches(12,6);
    #fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=0.3, hspace=0., left=0.15, right=0.95,bottom=0.15, top=0.95)

    # 2D plot
    varname1 = varnames[0];
    varname2 = varnames[1]+suffixes[1]
    v1 = dfmerge[varname1];
    v2 = dfmerge[varname2];
    y = (v1==v2)
    #print(v1);
    #print(v2);
    #print(y);
    #axs[0].plot(x,y,marker='o', markersize=0.5, linestyle='');
    hist, bins, paches = axs[0].hist(y,bins=2,range=(-0.5,1.5),histtype='stepfilled', align='mid', orientation='vertical',log=True,linewidth=0.5, linestyle='-', edgecolor='k');

    xmax = 36000;
    #axs[0].plot([-xmax,xmax],[-xmax,xmax],linestyle='-',color='k',linewidth=0.5);
    #axs[0].plot([-xmax,xmax],[0,0],linestyle='-',color='k',linewidth=0.5);
    #axs[0].plot([0,0],[-xmax,xmax],linestyle='-',color='k',linewidth=0.5);

    axs[0].grid(True);
    for i, count in enumerate(hist):
        axs[0].text(i, count*1., str(count), horizontalalignment='center',fontsize=12)
        pass;
    axs[0].set_title(f'{varname1} v.s. {varname2}',fontsize=8);
    axs[0].set_xlabel(r'DB1==DB2',fontsize=16);
    #axs[0].set_xlabel(r'$\theta_{\mathrm{det,wiregrid}}$ - 90 [deg.]',fontsize=16);
    #axs[0].set_ylabel(r'$\theta_{\mathrm{det,design}}$ [deg.]'+'\n("pol_angle" in focalplane database)',fontsize=16);
    #axs[0].set_xticks(np.arange(-360,360,45));
    #axs[0].set_yticks(np.arange(-360,360,45));
    #axs[0].set_xlim(-22.5,180);
    #axs[0].set_ylim(-22.5,180);
    axs[0].tick_params(labelsize=12);

    fig.savefig(outname+'.png');



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
