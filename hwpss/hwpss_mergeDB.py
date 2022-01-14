from mergeDB import *


if __name__=='__main__' :

    #########
    # addDB #
    #########
    # addDB = list of ['DB filename', 'tablename', 'selection', 'readout culumn name', 'how to merge']
    # how to merge: 
    #    left : The added DB is merged to the original DB.
    #    right: The original DB is merged to the added DB.

    ver = 3;
    tablename = 'hwpss'
    inputdir = './hwpss/output_ver{}/db'.format(ver);
    inputfile = '{}/hwpss_with_time_constant_v3'.format(inputdir);
    addDB=[
            # Focalplane DB: Modify how to merge in ver10
            ['data/pb2a-20210205/pb2a_mapping.db','pb2a_focalplane', "hardware_map_commit_hash='6f306f8261c2be68bc167e2375ddefdec1b247a2'",None,'right'],
            # Stimulator DB: Update in ver10
            ['data/pb2a-20211004/pb2a_stim.db','pb2a_stimulator', "run_id=='22300610' and run_subid=='[1, 4, 7, 10, 13, 16, 19, 22]'", 'Bolo_name','left'],
          ];

    doTauCalib = False;
    verbose = 1;
    
    # convert the merged sqlite3 db to pandas data
    convertSQLtoPandas(sqlfile=inputfile+'.db', outputfile=inputdir+'/all_pandas', 
            tablename=tablename, verbose=verbose, doTauCalib=doTauCalib, addDB=addDB, isHWPSS=True);

    pass;
    
