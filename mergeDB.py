import os, sys;
import numpy as np;

def getSQLColumnDefs(con, cur, tablename, verbose=0):
    # Get table info from filetmp
    cur.execute('PRAGMA TABLE_INFO({})'.format(tablename));
    info=cur.fetchall();
    if verbose>0 : print(info);
    
    # Retrieve columns info
    columndefs = [];
    notPrimaryColumns = [];
    for column in info :
      cid    = column[0];
      name   = column[1];
      ctype  = column[2];
      notnull= column[3];
      default= column[4];
      pk     = column[5];
    
      if name.endswith('deg') and '.' in name: name = '[{}]'.format(name);
      columndef = '{} {}'.format(name, ctype);
      if notnull : columndef += ' NOT NULL';
      if not default is None : columndef += ' DEFAULT {}'.format(default);
      if pk      : columndef += ' PRIMARY KEY';
      else       : notPrimaryColumns.append('{}'.format(name));
    
      print('column definition : {}'.format(columndef));
      columndefs.append(columndef);
      pass;
    return columndefs, notPrimaryColumns;

def mergeDB(newfile, filenames, tablename, verbose=0) :
    import sqlite3
    # Open new database
    con = sqlite3.connect(newfile)
    cur = con.cursor()
    # Remove table if exists
    con.execute('DROP TABLE if exists {}'.format(tablename));
    con.commit();
    
    # Attach one of input databases (filetmp) to retrieve table info
    con.execute("ATTACH DATABASE '%s' as filetmp" % (filenames[0]));
    con.commit();
    # Check tables in the filetmp
    if verbose>0:
        cur.execute("SELECT name FROM filetmp.sqlite_master WHERE type='table'");
        print('{} :'.format(filenames[0]));
        print('    tables = {}'.format(cur.fetchall()));
        cur.execute('SELECT  * FROM filetmp.{}'.format(tablename));
        print('    data of {} = {}'.format(tablename, cur.fetchall()));
        pass;
    

    # Get table definition
    columndefs, notPrimaryColumns = getSQLColumnDefs(con,cur,tablename,verbose);
    tabledef = ','.join(columndefs);
    
    # Create new table in newfile
    con.execute('CREATE TABLE {}({})'.format(tablename, tabledef));
    con.commit();
    
    # Detach filetmp
    sql = 'DETACH DATABASE filetmp';
    con.execute(sql);
    con.commit();
    
    # Insert into the new file from input files
    insertColumns = ','.join(notPrimaryColumns);
    for i,filename in enumerate(filenames):
        if i%100 == 0 : print('{}th file is being merged... ({})'.format(i, filename));
        #print('{}th file is being merged... ({})'.format(i, filename));
        # Attach input file
        con.execute("ATTACH DATABASE '{}' as file{}".format(filename,i));
        con.commit();
        # Insert data  from attached file
        insertcmd = 'INSERT INTO {table}({columns}) SELECT {columns} FROM file{i}.{table}'\
            .format(table=tablename, columns=insertColumns, i=i);
        #print(insertcmd);
        cur.execute(insertcmd);
        con.commit();
        # Detach the input file
        con.execute('DETACH DATABASE file{}'.format(i));
        con.commit();
        pass;
    
    # Close the new file
    print(cur);
    cur.close()
    con.close()
    
    # Check the new file
    con = sqlite3.connect(newfile);
    cur = con.cursor();
    print('{} in {} :'.format(tablename, newfile));
    cur.execute("SELECT * FROM sqlite_master WHERE type='table'");
    print('    tables = {}'.format(cur.fetchall()));
    #cur.execute('SELECT * FROM {}'.format(tablename));
    #print('    {}'.format(cur.fetchall()));
    cur.close();
    con.close();

    return 0;

def modifySQL(sqlfile, newfile, tablename='wiregrid', verbose=0) :
    import sqlite3
    # open SQL file
    con = sqlite3.connect(newfile);
    cur = con.cursor();
    # Remove table if exists
    con.execute('DROP TABLE if exists {}'.format(tablename));
    con.commit();
    con.execute('DROP TABLE if exists tabletmp');
    con.commit();
    # Copy table
    con.execute("ATTACH DATABASE '%s' as filetmp" % (sqlfile));
    con.commit();
    copycmd = 'CREATE TABLE {table} AS SELECT * FROM filetmp.{table}'.format(table=tablename);
    cur.execute(copycmd);
    con.commit();
    con.execute('DETACH DATABASE filetmp');
    con.commit();

    # Modification

    # Remove " in boloname
    replace_cmd = 'UPDATE {table} SET {column}=REPLACE({column},\'"\',"");'.format(table=tablename, column='boloname');
    if verbose>0 : print('Replace command: {}'.format(replace_cmd));
    con.execute(replace_cmd);
    con.commit();
    # Change column name: "boloname" --> "readout_name"
    columndefs, notPrimaryColumns = getSQLColumnDefs(con,cur,tablename,verbose);
    for i,column in enumerate(columndefs):
        print(column);
        if   column.startswith('boloname NUM'): columndefs[i]=column.replace('boloname NUM','readout_name TEXT');
        elif column.startswith('boloname '   ): columndefs[i]=column.replace('boloname '   ,'readout_name ');
        pass;
    tabledef = ','.join(columndefs);
    print(tabledef);
    con.execute('ALTER TABLE {table} RENAME TO tabletmp'.format(table=tablename));
    con.commit();
    con.execute('CREATE TABLE {table}({tabledef})'.format(table=tablename, tabledef=tabledef));
    con.commit();
    cur.execute('INSERT INTO {table} SELECT * FROM tabletmp;'.format(table=tablename));
    con.commit();
    # Add columns (theta_det = theta_wire0/2, theta_det_err = theta_wire0/2)
    con.execute('ALTER TABLE {table} ADD COLUMN theta_det'.format(table=tablename));
    con.commit();
    con.execute('ALTER TABLE {table} ADD COLUMN theta_det_err'.format(table=tablename));
    con.commit();
    cur.execute('UPDATE {table} SET(theta_det, theta_det_err) = (SELECT theta_wire0*0.5,theta_wire0_err*0.5 FROM tabletmp WHERE id = {table}.id) '.format(table=tablename));
    con.commit();
    con.execute('DROP TABLE tabletmp');
    con.commit();
    con.execute('VACUUM');
    con.commit();

    # Close SQL file
    cur.close();
    con.close();

    # Check SQL file
    con = sqlite3.connect(newfile);
    cur = con.cursor();
    cur.execute('SELECT * FROM sqlite_master where type="table"');
    print('Input DB table structure : {}'.format(cur.fetchall()[0]));
    cur.execute('SELECT * FROM {}'.format(tablename));
    print('{} in {} :'.format(tablename, sqlfile));
    print('    {}'.format(cur.fetchall()));

    return 0;

def convertSQLtoPandas(sqlfile, outputfile, tablename='wiregrid', addDB=[], doTauCalib=False, verbose=0):
    import sqlite3, pandas;
    import copy;
    # Open SQL
    conn = sqlite3.connect(sqlfile);
    # Convert SQL to Pandas
    print('Converting the SQL data in {} to pandas...'.format(sqlfile));
    df=pandas.read_sql_query('SELECT * FROM {}'.format(tablename), conn);
    if verbose>0 : 
        print('--- Pandas Data ---------------');
        print(df);
        print('---------------------------------');
        pass;
    print('--- Pandas header ---------------');
    print(df.head());
    print('---------------------------------');
    # Close SQL file
    conn.close();

    # Modify Pandas DB to add another DB
    #df['boloname'] = df['boloname'].str.replace('"(.*)"', r'\1', regex=True);
    #df = df.rename(columns={'boloname':'readout_name'});
    #print('--- Pandas header after removing " in readout_name(<-boloname) ----');
    #print(df.head());
    #print('-------------------------------------------------------------------');

    # Add DB from another database
    if len(addDB)>0 :
        for db in addDB :
            __dbfilename = db[0];
            __dbtablename= db[1];
            __dbselection= None;
            __dbboloname = None;
            if len(db)>2 and db[2]!='' : __dbselection= db[2];
            if len(db)>3 and db[3]!='' : __dbboloname = db[3];
            if not os.path.isfile(__dbfilename) :
                print('WARNING! There is no adding databese: {}'.format(__dbfilename));
                print('         --> Skip!!');
                continue;
            print('Adding {}..'.format(__dbfilename));
            __conn = sqlite3.connect(__dbfilename);
            __query = 'SELECT * FROM {} {}'.format(__dbtablename, ('where '+__dbselection) if not __dbselection is None else '' );
            print('query = {}'.format(__query));
            __df=pandas.read_sql_query(__query, __conn);
            # Modify __df
            if not __dbboloname is None :
                __dfnew = __df.rename(columns={__dbboloname:'readout_name'});
            else :
                __dfnew = __df; # No modification
                pass;
            if verbose>0 : 
                pandas.set_option('display.max_columns', 20)
                print('--- Added pandas header ---------------');
                print(__dfnew.head());
                print('---------------------------------------');
                pandas.set_option('display.max_columns', 5)
                pass;
            dfmerge = pandas.merge(df, __dfnew, how='left', on='readout_name');
            #__conn.close();
            #del __df;
            #del __dfnew;
            df = copy.deepcopy(dfmerge);
            print('=== Pandas Data after adding {} ============================='.format(__dbfilename));
            if verbose>0 : 
                print('--- Pandas Data -----------------');
                print(df);
                print('---------------------------------');
                pass;
            print('--- Pandas header ---------------');
            print(df.head());
            print('---------------------------------');
            pass;
        pass;

    # do Tau Calibration
    if doTauCalib and 'tau' in df.keys() :
        hwp_speed = 2.; # [Hz]
        df.loc[(df['tau']>0.), 'theta_det'] = df['theta_det'] - 2.*df['tau'] * (hwp_speed * 2. * np.pi );
        pass;
        
    # Check outputfile name
    if outputfile.endswith('.pkl') : outputfile = '.'.join(outputfile.split('.')[:-1]);
    if outputfile.endswith('.db')  : outputfile = '.'.join(outputfile.split('.')[:-1]);

    # Save Pandas to pickle file
    outputfullname = outputfile + '.pkl';
    print('Saving the pandas to a pickle file ({})...'.format(outputfullname));
    df.to_pickle(outputfullname);

    # Save Pandas to sqlite4 file
    outputfullname = outputfile + '.db';
    print('Saving the pandas to a sqlite3 file ({})...'.format(outputfullname));
    conn = sqlite3.connect(outputfullname);
    df.to_sql('wiregrid',conn,if_exists='replace',index=None);
    conn.close();
    del conn;

    return 0;

def mergeDBpkl(newfile, filenames, verbose=0) :
    import pickle;
    # Open new database
    outfile = open(newfile, 'wb');
    
    # Retrieve the data from ipnut pickle files
    data_array = [];
    for i,filename in enumerate(filenames):
        if i%100 == 0 : print('{}th file is being merged... ({})'.format(i, filename));
        pklfile = open(filename, 'rb');
        try : 
            data = pickle.load(pklfile);
        except EOFError:
            print('WARNING! There is no data in {}!'.format(filename));
            print('    --> Skip!!');
        else :
            data_array.append(data);
            pass;
        pklfile.close();
        pass;

    # Modify the data
    newdata_array = [];
    for i, data in enumerate(data_array) :
        newdata = {};
        for column in data :
            if column[0]=='id' : newdata[column[0]] = i;
            else               : newdata[column[0]] = column[2];
            pass;
        newdata_array.append(newdata);
        pass;
    pickle.dump(newdata_array,outfile);
    
    # Close the new file
    outfile.close();
    
    # Check the new file
    outfile = open(newfile,'rb');
    print('New file : {}'.format(newfile));
    data_array = pickle.load(outfile);
    for i, data in enumerate(data_array) :
        print('i={:4d} : {}'.format(i, data));
        pass;
    outfile.close();

    return 0;



def mergeAllDB(inputdir, newfile, ispickle=True, tablename='wiregrid', verbose=1) :
    ext       = 'pkl' if ispickle else 'db';

    if not newfile.endswith(ext) : newfile = '{}.{}'.format(newfile, ext);

    import subprocess;
    wafers=['PB20.13.13', 'PB20.13.15', 'PB20.13.28', 'PB20.13.11', 'PB20.13.12', 'PB20.13.10', 'PB20.13.31'];
    filenames = [];
    for w in wafers :
        lscmd = 'ls -1 {}/db/{}/*.{}'.format(inputdir,w,ext);
        print(lscmd);
        lsout = subprocess.getoutput(lscmd).split('\n');
        filenames += lsout;
        print('Current size of filenames = {}'.format(len(filenames)));
        pass;
    print('Final size of filenames = {}'.format(len(filenames)));

    if ispickle : mergeDBpkl(newfile, filenames, verbose=verbose);
    else        : mergeDB(newfile, filenames, tablename, verbose=verbose);
    return 0;
 


if __name__=='__main__' :
    tablename = 'wiregrid'
    #inputdir = './output_ver2';
    #inputdir = './output_ver3';
    #newfile = 'output_ver4/db/all';
    #inputdir = './output_ver5';
    #newfile = 'output_ver5/db/all';
    inputdir = './output_ver8';
    newfile = 'output_ver8/db/all';
    oldfile = '{}/db/all'.format(inputdir);
    #inputdir = '/home/cmb/sadachi/analysis_2021/output_ver2';
    #inputdir = '/Users/shadachi/Experiment/PB/analysis/analysis_2021/output_ver2';
    #newfile = '{}/db/all'.format(inputdir);
    doModify = False;
    doTauCalib = True;
    verbose = 1;
    
    # merge sqlite3 db files
    mergeAllDB(inputdir=inputdir, newfile=newfile, ispickle=False, tablename=tablename, verbose=verbose);

    # modify table (boloname "???" --> ???, column: boloname NUM-->readout_name TEXT)
    if doModify : modifySQL(sqlfile=oldfile+'.db', newfile=oldfile+'_mod.db', tablename=tablename, verbose=verbose);
    # convert the merged sqlite3 db to pandas data (in a pickle file)
    convertSQLtoPandas(sqlfile=oldfile+('_mod.db' if doModify else '.db'), outputfile=newfile+'_pandas', tablename=tablename, verbose=verbose, doTauCalib=doTauCalib,
            addDB=[
                ['data/pb2a-20210205/pb2a_mapping.db','pb2a_focalplane', "hardware_map_commit_hash='6f306f8261c2be68bc167e2375ddefdec1b247a2'",None],
                ['data/pb2a_stimulator_run223_20210223.db','pb2a_stimulator', "run_id=='22300610'", 'Bolo_name'],
                #['data/pb2a_stimulator_run223_20210223.db','pb2a_stimulator', "run_id=='22300607'", 'Bolo_name'],
                ]);

    # merge pickle files
    #mergeAllDB(inputdir=inputdir, newfile=newfile, ispickle=True, tablename=tablename, verbose=verbose);

    pass;
    
