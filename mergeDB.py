import os, sys;

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
    
      columndef = ' {} {}'.format(name, ctype);
      if notnull : columndef += ' NOT NULL';
      if not default is None : columndef += ' DEFAULT {}'.format(default);
      if pk      : columndef += ' PRIMARY KEY';
      else       : notPrimaryColumns.append(name);
    
      print('column definition : {}'.format(columndef));
      columndefs.append(columndef);
      pass;
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
        # Attach input file
        con.execute("ATTACH DATABASE '{}' as file{}".format(filename,i));
        con.commit();
        # Insert data  from attached file
        insertcmd = 'INSERT INTO {table}({columns}) SELECT {columns} FROM file{i}.{table}'\
            .format(table=tablename, columns=insertColumns, i=i);
        cur.execute(insertcmd);
        con.commit();
        # Detach the input file
        con.execute('DETACH DATABASE file{}'.format(i));
        con.commit();
        pass;
    
    # Close the new file
    cur.close()
    con.close()
    
    # Check the new file
    con = sqlite3.connect(newfile);
    cur = con.cursor();
    cur.execute('SELECT * FROM {}'.format(tablename));
    print('{} in {} :'.format(tablename, newfile));
    print('    {}'.format(cur.fetchall()));
    cur.close();
    con.close();

    return 0;

def convertSQLtoPandas(sqlfile, outputfile, tablename='wiregrid', addDB=[], verbose=0):
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
    df['boloname'] = df['boloname'].str.replace('"(.*)"', r'\1', regex=True);
    print('--- Pandas header after removing " in boloname ----');
    print(df.head());
    print('---------------------------------------------------');

    # Add DB from another database
    if len(addDB)>0 :
        for db in addDB :
            __dbfilename = db[0];
            __dbtablename= db[1];
            if not os.path.isfile(__dbfilename) :
                print('WARNING! There is no adding databese: {}'.format(__dbfilename));
                print('         --> Skip!!');
                continue;
            print('Adding {}..'.format(__dbfilename));
            __conn = sqlite3.connect(__dbfilename);
            __df=pandas.read_sql_query('SELECT * FROM {}'.format(__dbtablename), __conn);
            __dfnew = __df.rename(columns={'name':'boloname'});
            if verbose>0 : 
                print('--- Added pandas header ---------------');
                print(__dfnew.head());
                print('---------------------------------');
                pass;
            dfnew = pandas.merge(df, __dfnew, how='inner', on='boloname');
            #__conn.close();
            #del __df;
            #del __dfnew;
            df = copy.deepcopy(dfnew);
            pass;
        
        print('=== Pandas Data after adding other databases =============================');
        if verbose>0 : 
            print('--- Pandas Data ---------------');
            print(df);
            print('---------------------------------');
            pass;
        print('--- Pandas header ---------------');
        print(df.head());
        print('---------------------------------');
        pass;

    # Save Pandas to pickle file
    if not outputfile.endswith('.pkl') : outputfile += '.pkl';
    print('Saving the pandas to a pickle file ({})...'.format(outputfile));
    df.to_pickle(outputfile);
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
    inputdir = '/home/cmb/sadachi/analysis_2021/output_ver2';
    #inputdir = '/Users/shadachi/Experiment/PB/analysis/analysis_2021/output_ver2';
    newfile = 'output_ver2/db/all';
    verbose = 1;
    
    # merge sqlite3 db files
    #mergeAllDB(inputdir=inputdir, newfile=newfile, ispickle=False, tablename=tablename, verbose=verbose);
    # convert the merged sqlite3 db to pandas data (in a pickle file)
    convertSQLtoPandas(sqlfile=newfile+'.db', outputfile=newfile+'_pandas', tablename=tablename, addDB=[['data/boloid_pb2a_20210412.db','boloid']], verbose=verbose);

    # merge pickle files
    #mergeAllDB(inputdir=inputdir, newfile=newfile, ispickle=True, tablename=tablename, verbose=verbose);

    pass;
    
