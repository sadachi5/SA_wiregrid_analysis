import sqlite3


def mergeDB(newfile, filenames, tablename, verbose=0) :
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



if __name__=='__main__' :
    newfile = 'aho.db';
    tablename = 'wiregrid'

    #directory = '/home/cmb/sadachi/analysis_2021/output_ver2';
    directory = '/Users/shadachi/Experiment/PB/analysis/analysis_2021/output_ver2';
    filenames = [
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch22.db'.format(directory),
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch24.db'.format(directory),
        '{}/db/PB20.13.13/Fit_PB20.13.13_Comb08Ch25.db'.format(directory),
        ];

    verbose = 1;
    
    mergeDB(newfile, filenames, tablename, verbose=verbose);
    pass;
    
