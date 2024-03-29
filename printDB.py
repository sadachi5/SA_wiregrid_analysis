import os, sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd;

from DBreader import DBreader;
from utils import getPandasPickle;

def main(database):
    df = getPandasPickle(database);
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 50)
    print('---- Pandas head in {} -----------'.format(database));
    print(df.head());
    print('----------------------------------');
    print('---- Pandas all in {} -----------'.format(database));
    print(df);
    print('----------------------------------');

    print(df.query('pixel_name=="13.13_232"'));
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 5)



    #database2 = 'data/pb2a-20210205/pb2a_mapping.db';
    database2 = 'data/pb2a-20211004/pb2a_mapping.db';
    db = DBreader(dbfilename=database2, tablename='pb2a_focalplane', idcolumn='readout_name');
    
    return 0;


if __name__=='__main__' :
    database = 'output_ver2/db/all_pandas.pkl'
    if len(sys.argv)>1:
        database = sys.argv[1];
        pass;
    main(database);
    pass;
    
 
