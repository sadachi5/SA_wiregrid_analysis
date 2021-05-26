import os, sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd;

from DBread import DBreader;

def main():
    database = 'output_ver2/db/all_pandas.pkl';
    df = pd.read_pickle(database);
    pd.set_option('display.max_columns', 30)
    print('---- Pandas head in {} -----------'.format(database));
    print(df.head());
    print('----------------------------------');
    pd.set_option('display.max_columns', 5)

    db = DBreader(dbfilename='./data/boloid_pb2a_20210412.db');
    db.printAll();
    
    return 0;


if __name__=='__main__' :
    main();
    pass;
    
 
