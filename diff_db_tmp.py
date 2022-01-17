#!/bin/python
import os, sys;
import numpy as np;

from matplotlib import pyplot as plt;

from compare_db import compare_db




if __name__=='__main__' :

    ver = 3;
    #dbname = 'all_pandas_correct_label';
    dbname = 'pb2a_hwpss_ver3';
    outname = f'hwpss/output_ver{ver}/db/diff_db_{dbname}';

    #ver = 10.2;
    #dbname = 'pb2a_wiregrid_hwpss_v3';
    #outname = f'output_ver{ver}/db/diff_db_{dbname}';
    dbnames =[
            'data/pb2a-20211004/pb2a_mapping.db',
            f'hwpss/output_ver{ver}/db/{dbname}.db',
            #f'output_ver{ver}/db/{dbname}.db',
            ];
    tablenames=[
            'pb2a_focalplane',
            #'hwpss',
            'pb2a_hwpss',
            #'pb2a_focalplane',
            ];
    columns=[
            'pol_angle,pixel_type,bolo_name,pixel_name,bolo_type,band,pixel_handedness'#,theta_det',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    suffixes=[
            '',
            '_1',
            ];
    varnames=[
            'bolo_name',  # The output diff_db.csv has the detectors with different bolo_name
            'pol_angle', 
            'pixel_type', 'pixel_name', 'bolo_type', 'band', 'pixel_handedness',
            ];
    selections=[
            "hardware_map_commit_hash='13decf63ba87f93ae31ae0b3e76dd020c91babd6'",
            '',
            ];
    dropNan=[
            False,
            False,
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
            outname=outname,
            doPlotAll = True,
            );

    pass;
