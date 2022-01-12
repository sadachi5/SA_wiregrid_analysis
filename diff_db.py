#!/bin/python
import os, sys;
import numpy as np;

from matplotlib import pyplot as plt;

from compare_db import compare_db



if __name__=='__main__' :

    ver = 10;
    outname = f'output_ver{ver}/db/diff_db';
    dbnames =[
            'data/pb2a-20211004/pb2a_mapping.db',
            f'output_ver{ver}/db/pb2a_wiregrid_ver{ver}.db',
            ];
    tablenames=[
            'pb2a_focalplane',
            'pb2a_wiregrid',
            ];
    columns=[
            'pol_angle,pixel_type,bolo_name,pixel_name,bolo_type,band,pixel_handedness',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    suffixes=[
            '',
            '_1',
            ];
    varnames=[
            'pol_angle', 
            'pixel_type', 
            'bolo_name', 'pixel_name', 'bolo_type', 'band', 'pixel_handedness',
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
