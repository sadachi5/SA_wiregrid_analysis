#!/bin/python
import numpy as np;
import math, sqlite3;
from compare_db import compare_db;
from utils import theta0topi, rad_to_deg, deg_to_rad, diff_angle;

if __name__=='__main__' :
    dbname_out = 'output_ver4/db/pb2a_mapping_corrected_label_v2.db'
    dbnames =[
            'data/pb2a-20210205/pb2a_mapping.db',
            'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            'output_ver4/db/all_pandas.db',
            ];
    tablenames=[
            'pb2a_focalplane',
            'pb2a_focalplane',
            'wiregrid',
            ];
    columns=[
            '*',
            'pixel_name,band',
            'theta_det,theta_det_err,tau',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    suffixes=[
            '', # NEED TO BE EMPTY (No suffix)
            '_fix',
            '', # for wiregrid data
            ];
    varnames=[
            'pixel_name',
            ];
    if len(varnames)==1 : varnames = [varnames[0] for i in range(len(dbnames))];
    selections=[
            "hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            "hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            '',
            ];
    selection_wg = '';
    dropNan=[
            False,
            False,
            False,
            ];

    # dfmislabel: bolometers with
    # - pixel_name_original!=pixel_name_corrected or 
    # - nan pixel_name_corrected
    df0s, df, dfmislabel = compare_db(
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

    #print(df);
    #print(dfmislabel);
    nfails = [0,0,0];
    nwarns = [0,0,0,0,0];
    nwarns_angle= [0,0,0,0];
    names_fail = [];
    check_columns = ['pixel_type', 'bolo_type', 'pixel_handedness'];
    dfmislabel = dfmislabel.reset_index();
    # add mislabel column
    df_new = df0s[0];
    df_new['mislabel']=False;
    print(dfmislabel['readout_name']);
    for i in range(len(dfmislabel)) :
        print('******** i={} *********'.format(i));
        name          = dfmislabel.at[i,'readout_name'];
        boloname      = dfmislabel.at[i,'bolo_name'];
        # initialize variables
        band_fix       = np.nan;
        pixel_type_fix = '';
        bolo_type_fix  = '';
        pixel_handedness_fix = '';
        pol_angle_fix  = np.nan;
        if name is None :
            print('WARNING! readout_name is None (reaout_name={}).'.format(name));
            names_fail.append(name);
            nfails[0]+=1;
        else :
            # pixel_name
            pixel_name_fix= dfmislabel.at[i,'pixel_name_fix'];
            if pixel_name_fix!=pixel_name_fix : # if NaN
                print('WARNING! Could not find correct pixel data for {} (No pixel_name_fix).'.format(name));
                names_fail.append(name);
                nfails[1]+=1;
            else :
                # Get correct data
                querycmd = "pixel_name=='{}'".format(pixel_name_fix);
                print('querycmd = {}'.format(querycmd));
                correct_info  = df.query(querycmd);
                #print(correct_info);
                if len(correct_info)==0:
                    print('WARNING! Could not find correct pixel data for {} (No pixel_name={}).'.format(name,pixel_name_fix));
                    names_fail.append(name);
                    nfails[2]+=1;
                else :
                    # band
                    band_fix = dfmislabel.at[i,'band_fix'];
                    if np.isnan(band_fix):
                        print('WARNING! {} has NaN band!'.format(name));
                        nwarns[0]+=1;
                        pass;
                    # pixel_type
                    pixel_types_fix = np.array(correct_info['pixel_type']);
                    if np.all(pixel_types_fix==pixel_types_fix[0]) :
                        print('pixel_type : all same = {}'.format(pixel_types_fix[0]));
                        pixel_type_fix = pixel_types_fix[0];
                    else :
                        print('pixel_type : different = {}'.format(pixel_types_fix));
                        nwarns[1]+=1;
                        pass;
                    # bolo_type
                    bolo_types_fix = np.array(correct_info['bolo_type']);
                    if np.all(bolo_types_fix==bolo_types_fix[0]) :
                        print('bolo_type : all same = {}'.format(bolo_types_fix[0]));
                        bolo_type_fix = bolo_types_fix[0];
                    else :
                        print('bolo_type : different = {}'.format(bolo_types_fix));
                        nwarns[2]+=1;
                        pass;
                    # pixel_handedness
                    pixel_handednesses_fix = np.array(correct_info['pixel_handedness']);
                    if np.all(pixel_handednesses_fix==pixel_handednesses_fix[0]) :
                        print('pixel_handedness : all same = {}'.format(pixel_handednesses_fix[0]));
                        pixel_handedness_fix = pixel_handednesses_fix[0];
                    else :
                        print('pixel_handedness : different = {}'.format(pixel_handednesses_fix));
                        nwarns[3]+=1;
                        pass;
                    # pol_angle
                    pol_angles_fix = np.array(correct_info['pol_angle']);
                    if np.all(pol_angles_fix==pol_angles_fix[0]) :
                        print('pol_angle : all same = {}'.format(pol_angles_fix[0]));
                        pol_angle_fix = pol_angles_fix[0];
                    else :
                        print('pol_angle : different = {} [deg]'.format(pol_angles_fix));
                        nwarns[4]+=1;
                        theta_det = dfmislabel.at[i,'theta_det'];
                        tau       = dfmislabel.at[i,'tau'];
                        if not tau>0. :
                            print('tau measured by stimulator is not good (not tau>0.)');
                            print('--> theta_det = nan (skip correction)');
                            theta_det = np.nan;
                            pass;
                        print('theta_det = {} [rad]'.format(theta_det));
                        if theta_det!=theta_det :
                            nwarns_angle[0]+=1;
                            print('pol_angle : No theta_det = {}'.format(theta_det));
                        else :
                            #calib_angle = rad_to_deg(theta0topi(theta_det - np.pi/2.));
                            calib_angle = theta0topi(theta_det - np.pi/2.);
                            print('calib_angle = {} [rad]'.format(calib_angle));
                            diff_angles = rad_to_deg(diff_angle(deg_to_rad(pol_angles_fix), calib_angle, upper90deg=True));
                            # change nan --> 9999.
                            diff_angles = np.where(np.isnan(diff_angles), 9999., diff_angles);
                            print('diff_angles = {} [deg]'.format(diff_angles));
                            mindiff = min(diff_angles);
                            index = np.where(diff_angles == mindiff)[0];
                            # pol_angle
                            #print(index);
                            pol_angles_fix2 = [pol_angles_fix[i] for i in index];
                            #print(pol_angles_fix2);
                            if np.all(pol_angles_fix2==pol_angles_fix2[0]):
                                print('pol_angle after pol_angle selection : all same = {}'.format(pol_angles_fix2[0]));
                                pol_angle_fix = pol_angles_fix2[0];
                            else :
                                print('pol_angle after pol_anlge selection : different = {}'.format(pol_angles_fix2));
                                nwarns_angle[1]+=1;
                                pass;
                            # pixel_type
                            pixel_types_fix2 = [ pixel_types_fix[i] for i in index ];
                            # remove nan
                            pixel_types_fix2 = np.delete( pixel_types_fix2, np.where(not isinstance(pixel_types_fix2,str)) );
                            if len(pixel_types_fix2)>0 and np.all(pixel_types_fix2==pixel_types_fix2[0]):
                                print('pixel_type after pol_angle selection : all same = {}'.format(pixel_types_fix2[0]));
                                pixel_type_fix = pixel_types_fix2[0];
                            else :
                                print('pixel_type after pol_angle selection : different = {}'.format(pixel_types_fix2));
                                nwarns_angle[2]+=1;
                                pass;
                            # bolo_type
                            bolo_types_fix2 = [ bolo_types_fix[i] for i in index ];
                            # remove nan
                            bolo_types_fix2 = np.delete( bolo_types_fix2, np.where(not isinstance(bolo_types_fix2,str)) );
                            if len(bolo_types_fix2)>0 and np.all(bolo_types_fix2==bolo_types_fix2[0]):
                                print('bolo_type after pol_angle selection : all same = {}'.format(bolo_types_fix2[0]));
                                bolo_type_fix = bolo_types_fix2[0];
                            else :
                                print('bolo_type after pol_angle selection : different = {}'.format(bolo_types_fix2));
                                nwarns_angle[3]+=1;
                                pass;
                            pass;
                        pass;
                    pass;
                pass;
            pass;

        #index = df_new.reset_index().query("readout_name=='{}'".format(name)).index;
        index = df_new.reset_index().query("bolo_name=='{}'".format(boloname)).index;
        if len(index)>1 : 
            print('Error! There are multiple data for readout_name=={}'.format(boloname));
        else :
            index = index[0];
            pass;
        print('index for {} = {}'.format(boloname,index));
        df_new.at[index, 'readout_name'] = name if name is not None else '';
        df_new.at[index, 'pixel_type'] = pixel_type_fix;
        df_new.at[index, 'bolo_type']  = bolo_type_fix;
        df_new.at[index, 'pixel_handedness'] = pixel_handedness_fix;
        df_new.at[index, 'pol_angle']  = pol_angle_fix;
        df_new.at[index, 'mislabel']  = True;
        pass;
    
    print('# of bolos failing to get correct values for each data.');
    print('  band             = {}'.format(nwarns[0]));
    print('  pixel_type       = {}'.format(nwarns[1]));
    print('  bolo_type        = {}'.format(nwarns[2]));
    print('  pixel_handedness = {}'.format(nwarns[3]));
    print('  pol_angle        = {}'.format(nwarns[4]));
    print('    No theta_det          = {}/{}'.format(nwarns_angle[0], nwarns[4]));
    print('    Failed for pol_angle  = {}/{}'.format(nwarns_angle[1], nwarns[4]));
    print('    Failed for pixel_type = {}/{}'.format(nwarns_angle[2], nwarns[4]));
    print('    Failed for bolo_type  = {}/{}'.format(nwarns_angle[3], nwarns[4]));

    print('Total bolos size={}'.format(len(df_new)));
    print('Correct bolos size={}'.format(len(df_new)-sum(nfails)));
    print('  Failed 0 (readout_name=None ) size={}'.format(nfails[0]));
    print('  Failed 1 (pixel_name_fix=NaN) size={}'.format(nfails[1]));
    print('  Failed 2 (No pixel_name_fix ) size={}'.format(nfails[2]));
    print('Failed bolos (size={}):'.format(sum(nfails)));
    #print(names_fail);
    print(df_new);
    #for name in names_fail :
    #    print(name);
    #    pass;

    # Save
    conn = sqlite3.connect(dbname_out);
    df_new.to_sql('pb2a_focalplane',conn,if_exists='replace',index=None);
    conn.close();


