#!/bin/python
import numpy as np;
import math, sqlite3;
import pandas;
from compare_db import compare_db;
from utils import theta0topi, rad_to_deg, deg_to_rad, diff_angle;



def labelcorrection(
        dbname_out, compare_outname, 
        dbnames, tablenames, columns, selections,
        isCorrectHWPenc, isHWPSS=False):
    mode = 'wiregrid' if not isHWPSS else 'HWPSS';

    # variable setting
    primarycolumn = 'readout_name'; # target column in merging
    suffixes=[
            '', # NEED TO BE EMPTY (No suffix)
            '_fix',
            ];
    varnames=[
            'pixel_name', # compare column
            ];
    if len(varnames)==1 : varnames = [varnames[0] for i in range(len(dbnames))];
    dropNan=[
            # wiregrid DB
            False,
            # planet DB
            True,
            ];

    # compare DBs
    df0s, df, dfmislabel = compare_db(
            dbnames    = dbnames,
            tablenames = tablenames,
            columns    = columns,
            varnames   = varnames,
            selections = selections,
            suffixes   = suffixes,
            dropNan    = dropNan,
            primarycolumn=primarycolumn,
            outname=compare_outname
            );

    #print(df);
    #print(dfmislabel);
    nsuccess = 0;
    nfails = [0,0,0,0];
    nwarns = [0,0,0,0,0,0,0];
    nwarns_angle= [0,0,0,0,0];
    names_fail = [];
    dfmislabel = dfmislabel.reset_index();
    df_new = df0s[0];

    # Check bolo numbers
    ntotal = len(df_new);
    nnanReadout = sum(df_new['readout_name'].isnull());
    nmislabel = len(dfmislabel);
    ncorrectlabel = ntotal - nnanReadout - nmislabel;

    # add mislabel column
    df_new['mislabel']=False;
    df_new['isCorrectLabel']=True;
    # remove det_offset_x/y in df_new
    df_new = df_new.drop('det_offset_x', axis=1);
    df_new = df_new.drop('det_offset_y', axis=1);
    # add det_offset_x/y from Kyohei's DB
    #print('before add', df_new['det_offset_x']);
    print('New DB keys before add det_offset_x/y:', df_new.columns.values);
    print('df0s[1] keys:', df0s[1].columns.values)
    df_new = pandas.merge(df_new, df0s[1][['det_offset_x','det_offset_y',primarycolumn]], how='left', on=primarycolumn);
    # copy "band" from df0s[1]
    for i in range(len(df_new)):
        readout_name = df_new.loc[i,'readout_name'];
        band = df_new.loc[i,'band'];
        band_fix = df[df['readout_name']==readout_name]['band_fix']
        if len(band_fix)!=1:
            print('Warning! There is no correct reference DB for {} (size={})!'.format(readout_name, len(band_fix)));
            continue;
        band_fix = band_fix.reset_index().loc[0, 'band_fix'];
        print('band_fix = {}'.format(band_fix));
        if band!=band_fix:
            print('New DB has different band from the reference DB:');
            print('     band is changed! {} --> {}'.format(band, band_fix));
            df_new.loc[i, 'band'] = band_fix
            pass;
        pass;
    print('New DB after add:');
    print(df_new['det_offset_x']);
    print('New DB keys after add:', df_new.columns.values);

    ###################################
    # Loop over mislabeled bolometers #
    ###################################

    for i in range(len(dfmislabel)) :
        print('******** i={} *********'.format(i));
        name          = dfmislabel.at[i,'readout_name'];
        # initialize variables
        pixel_name_fix = '';
        pixel_number_fix = -1;
        band_fix       = -1;
        pixel_type_fix = '';
        bolo_type_fix  = '';
        bolo_name_fix  = '';
        pixel_handedness_fix = '';
        pol_angle_fix  = -1;
        success = False;
        if name is None :
            print('WARNING! readout_name is None (reaout_name={}).'.format(name));
            names_fail.append(name);
            nfails[0]+=1;
        else :
            # pixel_name
            pixel_name_fix= dfmislabel.at[i,'pixel_name_fix']; # Correct pixel name is obtained from Kyohei's DB.
            if pixel_name_fix!=pixel_name_fix : # if NaN
                print('WARNING! Could not find correct pixel data for {} (No pixel_name_fix).'.format(name));
                names_fail.append(name);
                nfails[1]+=1;
            else :
                # Get correct data
                querycmd = "pixel_name=='{}'".format(pixel_name_fix);
                print('Getting correct pixel name DBs: querycmd = {}'.format(querycmd));
                correct_info  = df.query(querycmd);
                #print(correct_info);
                if len(correct_info)==0:
                    print('WARNING! Could not find correct pixel data for {} (No pixel_name={}).'.format(name,pixel_name_fix));
                    names_fail.append(name);
                    nfails[2]+=1;
                else :
                    # pixel_number (from Kyohei's DB)
                    pixel_number_fix = dfmislabel.at[i,'pixel_number_fix'];
                    if pixel_number_fix!=pixel_number_fix:
                        print('WARNING! {} has NaN band!'.format(name));
                        nwarns[6]+=1;
                        pass;
                    # band (from Kyohei's DB)
                    band_fix = dfmislabel.loc[i,'band_fix'];
                    if band_fix!=band_fix or band_fix<=0:
                        print('WARNING! {} has NaN band!'.format(name));
                        nwarns[0]+=1;
                        pass;
                    print('band = {}'.format(band_fix));
                    # pixel_type
                    pixel_types_fix = np.array(correct_info['pixel_type']);
                    if np.all(pixel_types_fix==pixel_types_fix[0]) :
                        print('pixel_type : all same = {}'.format(pixel_types_fix[0]));
                        pixel_type_fix0 = dfmislabel.at[i,'pixel_type_fix'];
                        # Ignore this check if pixel_type_fix0 is None.
                        if pixel_types_fix[0]==pixel_type_fix0 or pixel_type_fix0 is None: 
                            pixel_type_fix = pixel_types_fix[0];
                        else :
                            print('pixel_type : different from the reference DB = {}'.format(pixel_type_fix0));
                            nwarns[1]+=1;
                            pass;
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
                        pixel_handedness_fix0 = dfmislabel.at[i,'pixel_handedness_fix'];
                        # Ignore this check if pixel_handedness_fix0 is None.
                        if pixel_handednesses_fix[0]==pixel_handedness_fix0 or pixel_handedness_fix0 is None:
                            pixel_handedness_fix = pixel_handednesses_fix[0];
                        else:
                            print('pixel_handedness : different from the reference DB = {}'.format(pixel_handedness_fix0));
                            nwarns[3]+=1;
                            pass;
                    else :
                        print('pixel_handedness : different = {}'.format(pixel_handednesses_fix));
                        nwarns[3]+=1;
                        pass;
                    # bolo_name
                    bolo_names_fix = np.array(correct_info['bolo_name']);
                    if np.all(bolo_names_fix==bolo_names_fix[0]) :
                        print('bolo_name : all same = {}'.format(bolo_names_fix[0]));
                        bolo_name_fix = bolo_names_fix[0];
                    else :
                        print('bolo_name : different = {}'.format(bolo_names_fix));
                        nwarns[4]+=1;
                        pass;
                    # pol_angle
                    pol_angles_fix = np.array(correct_info['pol_angle']);
                    if np.all(pol_angles_fix==pol_angles_fix[0]) :
                        print('pol_angle : all same = {}'.format(pol_angles_fix[0]));
                        pol_angle_fix = pol_angles_fix[0];
                    else :
                        print('pol_angle : different = {} [deg]'.format(pol_angles_fix));
                        nwarns[5]+=1;
                        theta_det = dfmislabel.at[i,'theta_det'];
                        print('theta_det = {} [rad]'.format(theta_det));
                        if theta_det!=theta_det :
                            nwarns_angle[0]+=1;
                            print('pol_angle : No theta_det = {}'.format(theta_det));
                        else :
                            if not isHWPSS:
                                if isCorrectHWPenc:
                                    calib_angle = theta0topi(theta_det - np.pi/2. + 2.*deg_to_rad(-16.71)); 
                                    # -16.71 is obtained from HWP offset angle in out_check_HWPzeroangle/check_HWPzeroangle_ver9.out 
                                else :
                                    #calib_angle = rad_to_deg(theta0topi(theta_det - np.pi/2.));
                                    calib_angle = theta0topi(theta_det - np.pi/2.);
                                    pass;
                            else:
                                if isCorrectHWPenc:
                                    calib_angle = theta0topi(theta_det - np.pi/2. + deg_to_rad(-16.71-45.)); 
                                else :
                                    calib_angle = theta0topi(-theta_det - np.pi/2.);
                                    pass;
                                pass;
                            print('calib_angle = {} [rad]'.format(calib_angle));
                            diff_angles = rad_to_deg(diff_angle(deg_to_rad(pol_angles_fix), calib_angle, upper90deg=True));
                            # change nan --> 9999.
                            diff_angles = np.where(np.isnan(diff_angles), 9999., diff_angles);
                            print('diff_angles btw calib_angle and pol_angle = {} [deg]'.format(diff_angles));
                            mindiff = min(diff_angles);
                            index = np.where(diff_angles == mindiff)[0];
                            # pol_angle
                            print('minimum diff_angle index = {}'.format(index));
                            pol_angles_fix2 = [pol_angles_fix[i] for i in index];
                            #print(pol_angles_fix2);
                            if np.all(pol_angles_fix2==pol_angles_fix2[0]):
                                print('pol_angle after pol_angle selection : all same = {}'.format(pol_angles_fix2[0]));
                                pol_angle_fix = pol_angles_fix2[0];
                            else :
                                print('pol_angle after pol_anlge selection : different = {}'.format(pol_angles_fix2));
                                nwarns_angle[1]+=1;
                                pass;
                            # bolo_type
                            bolo_types_fix2 = np.array([ bolo_types_fix[i] for i in index ]);
                            # remove nan
                            bolo_types_fix2 = bolo_types_fix2[bolo_types_fix2==bolo_types_fix2];
                            if len(bolo_types_fix2)>0 and np.all(bolo_types_fix2==bolo_types_fix2[0]):
                                print('bolo_type after pol_angle selection : all same = {}'.format(bolo_types_fix2[0]));
                                bolo_type_fix = bolo_types_fix2[0];
                            else :
                                print('bolo_type after pol_angle selection : different = {}'.format(bolo_types_fix2));
                                nwarns_angle[3]+=1;
                                pass;
                            # bolo_name
                            bolo_names_fix2 = np.array([ bolo_names_fix[i] for i in index ]);
                            # remove nan
                            bolo_names_fix2 = bolo_names_fix2[bolo_names_fix2==bolo_names_fix2];
                            if len(bolo_names_fix2)>0 and np.all(bolo_names_fix2==bolo_names_fix2[0]):
                                print('bolo_name after pol_angle selection : all same = {}'.format(bolo_names_fix2[0]));
                                bolo_name_fix = bolo_names_fix2[0];
                                pass;
                            else :
                                print('bolo_name after pol_angle selection : different = {}'.format(bolo_names_fix2));
                                band_in_boloname = np.array([ (int)(name.split('.')[2][:-1]) for name in bolo_names_fix2 ]);
                                is_correct_band = (band_in_boloname==(int)(band_fix));
                                correct_bolonames = bolo_names_fix2[is_correct_band];
                                print('bolo_name after band selection (={}) : candidates = {}'.format(band_fix, correct_bolonames));
                                if len(correct_bolonames)==1 :
                                    bolo_name_fix = correct_bolonames[0];
                                else:
                                    print('Failed to get correct bolo_name');
                                    nwarns_angle[4]+=1;
                                    pass;
                                pass;
                            pass;
                        pass;
                    # check bolo_name band / modify bolo_name
                    bolo_name_fix_parts = bolo_name_fix.split('.');
                    if len(bolo_name_fix)>0:
                        band_in_boloname_fix = (int)(bolo_name_fix_parts[2][:-1]);
                        if band_in_boloname_fix==(int)(band_fix):
                            print('band in bolo_name is correct!: {}'.format(bolo_name_fix));
                        else:
                            print('band in bolo_name is wrong!: {}'.format(bolo_name_fix));
                            bolo_name_fix = '{}.{}.{}{}'.format(
                                    bolo_name_fix_parts[0], bolo_name_fix_parts[1], (int)(band_fix), bolo_name_fix_parts[2][-1])
                            print('bolo_name is changed! --> {}'.format(bolo_name_fix));
                            pass;
                    # Check if the correct label is obtained or not
                    if  band_fix>0 and \
                        len(pixel_name_fix)>0 and \
                        len(pixel_type_fix)>0 and \
                        pixel_number_fix>0 and \
                        len(bolo_type_fix)>0 and \
                        len(bolo_name_fix)>0 and \
                        len(pixel_handedness_fix)>0 and \
                        pol_angle_fix>0. :
                        nsuccess+= 1;
                        success = True;
                    else:
                        nfails[3]+=1;
                        names_fail.append(name);
                        pass;
                    pass;
                pass;
            pass;

        index = df_new.reset_index().query("readout_name=='{}'".format(name)).index;
        if len(index)>1 : 
            print('Error! There are multiple data for readout_name=={}'.format(name));
        else :
            index = index[0]; # list -> one value
            pass;
        print('Mislabel index for {} = {}'.format(name,index));
        print('   --> set mislabel = True');
        print('   pixel_name correction= {} --> {}'.format(df_new.at[index,'pixel_name'], pixel_name_fix));
        print('   pixel_number correction= {} --> {}'.format(df_new.at[index,'pixel_number'], pixel_number_fix));
        print('   band correction= {} --> {}'.format(df_new.at[index,'band'], band_fix));
        print('   bolo_name correction= {} --> {}'.format(df_new.at[index,'bolo_name'], bolo_name_fix));
        print('   pixel_type correction = {} --> {}'.format(df_new.at[index,'pixel_type'], pixel_type_fix));
        print('   bolo_type correction= {} --> {}'.format(df_new.at[index,'bolo_type'], bolo_type_fix));
        print('   pixel_handedness correction = {} --> {}'.format(df_new.at[index,'pixel_handedness'], pixel_handedness_fix));
        print('   pol_angle correction = {} --> {}'.format(df_new.at[index,'pol_angle'], pol_angle_fix));
        print('   Correction success (isCorrectLabel) = {}'.format(success));
        df_new.loc[index, 'pixel_name'] = pixel_name_fix;
        df_new.loc[index, 'pixel_number'] = pixel_number_fix;
        df_new.loc[index, 'band'      ] = band_fix;
        df_new.loc[index, 'pixel_type'] = pixel_type_fix;
        df_new.loc[index, 'bolo_name']  = bolo_name_fix;
        df_new.loc[index, 'bolo_type']  = bolo_type_fix;
        df_new.loc[index, 'pixel_handedness'] = pixel_handedness_fix;
        df_new.loc[index, 'pol_angle']  = pol_angle_fix;
        df_new.loc[index, 'mislabel' ]  = True;
        df_new.loc[index, 'isCorrectLabel']  = success;
        pass;
    # convert to integer: band, pixel_number
    df_new = df_new.astype({'band':int, 'pixel_number':int});
    
    print('');
    print('');
    print('### Summary ###');
    print('');
    print('# of bolos failing to get correct values for each data.');
    print('  multiple candidates of pixel_number     = {}'.format(nwarns[6]));
    print('  multiple candidates of band             = {}'.format(nwarns[0]));
    print('  multiple candidates or unmatched reference DB of pixel_type       = {}'.format(nwarns[1]));
    print('  multiple candidates or unmatched reference DB of pixel_handedness = {}'.format(nwarns[3]));
    print('  multiple candidates of bolo_type        = {} (should be determined by {} cal.)'.format(nwarns[2], mode));
    print('  multiple candidates of bolo_name        = {} (should be determined by {} cal.)'.format(nwarns[4], mode));
    print('  multiple candidates of pol_angle        = {}'.format(nwarns[5]));
    print('    No theta_det          = {}/{}'.format(nwarns_angle[0], nwarns[5]));
    print('    Failed for pol_angle  = {}/{}'.format(nwarns_angle[1], nwarns[5]));
    print('    Failed for bolo_type  = {}/{}'.format(nwarns_angle[3], nwarns[5]));
    print('    Failed for bolo_name  = {}/{}'.format(nwarns_angle[4], nwarns[5]));

    nfailtotal  = sum(nfails);
    print('# of total bolos={}'.format(len(df_new)));
    print('# of correct label bolos = total - Nan readout_name - failures = {}'.format(ntotal-nfailtotal-nnanReadout));
    print('  Correct numbers:');
    print('  # of originally correct bolos = {}'.format(ncorrectlabel));
    print('  # of mislabel-corrected bolos = {}'.format(nsuccess));
    print('  Not-correct numbers:');
    print('  # of NaN readout_name bolos    = {}'.format(nnanReadout));
    print('  # of sum of correction failurs = {}'.format(nfailtotal));
    print('    Failure 0 (readout_name=None ) ={}'.format(nfails[0]));
    print('    Failure 1 (pixel_name_fix=NaN) ={}'.format(nfails[1]));
    print('    Failure 2 (No pixel_name_fix ) ={}'.format(nfails[2]));
    print('    Failure 3 (Could not identify the correct label by {} cal.) ={}'.format(nfails[3], mode));
    #for name in names_fail :
    #    print(name);
    #    pass;

    #################
    # Modify df_new #
    #################

    print('Size of the new DB = {}'.format(len(df_new)));
    # Drop NaN bolometers in readout_name
    print('Drop readout_name==NaN bolometers');
    df_new1 = df_new.dropna(subset=['readout_name']);
    print('Size of dropped DB by readout_name = {}'.format(len(df_new)-len(df_new1)));
    # Drop NaN bolometers in theta_det
    print('Drop theta_det==NaN bolometers (No wiregrid data)');
    df_new2 = df_new1.dropna(subset=['theta_det']);
    print('Size of dropped DB by theta_det = {}'.format(len(df_new1)-len(df_new2)));

    df_new = df_new2;
    print('Size of new DB = {}'.format(len(df_new)));

    ###############
    # Save df_new #
    ###############

    # Save to pickle file
    outputname = dbname_out + '.pkl';
    print('Saving the pandas to a pickle file ({})...'.format(outputname));
    df_new.to_pickle(outputname);
    # Save to sqlite3 file
    outputname = dbname_out + '.db';
    print('Saving the pandas to a sqlite3 file ({})...'.format(outputname));
    conn = sqlite3.connect(outputname);
    df_new.to_sql('wiregrid' if not isHWPSS else 'hwpss',conn,if_exists='replace',index=None);
    conn.close();

    return 0;


if __name__=='__main__' :

    ver='ver10'
    isCorrectHWPenc=True;
    dbname_out = 'output_{}/db/all_pandas_correct_label'.format(ver)
    dbnames =[
            # wiregrid DB
            'output_{}/db/all_pandas.db'.format(ver), 

            # planet DB
            # To get offset_det_x/y, band, pixel_type, bolo_type,pixel_handedness fixed by plant observations
            #'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            'data/ykyohei/mapping/pb2a_mapping_postv4.db', # ver10
            ];
    tablenames=[
            # wiregrid DB
            'wiregrid',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            #'pb2a_focalplane',
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            'pb2a_focalplane',
            ];
    columns=[
            # wiregrid DB
            '*',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            #'pixel_name,band,det_offset_x,det_offset_y,hardware_map_dir,hardware_map_commit_hash',
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            'pixel_name,pixel_number,bolo_name,band,pixel_type,bolo_type,pixel_handedness,det_offset_x,det_offset_y,hardware_map_dir,hardware_map_commit_hash',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    selections=[
            # wiregrid DB
            '',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv2.db',
            #"hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            "hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            ];

    compare_outname='output_{}/db/compare_db'.format(ver);

    labelcorrection(
        dbname_out, compare_outname, 
        dbnames, tablenames, columns, selections,
        isCorrectHWPenc, isHWPSS=False);
    pass;
