from compare_DB_for_labelcorrection import labelcorrection;

if __name__=='__main__' :

    ver='3'
    isCorrectHWPenc=True;
    dbname_out = 'hwpss/output_ver{}/db/all_pandas_correct_label'.format(ver)
    dbnames =[
            # hwpss DB
            'hwpss/output_ver{}/db/all_pandas.db'.format(ver, ver), 

            # planet DB
            # To get offset_det_x/y, band, pixel_type, bolo_type,pixel_handedness fixed by plant observations
            'data/ykyohei/mapping/pb2a_mapping_postv4.db', # ver10
            ];
    tablenames=[
            # hwpss DB
            'hwpss',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            'pb2a_focalplane',
            ];
    columns=[
            # hwpss DB
            '*',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            'pixel_name,pixel_number,bolo_name,band,pixel_type,bolo_type,pixel_handedness,det_offset_x,det_offset_y,hardware_map_dir,hardware_map_commit_hash',
            ];
    if len(columns)==1 : columns = [columns[0] for i in range(len(dbnames))];
    selections=[
            # hwpss DB
            'entry>0',

            # planet DB
            # For 'data/ykyohei/mapping/pb2a_mapping_postv4.db',
            "hardware_map_commit_hash=='6f306f8261c2be68bc167e2375ddefdec1b247a2'",
            ];

    compare_outname='hwpss/output_ver{}/db/compare_db'.format(ver);

    labelcorrection(
        dbname_out, compare_outname, 
        dbnames, tablenames, columns, selections,
        isCorrectHWPenc, isHWPSS=True);
    pass;
