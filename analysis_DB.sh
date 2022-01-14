#!/bin/bash
source env-shell.sh

############
# Wiregrid #
############

#VER='ver9'
#VER='ver10'
VER='ver10.2'
isCorrectHWPenc=1

# 1. check bsub jobs
#python3 check_jobs.py ${VER} 2>&1 | tee output_${VER}/check_jobs.out

# 2. merge DBs (NOTE: need to modify script)
#python3 merge.py 2>&1 | tee output_${VER}/db/merge.out

# 3.1 check the merged DB
#python3 checkDB.py ${VER} 2>&1 | tee output_${VER}/db/checkDB.out

# 3.2 make histograms of the merged DB
#python3 makehist.py ${VER} 2>&1 | tee output_${VER}/db/makehist.out

# 4. correct label (NOTE: need to modify script)
#python3 compare_DB_for_labelcorrection.py 2>&1 | tee output_${VER}/db/compare_DB_for_labelcorrection.out

# 5. check absolute angles (difference btw measured angles and design angles)
#python3 check_absolute.py ${VER} ${isCorrectHWPenc} "" "ploterr,eachwire" 2>&1 | tee output_${VER}/check_absolute.out
# For HWP optical axis check
#python3 check_absolute.py ${VER} 0 _noOffsetSubtract 2>&1 | tee output_${VER}/check_absolute_noOffsetSubtract.out
# For check before no label correction
#python3 check_absolute_nocorr.py ${VER}  ${isCorrectHWPenc} 'eachwire' 2>&1 | tee output_${VER}/check_absolute_nocorr.out
# For wobble check
#python3 check_absolute_eachwafer.py ${VER} ${isCorrectHWPenc} 2>&1 | tee output_${VER}/check_absolute_eachwafer.out

# 6. modify DB for public
#python3 modifyDB.py ${VER} 2>&1 | tee output_${VER}/modify.out



#########
# HWPSS #
#########

HWPSS_VER='ver3'
HWPSS_OUTDIR="hwpss/output_${HWPSS_VER}"

#<<'#__COMMENT__'
if [ ! -e ${HWPSS_OUTDIR} ]; then
    mkdir ${HWPSS_OUTDIR}
fi

# 1. merge DBs (NOTE: need to modify script)
#python3 hwpss/hwpss_mergeDB.py 2>&1 | tee ${HWPSS_OUTDIR}/mergeDB.out

# 2. correct label (NOTE: need to modify script)
#python3 hwpss/hwpss_compare_DB_for_labelcorrection.py 2>&1 | tee ${HWPSS_OUTDIR}/compare_DB_for_labelcorrection.out

# 3. check absolute angles (difference btw measured angles and design angles)
# For check after label correction
python3 hwpss/hwpss_check_absolute.py ${HWPSS_VER} ${isCorrectHWPenc} "" "" 2>&1 | tee ${HWPSS_OUTDIR}/check_absolute.out
# For check before no label correction
python3 hwpss/hwpss_check_absolute_nocorr.py ${HWPSS_VER}  ${isCorrectHWPenc} '' 2>&1 | tee ${HWPSS_OUTDIR}/check_absolute_nocorr.out

# 4. modify DB for public
python3 hwpss/hwpss_modifyDB.py ${HWPSS_VER} 2>&1 | tee ${HWPSS_OUTDIR}/modify.out

#__COMMENT__



