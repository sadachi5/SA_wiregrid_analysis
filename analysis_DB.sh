#!/bin/bash
source env-shell.sh
#VER='ver9'
VER='ver10'
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
#python3 check_absolute.py ${VER} ${isCorrectHWPenc} "" ploterr 2>&1 | tee output_${VER}/check_absolute.out
# For HWP optical axis check
#python3 check_absolute.py ${VER} 0 _noOffsetSubtract 2>&1 | tee output_${VER}/check_absolute_noOffsetSubtract.out
# For check before no label correction
#python3 check_absolute_nocorr.py ${VER}  ${isCorrectHWPenc} 2>&1 | tee output_${VER}/check_absolute_nocorr.out
# For wobble check
#python3 check_absolute_eachwafer.py ${VER} ${isCorrectHWPenc} 2>&1 | tee output_${VER}/check_absolute_eachwafer.out

# 6. modify DB for public
python3 modifyDB.py ${VER} 2>&1 | tee output_${VER}/modify.out
