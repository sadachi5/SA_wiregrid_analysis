module load gcc/930
module load openmpi/2.1.6-gcc930
module load intel/2020
module load git
PREFIX=/sw/cmb/polarbear/opt-202008
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64:$LD_LIBRARY_PATH

# For simons_array_offline_software
export PYTHONPATH=$PYTHONPATH:$PWD/library/simons_array_offline_software:$PWD/library/simons_array_offline_software/kms_git/kms_plot
