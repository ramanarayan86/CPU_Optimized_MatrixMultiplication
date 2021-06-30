### #!/bin/bash
set -v

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export KMP_AFFINITY=compact,granularity=fine,1,0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../build/

source /swtools/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64


echo "compiling ....."



icpc -o mm_flt matMul.cpp counters.c --std=c++11 -lmkl_intel_lp64 -lmkl_core -lm -liomp5 -mkl -DVTUNE_ANALYSIS -I/swtools/intel/vtune_amplifier/include/ -littnotify -L/swtools/intel/vtune_amplifier/lib64/ -qopenmp 



echo "Running ..."

## If the M = N = K
numactl -N 0 -m 0 ./mm_flt $1 $1 $1 

## If the M != N != K
# numactl -N 0 -m 0 ./mm_flt $1 $2 $3             
