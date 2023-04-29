parallel_job=35


for ((i=0; i<=parallel_job-1; i++));
do
    mpirun -n 1 --cpu-set $i --bind-to core $CONDA_PREFIX/bin/python3 ./mapping_splitwcdm/mapping_w2bin_RT.py $parallel_job $i &
done