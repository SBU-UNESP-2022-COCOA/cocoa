for i in {0..21}
do
    mpirun -n 1 --cpu-set $i --bind-to core $CONDA_PREFIX/bin/python3 ./mapping/mapping_w_2bin_RT.py 22 $i &
done