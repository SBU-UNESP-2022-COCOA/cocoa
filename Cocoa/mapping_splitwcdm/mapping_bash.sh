for i in {0..21}
do
    mpirun -n 1 --cpu-set $i --bind-to core $CONDA_PREFIX/bin/python3 ./mapping_splitwcdm/mapping_w0wa.py 22 $i &
done