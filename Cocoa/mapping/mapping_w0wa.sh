for i in {0..35}
do
    mpirun -n 1 --cpu-set $i --bind-to core $CONDA_PREFIX/bin/python3 ./mapping/mapping_w0wa.py 36 $i &
done