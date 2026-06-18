for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w5 --kernel fast --output spgemm_scale_threads_$t.json
done

