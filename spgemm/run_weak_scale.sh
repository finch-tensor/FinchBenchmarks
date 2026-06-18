t=1
OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w1 --kernel fast --output spgemm_scale_threads_weak_$t.json

t=2
OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w2 --kernel fast --output spgemm_scale_threads_weak_$t.json

t=4
OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w3 --kernel fast --output spgemm_scale_threads_weak_$t.json

t=8
OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w4 --kernel fast --output spgemm_scale_threads_weak_$t.json

t=16
OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t julia -t $t run_spgemm.jl --dataset w5 --kernel fast --output spgemm_scale_threads_weak_$t.json
