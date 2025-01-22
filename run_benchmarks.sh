#!/bin/bash
cd spmv; bash run_spmv.sh; cd -
cd spgemm; bash run_spgemm.sh; cd -
cd graphs; bash run_graphs.sh; cd -
cd images; bash run_morphology.sh; cd -