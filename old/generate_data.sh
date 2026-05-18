cd graphs; wget https://nrvis.com/download/data/soc/soc-orkut.zip -O soc-orkut.zip; unzip soc-orkut.zip; cd -

cd spgemm; julia make_scale.jl; cd -

cd graphs; ./rmat_gen 22 64; cd -
cd graphs; ./rmat_gen 23 32; cd -
cd graphs; ./rmat_gen 24 16; cd -
