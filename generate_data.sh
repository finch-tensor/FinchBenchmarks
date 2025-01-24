cd graphs; wget https://nrvis.com/download/data/soc/soc-orkut.zip -O soc-orkut.zip; unzip soc-orkut.zip; sed -i '1s/%MatrixMarket matrix coordinate pattern symmetric/%%MatrixMarket matrix coordinate pattern symmetric/' soc-orkut.mtx; mv soc-orkut.mtx soc-orkut; cd -

cd spgemm; julia make_scale.jl; cd -

cd graphs; ./rmat_gen 22 64; cd -
cd graphs; ./rmat_gen 23 32; cd -
cd graphs; ./rmat_gen 24 16; cd -
