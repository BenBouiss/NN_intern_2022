#OAR -n Nightly_bench
#OAR -l /nodes=1/core=5,walltime=24:00:00 
#OAR --project pr-mais
#OAR --property team='ciment'
#OAR --property network_address='luke2'
#OAR --stdout Test_2.out
#OAR --stderr Test_2.err
cd ..
conda activate py37
python Job_benchmark.py
