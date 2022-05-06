#OAR -n Mid_day_benchmark
#OAR -l /nodes=1/core=5,walltime=48:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke60'
#OAR --stdout Train_output.out
#OAR --stderr Train_error.err
cd ..
conda activate py37
python Job_benchmark.py