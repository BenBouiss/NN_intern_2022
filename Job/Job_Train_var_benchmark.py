#OAR -n Mid_day_benchmark
#OAR -l /nodes=1/core=10,walltime=48:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke60'
#OAR --stdout Logs/Train.out
#OAR --stderr Logs/Train.err
cd ..
cd Job_script
conda activate py37
python Train_var_benchmark.py