#OAR -n Mid_day_benchmark
#OAR -l /nodes=1/core=8,walltime=120:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke62'
#OAR --stdout Logs/Train.out
#OAR --stderr Logs/Train.err
cd ..
cd Job_script
conda activate py37
python Train_big.py