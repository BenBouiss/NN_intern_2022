#OAR -n Mid_day_benchmark
#OAR -l /nodes=1/core=10,walltime=48:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke60'
#OAR --stdout Logs/Train_architecture_relu_swish.out
#OAR --stderr Logs/Train_architecture_relu_swish.err
cd ..
conda activate py37
python Train_job.py