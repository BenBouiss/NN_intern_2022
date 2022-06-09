#OAR -n Mid_day_training
#OAR -l /nodes=1/core=7,walltime=48:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke62'
#OAR --stdout Test.out
#OAR --stderr Test.err
cd ..
cd Job_script
conda activate py37
python Job_variable_benchmark.py
