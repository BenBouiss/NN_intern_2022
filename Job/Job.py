#OAR -n Nightly_training
#OAR -l /nodes=1/core=11,walltime=18:00:00 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke60'
#OAR --stdout Test.out
#OAR --stderr Test.err
cd ..
conda activate py37
python Train_big.py
