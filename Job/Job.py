#OAR -n Nightly training
#OAR -l /nodes=1/core=6,walltime=07:00:30 
#OAR --project pr-ice_speed
#OAR --property team='ige'
#OAR --property network_address='luke62'
#OAR --stdout Test.out
#OAR --stderr Test.err
cd ..
conda activate py37
python Train_big.py
