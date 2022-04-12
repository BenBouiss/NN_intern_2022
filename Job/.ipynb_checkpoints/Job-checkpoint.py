#OAR -n Nightly training
#OAR -l /nodes=1/core=6,walltime=04:00:30
#OAR --project mais
#OAR --stdout Test.out
#OAR --stderr Test.err
cd ..
conda activate py37
python Train_big.py
