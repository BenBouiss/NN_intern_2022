import sys
sys.path.append('../')
from Scripts import Trainings
from Scripts import Plotting
from Scripts import Computing_functions
import os
import importlib
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

OcT = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4']
OcTPlus = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst']
Var_X1 = ['x', 'y', 'thermalDriving', 'halineDriving', 'iceDraft']
Var_X2 = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft']
Var_X_non_position = ['temperatureYZ', 'salinityYZ', 'iceDraft']
Var_X_BIG = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'Big_T', 'Big_S']
Var_X_Extra = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'temperatureYZ', 'salinityYZ']
Var_X_BIG_test = ['iceDraft','Big_T', 'Big_S']
Var_X_BIG_TS = ['Big_T', 'Big_S']

Var_X_slopexy = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'temperatureYZ', 'salinityYZ',
                  'Distances_ground_line', 'Distances_front_line']

Var_X_BIG_Extra = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']
Composite = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']

#CPLs_test = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst']
#Drops = [0,4, 0.2]
#for i in range(2):

#class model_NN():
#def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5, Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8, Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, Fraction = None, Fraction_save = None,
#Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2, min_delta = 0.007, Pruning = False, Pruning_type = 'Constant', initial_sparsity = 0, target_sparsity = 0.5, Random_seed = None):
## Submit job oarsub -S ./Job.py
## Submit this particular job : oarsub -S ./Job_for_Train_big.py


#Best_Neur = ['96_96_96_96_96'] #, '96_96_96_96_96', '64_64_64_96_96', '32_32_32_64']
#Best_Neur = ['0'] 
#Best_Neur = ['128_128_128_128_128']
#Best_Neur = ['32_32_96_96']
# Training.training(training_extent = 0, verbose = 1, batch_size = 1028, Exact = 1, message = 1,
#             Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 128, 
#             Var_X = Var_X_BIG_Extra, Verify = 0, Better_cutting = '10%', Extra_n = '10percent',
#             Similar_training = False, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
#             Scaling_lr = True, Fraction_save = 10)

prunes = (1 - np.around(np.linspace(1, 10, 16, endpoint = False), decimals = 2) ** 2 / 100)[::-1]

Best_Neur = ['128_128_128_128_128']
Frac = (np.around(np.linspace(1, 10, 16, endpoint = False), decimals = 2) ** 2 / 100)[::-1]
Training = Trainings.Sequencial_training(Trainings.model_NN)
#for pr in prunes:
for _ in range(1):
    Training = Trainings.Sequencial_training(Trainings.model_NN)
    #for F in Frac:
#    for Batch in [64, 128, 1024, 2048]:
        ### Batch size benchmark
#        Training.training(verbose = 0, batch_size = Batch, message = 1,
#              Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64,
#              Var_X = Var_X_BIG_Extra, Extra_n = 'Batch_Size_Benchmark',
#              Similar_training = True, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
#              Scaling_lr = True, Fraction_save = 32, Scaling_type = 'Plateau', LR_min = 0.0000016, LR_Patience = 8, LR_Factor #= 2)


 ### Pruning benchmark composite
    for pr in prunes:
        Training = Trainings.Sequencial_training(Trainings.model_NN)
        Training.training(verbose = 0, batch_size = 1028, message = 1,
              Standard_train = Best_Neur, Dataset_train = Composite, Epoch = 64,
              Var_X = Var_X_BIG_Extra, Extra_n = 'Prune_benchmark_Composite',
              Similar_training = False, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
              Scaling_lr = True, Fraction_save = 32, Scaling_type = 'Plateau', LR_min = 0.0000016, LR_Patience = 8, LR_Factor = 2, Pruning = True, Pruning_type = 'Constant', target_sparsity = pr, Fraction = 0.6)

#    Training.training(verbose = 0, batch_size = 1028, message = 1,
#              Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64,
#              Var_X = Var_X_BIG_Extra, Extra_n = 'Seed_Benchmark', Random_seed = 123456,
#              Similar_training = True, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
#              Scaling_lr = True, Fraction_save = 32, Scaling_type = 'Plateau', LR_min = 0.0000016, LR_Patience = 8, LR_Factor = 2)
        # Training.training(verbose = 1, batch_size = 1028, message = 1,
        #       Standard_train = Best_Neur, Dataset_train = Composite, Epoch = 64, Fraction = 0.6,
        #       Var_X = Var_X_BIG_Extra, Extra_n = 'Complexity_downgrade_composite',
        #       Similar_training = False, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
        #       Scaling_lr = True, Fraction_save = 32, Scaling_type = 'Plateau', LR_min = 0.0000016, LR_Patience = 8, LR_Factor = 2)
        
        ### Fraction benchmark
        # Training.training(verbose = 0, batch_size = 1028, message = 1,
        #       Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64, Fraction = F,
        #       Var_X = Var_X_BIG_Extra, Extra_n = 'Fraction_dataset_Benchmark',
        #       Similar_training = True, Norm_Choix = 0, Method_data = 4, activ_fct= "swish", 
        #       Scaling_lr = True, Fraction_save = 32, Scaling_type = 'Plateau', LR_min = 0.0000016, LR_Patience = 8, LR_Factor = 2)