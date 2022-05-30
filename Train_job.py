import sys
#sys.path.append('../')
import os
os.chdir('../')
from Scripts import Trainings
from Scripts import Plotting
from Scripts import Computing_functions
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
Var_X_non_position_driving = ['thermalDriving', 'halineDriving', 'iceDraft']
Var_X_non_position = ['temperatureYZ', 'salinityYZ', 'iceDraft']
Var_X = ['temperatureYZ', 'salinityYZ']

Var_X_BIG = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances_ground_line', 'Big_T', 'Big_S']
Var_X_Extra = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances_ground_line', 'temperatureYZ', 'salinityYZ']
Var_X_BIG_test = ['iceDraft','Big_T', 'Big_S']

Var_X_BIG_Step2 = ['iceDraft','Big_T', 'Big_S', 'Distances_ground_line', 'Distances_front_line']

Var_X_BIG_TS = ['Big_T', 'Big_S']

Var_X_slopexy = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'temperatureYZ', 'salinityYZ',
                  'Distances_ground_line', 'Distances_front_line']

Var_X_BIG_Slope = ['iceDraft', 'bathymetry', 'Slope_iceDraft', 'Slope_bathymetry','Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']

Var_X_BIG_Slopexy = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']
CPLs_test = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']

Only_CPLs = ['CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']
#class model_NN():
#def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS',
#Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', 
#Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5,
#Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8,
#Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, 
#Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2):
## Submit job oarsub -S ./Job.py


Training = Trainings.Sequencial_training(Trainings.model_NN)
#Best_Neur = ['96_96_96_96_96'] #, '96_96_96_96_96', '64_64_64_96_96', '32_32_32_64']
#Best_Neur = ['128_128_128_128_128_128']

Neur_bench = ['1', '8', '4_4', '16_16', '32_32', '16_16_16', '32_32_32', '64_64_64', '64_64_64_64', '96_96_96_96', '96_96_96_96_96', '128_128_128_128_128']
Neur_bench = ['8']
Activ_fct = ['relu', 'swish']
Activ_fct = ['swish']
#Best_Neur = ['128_128_128_128_128']
Best_Neur = ['32_32_96_96']


#Best_Neur = ['32_64_32']
#for act in Activ_fct:
#def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5, Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8, Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, Fraction = 1, Fraction_save = None,
#Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2, min_delta = 0.007):
#New_oc = [['Ocean1', 'Ocean2'], ['Ocean3', 'Ocean4']]

#for oc in New_oc:
#Fractions = 1 - np.arange(0, 100, 5)/100
#for Fract in Fractions:
# for v in Vars:
#     if v == Var_X_non_position:
#         Method = 3
#     else:
#         Method = 4   
#Vars = [Var_X_BIG_Slope, Var_X_BIG_Slopexy]

#CPLs_test = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP11_rst','CPL_EXP12_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']

#Only_CPLs = ['CPL_EXP11_rst','CPL_EXP12_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']

#Best_Neur = ['32_64_32']
#Best_Neur = ['128_128_128_128_128_128']

for i in range(2):
    print(f'Init training {i}')
    Training.training(training_extent = 0, verbose = 1, batch_size = 128, Exact = 1, message = 1,
            Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 32, 
            Var_X = Var_X_BIG_Slopexy, Verify = 0, Extra_n = 'Varxy',
            Similar_training = True, Norm_Choix = 0, Method_data = 4, 
            Fraction_save = 1, LR_min = 4*10**(-5),
            Scaling_lr = False)


# Training.training(training_extent = 0, verbose = 0, batch_size = 2048, Exact = 1, message = 1,
#             Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64, 
#             Var_X = Var_X_BIG_Slopexy, Verify = 0, Extra_n = 'Scalinglrforbatch2048',
#             Similar_training = True, Norm_Choix = 0, Method_data = 4, 
#             Fraction_save = 50, LR_min = 4*10**(-5),
#             Scaling_lr = False)

# Training.training(training_extent = 0, verbose = 0, batch_size = 1024, Exact = 1, message = 1,
#             Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64, 
#             Var_X = Var_X_BIG_Slopexy, Verify = 0, Extra_n = 'Scalinglrforbatch1024',
#             Similar_training = True, Norm_Choix = 0, Method_data = 4, 
#             Fraction_save = 50, LR_min = 4*10**(-5),
#             Scaling_lr = False)

# Training.training(training_extent = 0, verbose = 0, batch_size = 128, Exact = 1, message = 1,
#             Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 64, 
#             Var_X = Var_X_BIG_Slopexy, Verify = 0, Extra_n = 'Scalinglrforbatch128',
#             Similar_training = True, Norm_Choix = 0, Method_data = 4, 
#             Fraction_save = 50, LR_min = 4*10**(-5),
#             Scaling_lr = False)