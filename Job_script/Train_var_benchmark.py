import sys
sys.path.append('../')
import os

from Scripts import Trainings
from Scripts import Plotting
from Scripts import Computing_functions
#import os
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
Var_X_BIG_Extra = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']
CPLs_test = ['CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']
#Drops = [0,4, 0.2]
#for i in range(2):

#class model_NN():
#def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS',
#Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', 
#Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5,
#Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8,
#Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, 
#Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2):


## Submit job oarsub -S ./Job.py
i = 0
#Uniq_var = ['iceDraft', 'bathymetry', 'Slope_iceDraft', 'Slope_bathymetry',
#                'Big_T', 'Big_S', 'Distances_ground_line', 'Distances_front_line']
Uniq_var = ['iceDraft', 'bathymetry', 'Slope_iceDraft', 'Slope_bathymetry',
                'Big_T', 'Big_S', 'Distances_ground_line', 'Distances_front_line', ['Big_T', 'Big_S']]
for var in Uniq_var[2:]:
    li = list(Var_X_BIG_Extra)
    for v in li:
        if type(var) != list:
            if var in v and 'Slope' in var:
                li.remove(v)
        else:
            for Drop in var:
                if Drop in li:
                    li.remove(Drop)
            
    if 'Slope' not in var and type(var) != list:
        li.remove(var)
    print(li, len(li))
    
    Training = Trainings.Sequencial_training(Trainings.model_NN)
    #Best_Neur = ['96_96_96_96_96'] #, '96_96_96_96_96', '64_64_64_96_96', '32_32_32_64']
    Best_Neur = ['128_128_128_128_128']
    Training.training(training_extent = 0, verbose = 0, batch_size = 1024, Exact = 1, message = 1,
                Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 128, 
                Var_X = li, Verify = 0, Extra_n = 'Drop_Var_bench_128', Fraction_save = 100,
                Similar_training = True, Norm_Choix = 0, Method_data = 4,activ_fct= "swish", 
                Scaling_lr = True, TensorBoard_logs = True , Hybrid = False, Scaling_type = 'Plateau', 
                LR_Patience = 8, LR_min = 0.0000016, LR_Factor = 2, min_delta = 0.0001)
