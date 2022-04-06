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
Var_X1 = ['x', 'y', 'thermalDriving', 'halineDriving', 'iceDraft']
Var_X2 = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft']
Var_X_non_position = ['temperatureYZ', 'salinityYZ', 'iceDraft']
Var_X_BIG = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'Big_T', 'Big_S']
Var_X_Extra = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'temperatureYZ', 'salinityYZ']
Var_X_BIG_test = ['iceDraft','Big_T', 'Big_S']
Var_X_BIG_Extra = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']

for _ in range(2):
    Training = Trainings.Sequencial_training(Trainings.model_NN)
    Best_Neur = ['96_96_96_96_96'] #, '96_96_96_96_96', '64_64_64_96_96', '32_32_32_64']
    Training.training(training_extent = 0, verbose = 1, batch_size = 1024, Exact = 1, message = 1,
                Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 30, 
                Var_X = Var_X_BIG_Extra, Verify = 0, Extra_n = 'Same_ind', 
                Similar_training = 1, Norm_Choix = 0, Method_data = 4, 
                Scaling_lr = True, Frequence_scaling_change = 8, Scaling_change = 5, TensorBoard_logs = True)