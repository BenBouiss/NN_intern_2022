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

importlib.reload(Trainings)
Training = Trainings.Sequencial_training(Trainings.model_NN)

Best_Neur = ['32_32_96_96'] #, '96_96_96_96_96', '64_64_64_96_96', '32_32_32_64']
Training.training(training_extent = 0, verbose = 1, batch_size = 128, Exact = 1, message = 1,
            Standard_train = Best_Neur, Dataset_train = OcT, Epoch = 8, 
            Var_X = Var_X_BIG, Verify = 0, Extra_n = 'Same_ind', 
            Similar_training = 1, Norm_Choix = 0, Method_data = 4)