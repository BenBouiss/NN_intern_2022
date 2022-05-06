import sys
#sys.path.append('../')
import os
from Scripts import Trainings
from Scripts import Plotting
from Scripts import Computing_functions
import numpy as np
import time
#import os
import glob
import pandas as pd
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import xarray as xr




OcT = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4']
Var_X_BIG = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'Big_T', 'Big_S']
Var_X_Extra = ['iceDraft', 'Slope_iceDraft', 'Slope_bathymetry', 'Distances', 'temperatureYZ', 'salinityYZ']
Var_X_BIG_test = ['iceDraft','Big_T', 'Big_S']
Var_X_BIG_Extra = ['iceDraft', 'bathymetry', 'Slope_iceDraft_x', 'Slope_bathymetry_x',
                   'Slope_iceDraft_y', 'Slope_bathymetry_y', 'Big_T', 'Big_S',
                  'Distances_ground_line', 'Distances_front_line']
OcTPlus = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst']
CPLs_test = ['CPL_EXP10_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']
CPLs_test = ['CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']
ALL_EXP = ['CPL_EXP10_rst','CPL_EXP11_rst', 'CPL_EXP12_rst','CPL_EXP13_rst','CPL_EXP20_rst','CPL_EXP21_rst','CPL_EXP22_rst', 'CPL_EXP23_rst']
Train_oc_exp = ['Ocean1', 'Ocean2', 'Ocean3', 'Ocean4', 'CPL_EXP10_rst','CPL_EXP13_rst', 'CPL_EXP22_rst', 'CPL_EXP23_rst']



Dataset = 'Ocean1'
Oc_mod_type = 'COM_NEMO-CNRS'
Bet_path = '/bettik/bouissob/'
D_path = os.path.join(Bet_path, 'Method_Data/COM_NEMO-CNRS/Method_4', 'Ocean4.csv')


Uniq_id = int(time.time())
li = []
ALL_OC = OcT + ALL_EXP
oc = ALL_OC
NN_attributes = {'Epoch' : 128, 'Ocean' : Train_oc_exp, 'Exact' : 1, 'Method_data' : 4}
Mod_p = Trainings.Get_model_path_json(**NN_attributes)[0]
Data = Trainings.Get_model_attributes(Mod_p)
Ident = Data.get('Uniq_id')
folder_p = f'{os.getcwd()}/Cached_data/Shuffle_benchmark/Var_benchmark_{Ident}/'
Trainings.Make_dire(folder_p)

for i in range(5):
    
    li = Computing_functions.Compute_shuffle_benchmark(Oc = oc, NN_attributes = NN_attributes)
    
    li.extend(Computing_functions.Compute_shuffle_benchmark(Oc = oc, NN_attributes = NN_attributes,
             Specific_shuffle = ['bathymetry', 'iceDraft']))
    
    New_l = pd.DataFrame(li, columns = ['RMSEs', 'RMSE_tot', 'Var', 'Oc'])
    
    
    df = pd.DataFrame(li, columns = ['RMSEs', 'RMSE_tot', 'Var', 'Oc'])
    
    #p = f'{os.getcwd()}/Cached_data/Shuffle_benchmark/Var_benchmark_{i}_.csv'
    #df.to_csv(p, index = False)
    p = f'{folder_p}Var_benchmark_{Ident}_{i}_.csv'
    
    RMSEs, RMSE_tot, Var, Oc,  = [], [], [], []
    
    for l in li:
        RMSEs.append(l[0])
        RMSE_tot.append(l[1])
        Var.append(l[2])
        Oc.append(l[3])
        
    df = pd.DataFrame()
    df['Var_shuffled'] = Var
    df['RMSE_tot'] = RMSE_tot
    df['Ocean'] = Oc
    df[Oc[0]] = RMSEs
    df.to_csv(p, index = False)
    
    
