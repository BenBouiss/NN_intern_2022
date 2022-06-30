import os
import numpy as np
import tensorflow as tf
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re 
import sys
import pickle
import pathlib
import matplotlib.colors as mcolors
import json
import itertools

from .Trainings import *



PWD = os.getcwd()
Bet_path = '/bettik/bouissob/'
Yr_t_s = 3600 * 24 * 365 # s/yr
Rho = 920 #Kg/m**3
Horiz_res = 2 #km/pix
S = (Horiz_res*10**3) ** 2  #m**2/pix

def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
def Fetch_data(Ocean_target, Type_tar, Method):
    df = pd.read_csv(Getpath_dataset(Ocean_target, Type_tar, Method))
    return df

def Fetch_model(model_path):
    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    elif os.path.isfile(model_path + '/' + 'model.h5'):
        return tf.keras.models.load_model(model_path + '/' + 'model.h5', compile=False)
    else:
        print('File {} not found'.format(model_path))
        files = glob.glob(model_path + '/*')
        return None


                     
def Fetch_model_stats(model_path, Choix):
    if str(Choix) == '0':
        StdY = np.loadtxt(model_path + '/' + 'StdY.csv')
        MeanY = np.loadtxt(model_path + '/' + 'MeanY.csv')
        MeanX = pd.read_pickle(model_path + '/' + 'MeanX.pkl')
        StdX = pd.read_pickle(model_path + '/' + 'StdX.pkl')
        return MeanX, MeanY, StdX, StdY, MeanX.index
    if str(Choix) == '1':
        MinY = np.loadtxt(model_path + '/' + 'MinY.csv')
        MaxY = np.loadtxt(model_path + '/' + 'MaxY.csv')
        MinX = pd.read_pickle(model_path + '/' + 'MinX.pkl')
        MaxX = pd.read_pickle(model_path + '/' + 'MaxX.pkl')
        return MaxX, MinX, MaxY, MinY, MaxX.index
    if str(Choix) == '2':
        MedY = np.loadtxt(model_path + '/' + 'MedY.csv')
        iqrY = np.loadtxt(model_path + '/' + 'iqrY.csv')
        MedX = pd.read_pickle(model_path + '/' + 'MedX.pkl')
        iqrX = pd.read_pickle(model_path + '/' + 'iqrX.pkl')
        return MedX, iqrX, MedY, iqrY, MedX.index
    
def Gather_datasets(Datasets, Type_tar, save_index = False, Method = None):
    li = []
    if type(Datasets) != list:
        Datasets = [Datasets]
    for Ind, Dataset in enumerate(Datasets):
        if Method == None:
            D_path = os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Type_tar))
        else:
            if Method == 4 or Method == 2 or Method == 5:
                D_path = Bet_path + f'Method_Data/{Type_tar}/Method_{Method}/{Dataset}_lite.csv'
            else:
                D_path = Bet_path + f'Method_Data/{Type_tar}/Method_{Method}/{Dataset}.csv'
        print(D_path)
        df = pd.read_csv(D_path)
        if save_index:
            df['Oc'] = Ind + 1
        li.append(df)
    return pd.concat(li, ignore_index= True)
    
def Normalize_dataset(Dataset, Choix, Path):
    Data_Norm = Fetch_model_stats(Path, Choix)
    if str(Choix) == '0':
        MeanX, MeanY, StdX, StdY, Var_X = Data_Norm
        X = Dataset[Var_X]
        X = np.array((X - MeanX)/StdX).reshape(-1, len(Var_X), )
    elif str(Choix) == '1':
        MaxX, MinX, MaxY, MinY, Var_X = Data_Norm
        X = Dataset[Var_X]
        X = np.array((X - MinX)/(MaxX - MinX)).reshape(-1, len(Var_X), )
    elif str(Choix) == '2':
        MedX, iqrX, MedY, iqrY, Var_X = Data_Norm
        X = Dataset[Var_X]
        X = np.array((X - MedX)/(iqrX)).reshape(-1, len(Var_X), )
    return X

def Compute_rmse(Tar, Mod):
    return np.sqrt(np.mean((Mod-Tar)**2))

def Compute_data_from_model(X, Model, Choix, Data_norm):
    if str(Choix) == '0':
        Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
    elif str(Choix) == '1':
        Y = np.array(Model(X) * (Data_norm[2] - Data_norm[3]) + Data_norm[3])
    elif str(Choix) == '2':
        Y = np.array(Model(X) * (Data_norm[3]) + Data_norm[2])
    else:
        return None
    return Y
    

def Get_paths_from_attributes(NN_attributes, NN_path_directory = None):
    if NN_path_directory == None:
        NN_path_directory = Get_model_path_json(**NN_attributes)[0]
    if '.h5' in NN_path_directory:
        NN_model_path = NN_path_directory
        NN_path_directory = os.path.dirname(NN_path_directory)
    else:
        Config = Get_model_attributes(NN_path_directory)
        NN_model_path = os.path.join(NN_path_directory, 'model_{}.h5'.format(Config['Epoch']))
    return NN_path_directory, NN_model_path

def Compute_NN_oceans(NN_attributes, Ocean_target : list, Type_target = 'COM_NEMO-CNRS', T = None, shuffle = None, NN_path = None):
    
    '''
    Input :
    Ocean target : can be either list of string, represent the ocean targets to which computing will occur on.
    NN_attributes : is a dictionnary containing all the attributes of the NN to be used.
    T[month] : Either list int or other, if list or int the computing will occur on either multiple or one timestep. If other, the computing will occur on all timesteps.
    Shuffle : List, must contains atleast one variable name to be shuffled if '-ms' is in the list mixed shuffling will occcur.
    
    Output:
    if no T is given, the function ouputs integrated melt rates over the entire ice shelf. If a non None value for T is given say 'ALL' or anything else, the function outputs horizontal profile of melt rates at the time steps dependant on T.
    '''
    if type(Ocean_target) != list:Ocean_target = [Ocean_target]
    NN_path_directory, NN_model_path = Get_paths_from_attributes(NN_attributes, NN_path_directory = NN_path)
    print(f'Model used : {NN_model_path}')
    Config = Get_model_attributes(NN_path_directory)    
    All_melts, All_reference_melts, RMSEs = [], [], []
    ds = []
    for Ocean in Ocean_target:
        Dataset = Fetch_data(Ocean, Type_target, Method = Config['Method_data'])
        Start = time.time()
        if T == None:
            Melts, Reference_melts, RMSE = Compute_RMSE_from_model_ocean(NN_path_directory, NN_model_path, Config, Dataset, shuffle, integrate = True)
            All_melts.append(Melts.to_list())
            All_reference_melts.append(Reference_melts.to_list())
            RMSEs.append(RMSE)
        else:
            if type(T) == list or type(T) == int:
                Dataset = Dataset.loc[Dataset.date.isin(T)]
            d = Compute_RMSE_from_model_ocean(NN_path_directory, NN_model_path, Config, Dataset, shuffle, integrate = False)
            ds.append(d)
        print(f'Done computing for {Ocean} in : {int(time.time() - Start)} s')

    if T == None:
        Overall_RMSE = Compute_rmse(np.array(np.concatenate(All_melts).flat), np.array(np.concatenate(All_reference_melts).flat))
        return All_melts, All_reference_melts, RMSEs, Overall_RMSE, Ocean_target
    else:
        return ds, Ocean_target
                                  
def Shuffling_variables(Data, shuffle):
    if '-ms' in shuffle:
        print('Mixed shuffling')
        shuffle.remove('-ms')
        for v in shuffle:
            Data[v] = Data[v].sample(frac=1).values    
    else :
        Data[shuffle] = Data[shuffle].sample(frac=1).values
    return Data
                                  
def Compute_RMSE_from_model_ocean(NN_path_directory, NN_path : str, Config, Dataset, shuffle, integrate):
    NN = Fetch_model(NN_path)
    X = Normalize_dataset(Dataset, Config['Choix'], NN_path_directory)
    if shuffle != None:
        X = Shuffling(X, shuffle)
    Melts, Reference_melts = Compute_NN_results(NN, X, Config, NN_path_directory, Dataset, integrate)
    RMSE = Compute_rmse(Melts, Reference_melts)
    return Melts, Reference_melts, RMSE

def Compute_NN_results(NN, X, Config, NN_path_directory, Dataset, integrate = True):
    #print(Dataset.head())
    Dataset['Mod_melt'] = Compute_data_from_model(X, NN, Config, NN_path_directory)
    if integrate == True:
        Melts = (Dataset.groupby('date')['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
        Modded_melts = (Dataset.groupby('date')['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
        return Melts, Modded_melts
    elif integrate == False:
        return Dataset.set_index(['date', 'y', 'x']).to_xarray()
                                  
def Compute_data_from_model(X, Model, Config, NN_path_directory):
    Choix = Config['Choix']
    Data_norm = Fetch_model_stats(NN_path_directory, Choix)
    if str(Choix) == '0': # (X-mean / std) Mean/standard deviation normalisation method
        Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
    elif str(Choix) == '1': #X-min/(Max-min) normalisation method
        Y = np.array(Model(X) * (Data_norm[2] - Data_norm[3]) + Data_norm[3]) 
    elif str(Choix) == '2': #X-med/(iqr) interquartile normalisation method
        Y = np.array(Model(X) * (Data_norm[3]) + Data_norm[2]) 
    else:
        print('No valid normalisation choice was found.')
        return None
    return Y
                                  
def Compute_RMSEs_Total_Param(**kwargs):
    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, _, t, ids = Compute_RMSE_from_model_ocean(**kwargs)
    if len(Params) == len(Melts):
        df = pd.DataFrame()
        df['Melts'] = Melts
        df['Modded_melts'] = Modded_melts
        df['Params'] = Params
        df['Neurs'] = Neurs
        df['uniq_id'] = ids
        RMSE = []
        Neurs = []
        T = []
        Param = []
        ids = np.unique(ids)
        for i in ids:
            Cur = df.loc[df.uniq_id == i]
            RMSE.append(Compute_rmse(np.array(Cur.Melts), np.array(Cur.Modded_melts)))
            Neurs.append(np.unique(Cur.Neurs))
            Param.append(np.unique(Cur.Params))
    else:
        RMSE = RMSEs
        Param = Params
    T = t
    print(f"T size {len(T)} \n RMSE size : {len(RMSE)} \n Param size : {len(Param)}")
    return Param, T, RMSE, Neurs


def Compute_benchmark(name : str, Oc, Compute_at_ind = False, NN_attributes = {}):
    Models_p = Get_model_path_json(Extra_n = name, return_all = True, **NN_attributes)
    li_RMSE = []
    for i, mod in enumerate(Models_p):
        print(f'Starting {mod}        {i+1}/{len(Models_p)}')
        RMSEs, _, Melts, Modded_melts, _, Oc_mask, Ocean_trained, Ocean_target, _, _, _ = Compute_RMSE_from_model_ocean(Compute_at_ind = Compute_at_ind, Ocean_target = Oc, Type_tar = 'COM_NEMO-CNRS', message = 0, Models_paths = mod)
        config = Get_model_attributes(mod)
        li_RMSE.append([RMSEs, Compute_rmse(Melts, Modded_melts), config['Var_X']])
    return li_RMSE
        

def Compute_shuffle_benchmark(Oc, Compute_at_ind = False, NN_attributes = {}, Specific_shuffle = []):
    Model_p = Get_model_path_json(return_all = False, **NN_attributes)[0]
    li_RMSE = []
    print(f'Model selected : {Model_p}')
    Config = Get_model_attributes(Model_p)
    Vars = Config.get('Var_X')
    if 'T_0' in Vars:
        inst = []
        for i in range(40):
            Vars.remove(f'T_{i}')
            inst.append(f'T_{i}')
        Vars.append(inst)
        Vars.append(inst + ['-ms'])
    if 'S_0' in Vars:
        inst = []
        for i in range(40):
            Vars.remove(f'S_{i}')
            inst.append(f'S_{i}')
        Vars.append(inst)
        Vars.append(inst + ['-ms'])
    if 'Slope_iceDraft_x' in Vars:
        Vars.remove('Slope_iceDraft_x')
        Vars.remove('Slope_iceDraft_y')
        Vars.append(['Slope_iceDraft_x', 'Slope_iceDraft_y'])
        
    if 'Slope_bathymetry_x' in Vars:
        Vars.remove('Slope_bathymetry_x')
        Vars.remove('Slope_bathymetry_y')
        Vars.append(['Slope_bathymetry_x', 'Slope_bathymetry_y'])
        
    if Specific_shuffle != []:
        Vars = [Specific_shuffle]
        
        
    for i, shuffle in enumerate(Vars):
        print(f'Starting {shuffle}        {i+1}/{len(Vars)}')
        RMSEs, _, Melts, Modded_melts, _, Oc_mask, Ocean_trained, Ocean_target, _, _, _ = Compute_RMSE_from_model_ocean(Compute_at_ind = Compute_at_ind, Ocean_target = Oc, Type_tar = 'COM_NEMO-CNRS', message = 0, Models_paths = Model_p, shuffle = shuffle if type(shuffle) != list else shuffle.copy())
        li_RMSE.append([RMSEs, Compute_rmse(Melts, Modded_melts), shuffle, Ocean_target])
    return li_RMSE



def Compute_benchmark_function(NN_attributes = {}, li_activation = [], Ocean_target = 'Ocean1', Compute_at_ind = False, message = 1, Get_time = False):
    
    
    Tot = []
    li_fct = []
    for indAc, Activation in enumerate(li_activation):
        li_fct.append(Activation)
        NN_attributes['Activation_fct'] = Activation
        Model_ps = Get_model_path_json(return_all = True, **NN_attributes)
        P = []
        RMSEs_tot = []
        Time = []
        Neur = []
        for indp, p in enumerate(Model_ps):
            print(f'{Activation}  : {indAc+1}/{len(li_activation)}            {p}  :  {indp+1}/{len(Model_ps)}')
            if Get_time:
                Data = Get_model_attributes(p)
                Time.append(Data.get('Training_time'))
                P.append(Data.get('Param'))
                Neur.append(Data.get('Neur_seq'))
                print(Neur)
            else:
                RMSEs, Param, Melts, Modded_melts, _, _, _, _, _, _, _ = Compute_RMSE_from_model_ocean(Compute_at_ind = Compute_at_ind, 
                                        Ocean_target = Ocean_target, Type_tar = 'COM_NEMO-CNRS', message = message, Models_paths = p)
                P.append(np.unique(Param).astype(int)[0])
                RMSEs_tot.append(Compute_rmse(Melts, Modded_melts))
        if Get_time:
            Tot.append([Time, P, Neur])
        else:
            Tot.append([P, RMSEs_tot])
    return Tot

def Compute_general_benchmark(Var_to_bench : str, NN_attributes = {}, **kwargs):

    Model_paths = Get_model_path_json(return_all = True, **NN_attributes)
    Interest = []
    ALL_RMSEs = []
    ALL_Overall_RMSE = []
    dfT = pd.DataFrame()
    for i, p in enumerate(Model_paths):
        print(f'Model : {p}, ( {i+1} / {len(Model_paths)})')
        Config = Get_model_attributes(p)
        #RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, *_ = Compute_RMSE_from_model_ocean(**kwargs, Models_paths = p)
        All_melts, All_reference_melts, RMSEs, Overall_RMSE, Ocean_target = Compute_NN_oceans(NN_attributes = {}, NN_path = p, **kwargs)
        Interest.append(Config.get(Var_to_bench))
        ALL_RMSEs.append(RMSEs)
        ALL_Overall_RMSE.append(Overall_RMSE)
        Path = PWD + '/Cached_data/Generic_benchmark/'
    df = pd.DataFrame()
    df[Var_to_bench] = Interest
    df['Overall_RMSE'] = ALL_Overall_RMSE
    df[Ocean_target] = ALL_RMSEs
#    return Interest, ALL_RMSEs, Overall_RMSE, Oc_tar
    pd.DataFrame.to_csv(df, Path + f'{Var_to_bench}_{int(time.time())}.csv', index = False)
    return df
        
