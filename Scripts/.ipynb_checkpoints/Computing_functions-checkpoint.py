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

PWD = os.path.dirname(os.getcwd()) 
Bet_path = '/bettik/bouissob/'
Yr_t_s = 3600 * 24 * 365 # s/yr
Rho = 920 #Kg/m**3
Horiz_res = 2 #km/pix
S = (Horiz_res*10**3) ** 2  #m**2/pix
    
def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
def Fetch_data(Ocean_target, Type_tar):
    df = pd.read_csv(Getpath_dataset(Ocean_target, Type_tar))
    return df

def Fetch_model(model_path, name):
    if os.path.isfile(model_path + '/' + name):
        return tf.keras.models.load_model(model_path + '/' + name)
    elif os.path.isfile(model_path + '/' + 'model.h5'):
        return tf.keras.models.load_model(model_path + '/' + 'model.h5')
    else:
        print('File {} not found'.format(model_path + '/' + name))
        files = glob.glob(model_path + '/*')
        return None
                     
                     
def Fetch_model_data(model_path, Choix):
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

def Normalizer(d, Choix, Data_Norm):
    if str(Choix) == '0':
        MeanX, MeanY, StdX, StdY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MeanX)/StdX).reshape(-1, len(Var_X), )
    if str(Choix) == '1':
        MaxX, MinX, MaxY, MinY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MinX)/(MaxX - MinX)).reshape(-1, len(Var_X), )
    return X, Var_X
                     
def Get_model_attributes(model_p):
    if os.path.isfile(f + '/config.json'):
        with open(model_p + '/config.json') as json_file:
            data = json.load(json_file)
        return data
    else:
        return None

def Compute_rmse(Tar, Mod):
    return np.sqrt(np.mean((Mod-Tar)**2))

def Compute_data_from_model(X, Model, Choix, Data_norm):
    if str(Choix) == '0':
        Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
    elif str(Choix) == '1':
        Y = np.array(Model(X) * (Data_norm[2] - Data_norm[3]) + Data_norm[3])
    else:
        return None
    return Y
    
def Compute_datas(Model_path, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = False, Compute_at_ind = False, Datas = None):
    if Compute_at_ind :
        Data = Datas
    else:
        Data = Fetch_data(Ocean_target, Type_tar)
    tmx = int(max(np.array(Data.date)))
    Model = Fetch_model(Model_path, name = 'model_{}.h5'.format(Epoch))
    if Model == None:
        return None
    Data_norm = Fetch_model_data(Model_path, Choix)
    Var_X = Data_norm[-1]
    if message:
        print('Data variables used : {}'.format(' '.join(Var_X)))
    Melts, Modded_melts = [], []
    if not Compute_at_t :
        for t in range(tmx):
            if message and (t+1)%int(tmx/5) == 0:
                print('Starting {} / {}'.format(t+1, tmx) , end='\r')
            Melt, Modded = Compute_data_at_T(Data, t, Choix, Data_norm, Integrate = True)
            Melts.append(Melt)
            Modded_melts.append(Modded)
        Melts = np.array(Melts)
        Modded_melts = np.array(Modded_melts)
        return Melts, Modded_melts, Compute_rmse(Melts, Modded_melts), Model.count_params()
    else:
        Dataset = Compute_data_at_T(Data, Compute_at_t, Choix, Data_norm, Integrate = False)
        return Dataset

def Compute_data_at_T(Data, T, Choix, Data_norm, Integrate = True):
    Cur = Data.loc[Data.date == t].reset_index(drop = True)
    X, Var_X = Normalizer(Cur, Model_path, Choix, Data_norm)
    Cur['Mod_melt'] = Compute_data_from_model(X, Model, Choix)
    if Integrate:
        Melts = (Cur['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
        Modded_melts = (Cur['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
        return Melts, Modded_melts
    else:
        return Cur.set_index(['date', 'y', 'x']).to_xarray()

def Gather_datasets(Datasets, Type_tar, save_index = False):
    li = []
    for Ind, Dataset in enumerate(Datasets):
        D_path = os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Type_trained))
        df = pd.read_csv(D_path)
        if save_index:
            df['Oc'] = Ind + 1
        li.append(df)
    return pd.concat(li, ignore_index= True)
def Compute_data_for_plotting(Epoch = 4, Ocean_trained = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Compute_at_ind = False,
             Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1, index = None, Time = False, NN_attributes = {}):
    #Models_paths, path = Get_model_path_condition(Epoch, Ocean_trained, Type_trained, Exact, Extra_n)
    Models_paths = Get_model_path_json(Epoch = Epoch, Ocean = Ocean_trained, Type_trained = Type_trained,index = index, **NN_attributes)
    RMSEs, Params, Neurs, Melts, Modded_melts, Oc_mask, t = [], [], [], [], [], [], []
    #print(path + '/Ep_{}'.format(Epoch))
    if type(Ocean_target) != list:
        Ocean_target = [Ocean_target]
    if Compute_at_ind:
        DF = Gather_datasets(Ocean_target, Type_tar, save_index = True)
        Masks = os.path.join(PWD, 'Auto_model', 'tmp', 'Ocean1_Ocean2_Ocean3_Ocean4', 'ind_1646995963.csv')
        DF = DF.loc[np.loadtxt(Masks).astype(int)]
    for Oc in Ocean_target:
        Oc_m = int(Oc[-1])
        for ind, model_p in enumerate(Models_paths):
            print('Starting {}/{} model {}'.format(ind + 1, len(Models_paths), model_p.split('/')[-1]), end = '\n')
            Model_name = model_p.split('/')[-1]
            EpochM, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
            if Compute_at_ind:
                Datas = DF.loc[DF.Oc == Oc_m]
                Comp = Compute_datas(model_p, Choix, Oc, Type_tar, Epoch, message, Compute_at_ind = Datas)
            else:
                Comp = Compute_datas(model_p, Choix, Oc, Type_tar, Epoch, message)
            Melt, Modded_melt, RMSE, Param = Comp
            RMSEs.append(RMSE)
            if len(Ocean_target) != 1:
                Params = np.append(Params, np.full_like(Melt, Param))
                Neurs = np.append(Neurs, np.full_like(Melt, Neur))
            else:
                Params.append(Param)
                Neurs.append(Neur)
            Melts = np.append(Melts, Melt)
            Modded_melts = np.append(Modded_melts, Modded_melt)
            Oc_mask = np.append(Oc_mask, np.full_like(Melt, Oc_m))
            data = Get_model_attributes(model_p)
            t.append(data['Training_time'])
    return np.array(RMSEs), np.array(Params), Melts, Modded_melts, Neurs, Oc_mask, Ocean_trained, Ocean_target, Epoch, t

def Get_model_path_json(Var = None, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = 0, Extra_n = None, Choix = None, Neur = None, Batch_size = None, index = None):
    if type(Ocean) != list:
        Ocean = [Ocean]
    path = os.path.join(PWD, 'Auto_model', Type_trained, '_'.join(Ocean))
    if Exact == 1:
        Model_paths = glob.glob(path + '/Ep_{}*'.format(Epoch))
    else:
        Model_paths = glob.glob(path + '/Ep_*')
    #Mod = list(Model_paths)
    for f in list(Model_paths):
        data = Get_model_attributes(f)
        if data != None:
            if ((Choix != None and data['Choix'] != int(Choix)) or (Var != None and sorted(data['Var_X']) != sorted(Var))):
                Model_paths.remove(f)
                continue
            if (Neur != None and data['Neur_seq'] != Neur) or (Batch_size != None and Batch_size != data['batch_size']):
                Model_paths.remove(f)
                continue
            if (Exact != 1 and data['Epoch'] != Epoch) or (Extra_n != None and data['Extra_n'] != Extra_n):
                Model_paths.remove(f)
                continue
        else:
            Model_paths.remove(f)
    if index != None:
        if index >= len(Models_paths):
            index = len(Models_paths) - 1
        Models_paths = [Models_paths[index]]
        print(f'{Models_paths} \n', end = '\r')
    return Model_paths
