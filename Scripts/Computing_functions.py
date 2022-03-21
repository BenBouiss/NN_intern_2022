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

def Getpath_dataset(Dataset, Oc_mod_type):
    Bet_path = '/bettik/bouissob/'
    return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))

def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
def Fetch_data(Ocean_target, Type_tar):
    df = pd.read_csv(Getpath_dataset(Ocean_target, Type_tar))
    return df

def Fetch_model(model_path):
    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path)
    elif os.path.isfile(model_path + '/' + 'model.h5'):
        return tf.keras.models.load_model(model_path + '/' + 'model.h5')
    else:
        print('File {} not found'.format(model_path))
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
    if str(Choix) == '2':
        MedY = np.loadtxt(model_path + '/' + 'MedY.csv')
        iqrY = np.loadtxt(model_path + '/' + 'iqrY.csv')
        MedX = pd.read_pickle(model_path + '/' + 'MedX.pkl')
        iqrX = pd.read_pickle(model_path + '/' + 'iqrX.pkl')
        return MedX, iqrX, MedY, iqrY, MedX.index
    
    
def Normalizer(d, Choix, Data_Norm):
    if str(Choix) == '0':
        MeanX, MeanY, StdX, StdY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MeanX)/StdX).reshape(-1, len(Var_X), )
    elif str(Choix) == '1':
        MaxX, MinX, MaxY, MinY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MinX)/(MaxX - MinX)).reshape(-1, len(Var_X), )
    elif str(Choix) == '2':
        MedX, iqrX, MedY, iqrY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MedX)/(iqrX)).reshape(-1, len(Var_X), )
    return X, Var_X

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
    
def Compute_datas(Model,Model_path, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = False, Compute_at_ind = False, Datas = None):
    if Compute_at_ind :
        Data = Datas
    else:
        Data = Fetch_data(Ocean_target, Type_tar)
    tmx = int(max(np.array(Data.date)))
    
    if Model == None:
        return None
    Data_norm = Fetch_model_data(Model_path, Choix)
    Var_X = Data_norm[-1]
    if message:
        print('Data variables used : {}'.format(' '.join(Var_X)))
    Melts, Modded_melts = [], []
    if not type(Compute_at_t) == int :
        for t in range(tmx):
            if message and (t+1)%int(tmx/5) == 0:
                print('Starting {} / {}'.format(t+1, tmx) , end='\r')
                
            Cur = Data.loc[Data.date == t].reset_index(drop = True)
            Melt, Modded = Apply_NN_to_data(Model, Cur, Choix, Data_norm, Integrate = True)
            Melts.append(Melt)
            Modded_melts.append(Modded)
        Melts = np.array(Melts)
        Modded_melts = np.array(Modded_melts)
        return Melts, Modded_melts, Compute_rmse(Melts, Modded_melts), Model.count_params()
    else:
        Cur = Data.loc[Data.date == Compute_at_t].reset_index(drop = True)
        Dataset = Apply_NN_to_data(Model, Cur, Choix, Data_norm, Integrate = False)
        return Dataset

def Apply_NN_to_data(Model, Data, Choix, Data_norm, Integrate = True):
    X, Var_X = Normalizer(Data, Choix, Data_norm)
    Data['Mod_melt'] = Compute_data_from_model(X, Model, Choix, Data_norm)
    if Integrate:
        Melts = (Data['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
        Modded_melts = (Data['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
        return Melts, Modded_melts
    else:
        return Data.set_index(['date', 'y', 'x']).to_xarray()

def Gather_datasets(Datasets, Type_tar, save_index = False):
    li = []
    for Ind, Dataset in enumerate(Datasets):
        D_path = os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Type_tar))
        df = pd.read_csv(D_path)
        if save_index:
            df['Oc'] = Ind + 1
        li.append(df)
    return pd.concat(li, ignore_index= True)

def Compute_RMSE_from_model_ocean(Compute_at_ind = False, Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1, index = None, Time = False, NN_attributes = {}, Models_paths = None):
    if Models_paths == None:
        Models_paths = Get_model_path_json(index = index, **NN_attributes)
        
    RMSEs, Params, Neurs, Melts, Modded_melts, Oc_mask, t, uniq_id = [], [], [], [], [], [], [], []
    #print(path + '/Ep_{}'.format(Epoch))
    if type(Ocean_target) != list:
        Ocean_target = [Ocean_target]
    if Compute_at_ind:
        DF = Gather_datasets(Ocean_target, Type_tar, save_index = True)
        Att = Get_model_attributes(Models_paths[0])
        Masks = os.path.join(PWD, 'Auto_model', 'tmp', '_'.join(Ocean_target), f"ind_{Att['Similar_training']}.csv")
        DF = DF.loc[np.loadtxt(Masks).astype(int)]

    for ind, model_p in enumerate(Models_paths):
        print('Starting {}/{} model {}                                                   '.format(ind + 1, len(Models_paths), model_p.split('/')[-1]), end = '\r')
        for ind_o, Oc in enumerate(Ocean_target):
            Oc_m = int(Oc[-1])
            if '.h5' in model_p:
                Model = Fetch_model(os.path.join(model_p))
                Model_name = model_p.split('/')[-2]
                Neur, Choix = re.findall('N_(\w+)_Ch_(\d+)', Model_name)[0]
                Mod = model_p.split('/')[-1]
                Epoch = re.findall('model_(\d+).h5', Mod)[0]
                model_p = os.path.dirname(model_p)
            else:
                Model_name = model_p.split('/')[-1]
                Model = Fetch_model(os.path.join(model_p, 'model_{}.h5'.format(Epoch)))
                Epoch, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
            if Compute_at_ind:
                Datas = DF.loc[DF.Oc == Oc_m]
                Comp = Compute_datas(Model,model_p, Choix, Oc, Type_tar, Epoch, message, Compute_at_ind = True, Datas = Datas)
            else:
                Comp = Compute_datas(Model,model_p, Choix, Oc, Type_tar, Epoch, message)
            Melt, Modded_melt, RMSE, Param = Comp
            RMSEs.append(RMSE)
            data = Get_model_attributes(model_p)
            if len(Ocean_target) != 1:
                Params = np.append(Params, np.full_like(Melt, Param))
                #Neurs = np.append(Neurs, np.full_like(Melt, Neur))
                Neurs.extend([Neur] * len(Melt))
                uniq_id = np.append(uniq_id, np.full_like(Melt, data['Uniq_id']))
            else:
                Params.append(Param)
                Neurs.append(Neur)
                uniq_id.append(data['Uniq_id'])
            Melts = np.append(Melts, Melt)
            Modded_melts = np.append(Modded_melts, Modded_melt)
            Oc_mask = np.append(Oc_mask, np.full_like(Melt, Oc_m))
        t.append(data['Training_time'])
        Ocean_trained = data['Dataset_train']
    return np.array(RMSEs), np.array(Params), Melts, Modded_melts, Neurs, Oc_mask, Ocean_trained, Ocean_target, Epoch, t, uniq_id


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