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
    
def Compute_datas(Model,Model_path, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = False, Compute_at_ind = False, Datas = None, Method = None, shuffle = [], return_df = False):
    if Compute_at_ind :
        Data = Datas
    else:
        Data = Fetch_data(Ocean_target, Type_tar, Method)
        
        #Data[shuffle] = np.random.permutation(Data[shuffle].values)
        if '-ms' in shuffle:
            print('Mixed shuffling')
            shuffle.remove('-ms')
            for v in shuffle:
                Data[v] = Data[v].sample(frac=1).values    
        else :
            Data[shuffle] = Data[shuffle].sample(frac=1).values
        #df1['HS_FIRST_NAME'] = df[4].sample(frac=1).values
        
    tmx = int(max(np.array(Data.date)))
    
    if Model == None:
        return None
    Data_norm = Fetch_model_data(Model_path, Choix)
    Var_X = Data_norm[-1]
    if message:
        print('Data variables used : {}'.format(' '.join(Var_X)))
    Melts, Modded_melts = [], []
    if Compute_at_t == False:
        for t in range(tmx):
            if (t+1)%int(tmx/5) == 0:
                print('Starting {} / {}'.format(t+1, tmx) , end='\r')
                
            Cur = Data.loc[Data.date == t].reset_index(drop = True)
            Melt, Modded = Apply_NN_to_data(Model, Cur, Choix, Data_norm, Integrate = True)
            Melts.append(Melt)
            Modded_melts.append(Modded)
        Melts = np.array(Melts)
        Modded_melts = np.array(Modded_melts)
        return Melts, Modded_melts, Compute_rmse(Melts, Modded_melts), Model.count_params()
    else:
        if type(Compute_at_t) == int:
            Cur = Data.loc[Data.date == Compute_at_t].reset_index(drop = True)
            Dataset = Apply_NN_to_data(Model, Cur, Choix, Data_norm, Integrate = False)
            return Dataset
        else:
            d = []
            if Compute_at_t == 'ALL':
                Dataset = Apply_NN_to_data(Model, Data, Choix, Data_norm, Integrate = False)
                return Dataset
            for t in Compute_at_t:
                print(f"Starting {t} / {Compute_at_t}        ", end = '\r')
                Cur = Data.loc[Data.date == t].reset_index(drop = True)
                Dataset = Apply_NN_to_data(Model, Cur, Choix, Data_norm, Integrate = False)
                d.append(Dataset)
                if t != Compute_at_t[-1]:
                    print(" ######################################################## ", end = '\r')
            return d
        
def Apply_NN_to_data(Model, Data, Choix, Data_norm, Integrate = True):
    X, Var_X = Normalizer(Data, Choix, Data_norm)
    Data['Mod_melt'] = Compute_data_from_model(X, Model, Choix, Data_norm)
    if Integrate:
        Melts = (Data['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
        Modded_melts = (Data['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
        return Melts, Modded_melts
    else:
        print('Done computing')
        return Data.set_index(['date', 'y', 'x']).to_xarray()

def Gather_datasets(Datasets, Type_tar, save_index = False, Method = None):
    li = []
    for Ind, Dataset in enumerate(Datasets):
        if Method == None:
            D_path = os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Type_tar))
        else:
            if Method == 4 or Method == 2 or Method == 5:
                D_path = Bet_path + f'Method_Data/{Type_tar}/Method_{Method}/{Dataset}_lite.csv'
            else:
                D_path = Bet_path + f'Method_Data/{Type_tar}/Method_{Method}/{Dataset}.csv'
        df = pd.read_csv(D_path)
        if save_index:
            df['Oc'] = Ind + 1
        li.append(df)
    return pd.concat(li, ignore_index= True)

def Compute_RMSE_from_model_ocean(Compute_at_ind = False, Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1, index = None, Time = False, NN_attributes = {}, Models_paths = None, shuffle = []):
    if Models_paths == None:
        Models_paths = Get_model_path_json(**NN_attributes)
        if message == 1:
            print(Models_paths)
        else:
            print(f'Number of model used : {len(Models_paths)}')
    
    if type(Models_paths) != list:
        Models_paths = [Models_paths]
    RMSEs, Params, Neurs, Melts, Modded_melts, Oc_mask, t, uniq_id = [], [], [], [], [], [], [], []
    #print(path + '/Ep_{}'.format(Epoch))
    if type(Ocean_target) != list:
        Ocean_target = [Ocean_target]
    if Compute_at_ind:
        data = Get_model_attributes(Models_paths)
        DF = Gather_datasets(Ocean_target, Type_tar, save_index = True, Method = data.get('Method_data'))
        Att = Get_model_attributes(Models_paths[0])
        Masks = os.path.join(PWD, 'Auto_model', 'tmp', '_'.join(Ocean_target), f"ind_{Att['Similar_training']}.csv")
        DF = DF.loc[np.loadtxt(Masks).astype(int)]

    for ind, model_p in enumerate(Models_paths):
        if '.h5' in model_p:
            print('Starting {}/{} model {}                                                   '.format(ind + 1, len(Models_paths), '/'.join(model_p.split('/')[-2:])), end = '\r')
        else:
            print('Starting {}/{} model {}                                                   '.format(ind + 1, len(Models_paths), model_p.split('/')[-1]), end = '\r')
        for ind_o, Oc in enumerate(Ocean_target):
            if Oc[0:5] == 'Ocean':
                Oc_m = int(Oc[-1])
            elif 'IceOcean' in Oc:
                Oc_m = re.findall('IceOcean(\d+)', Oc)[0]
                'IceOcean1r_ElmerIce'
            else:
                Oc_m = re.findall('CPL_EXP(\d+)_rst', Oc)[0]
            data = Get_model_attributes(model_p)
            if '.h5' in model_p:
                Model = Fetch_model(model_p)
                Model_name = model_p.split('/')[-2]
                Neur, Choix = re.findall('N_(\w+)_Ch_(\d+)', Model_name)[0]
                Mod = model_p.split('/')[-1]
                Epoch = re.findall('model_(\d+).h5', Mod)[0]
                model_p = os.path.dirname(model_p)
                data = Get_model_attributes(model_p)
            else:
                Model_name = model_p.split('/')[-1]
                Epoch, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
                Model = Fetch_model(os.path.join(model_p, 'model_{}.h5'.format(Epoch)))
            if Compute_at_ind:
                Datas = DF.loc[DF.Oc == Oc_m]
                Comp = Compute_datas(Model,model_p, Choix, Oc, Type_tar, Epoch, message, Compute_at_ind = True, Datas = Datas, shuffle = shuffle)
            else:
                Comp = Compute_datas(Model,model_p, Choix, Oc, Type_tar, Epoch, message, Method = data.get('Method_data'), shuffle = shuffle)
            Melt, Modded_melt, RMSE, Param = Comp
            RMSEs.append(RMSE)
            
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



def Compute_benchmark_function(NN_attributes = {}, li_activation = [], Ocean = 'Ocean1', Compute_at_ind = False, message = 1, Get_time = False):
    
    
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
                                        Ocean_target = Ocean, Type_tar = 'COM_NEMO-CNRS', message = message, Models_paths = p)
                P.append(np.unique(Param).astype(int)[0])
                RMSEs_tot.append(Compute_rmse(Melts, Modded_melts))
        if Get_time:
            Tot.append([Time, P, Neur])
        else:
            Tot.append([P, RMSEs_tot])
    return Tot