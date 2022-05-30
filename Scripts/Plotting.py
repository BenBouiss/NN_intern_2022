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
import seaborn as sns
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib import ticker
import matplotlib

sns.set_context('paper')
from .Computing_functions import *
from .Trainings import *


from mpl_toolkits.axes_grid1 import make_axes_locatable



PWD = os.getcwd()
Bet_path = '/bettik/bouissob/'
SIZE = 13
def Plot_RMSE_to_param(save = False, **kwargs):
    RMSEs, Params, _, _, Neurs, _, Oc_tr, Oc_tar, Ep, t = Compute_RMSE_from_model_ocean(**kwargs)
    fig, ax = plt.subplots()
    ax.scatter(Params, RMSEs, color = 'blue')
    if t != []:
        ax2 = ax.twinx()
        ax2.scatter(Params, t, s = 10, color = 'red')
        ax2.set_ylabel("Training time(s)",color="red")

    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('RMSE of NN vs modeled melt rates(Gt/yr)', color = 'blue')
    plt.title('RMSE as a function of parameters trained \n (NN trained on {} applied to {})'.format(Concat_Oc_names(Oc_tr), Concat_Oc_names(Oc_tar)))
    if save:
        plt.savefig(os.path.join(PWD, 'Image_output', 'RMSE_param_Ep{}_Tr{}_Tar{}_{}'.format(Ep, 
                    Concat_Oc_names(Oc_tr), Concat_Oc_names(Oc_tar), int(time.time()))), facecolor = 'white')
    return RMSEs, Params, Neurs, t
def Plot_total_RMSE_param(save = False, message_p = 1,load = False,See_best = False,NN_attributes = {},Axis_type = False, **kwargs):
    if load == True:
        Path = os.path.join(os.getcwd(), 'Cached_data', 'Total_RMSE_Ep_8_Ch_0_OcT_VarX_non_posit_same_ind.csv')
        df = pd.read_csv(Path)
        Neur = df['Neur'].values
        Param = df['Params'].values
        RMSE = df['RMSE'].values
        T = df['T'].values
    elif load == 'Hist':
        print('Loading histories')
        Paths = Get_model_path_json(**NN_attributes, return_all = True)
        Param = []
        RMSE = []
        T = []
        Neur = []
        for p in Paths:
            Train_hist = pd.read_pickle(p + '/TrainingHistory')
            Data = Get_model_attributes(p)
            #Param.append(Data.get('Param'))
            Model = Fetch_model(os.path.join(p, 'model_{}.h5'.format(Data.get('Epoch'))))
            Param.append(Model.count_params())
            RMSE.append(Train_hist['loss'][-1])
            T.append(Data.get('Training_time'))
            Neur.append(Data.get('Neur_seq'))
    else:
        Param, T, RMSE, Neur = Compute_RMSEs_Total_Param(NN_attributes = NN_attributes, **kwargs)
    fig, ax = plt.subplots()
    dim = (8.75/3*3 * 0.75, 8.75/3*2 * 0.75)
    fig.set_size_inches(dim)
    ax.scatter(Param, RMSE)
    ax2 = ax.twinx()
    ax2.scatter(Param, T, s = 10, color = 'red')
    ax2.set_ylabel("Training time(s)",color="red", fontsize = SIZE)
    ax.set_xlabel('Number of parameters', fontsize = SIZE)
    ax.set_ylabel('RMSE of NN vs reference \n integrated melt rates(Gt/yr)', fontsize = SIZE)
    
    if Axis_type == 'Log':
        ax.set_xscale('log')
        ax2.set_xscale('log')
    if See_best:
        Bests = RMSE.argsort()[:4]
        ax.scatter(Param[Bests], RMSE[Bests])
    
    #sns.despine()
    ax.tick_params(labelsize = SIZE - 2)
    ax2.tick_params(labelsize = SIZE - 2)
    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(PWD, 'Image_output', 'Tot_RMSE_param_{}'.format(int(time.time()))), facecolor = 'white')
    return Param, RMSE, Neur, T
    
def Plot_Melt_time_function(ind = 0, save = False, Nothing = False,Save_name = '',Indep = True, Display_label = True,
                            NN_attributes = {},Labels = [], Display_title = True,Title = None, SLC = False, Axis_type = None, **kwargs):
    li_NN = [NN_attributes] if type(NN_attributes) != list else NN_attributes
    
    Gt_ice_to_mm = 1 / 361.8
    if Indep == False:
        f = plt.figure(1)
    else:
        f = plt.figure()
    dim = (8.75/4 * 2.4, 8.75/4 * 3* 0.5)
    dim = (8.75/3*3 * 0.75, 8.75/3*2 * 0.75)
    f.set_size_inches(dim)
    RMSEs = []
    RMSE_SLC = []
    for i, NN in enumerate(li_NN):
        RMSE, _, Melts, Modded_Melts, _, _, Oc_train, Oc_tar, *_ = Compute_RMSE_from_model_ocean(index = ind, NN_attributes = NN, **kwargs)
        RMSEs.append(RMSE)
        x = np.arange(1, len(Modded_Melts) + 1)
        if len(Labels) >= len(li_NN):
            label = Labels[i]
        else:
            label = f'NN trained on {Concat_Oc_names(Oc_train)}'
        
        if SLC:
            Real = np.cumsum(Melts) * (1/12) * Gt_ice_to_mm
            Mod = np.cumsum(Modded_Melts) * (1/12) * Gt_ice_to_mm
            
            Melts = Real
            Modded_Melts = Mod
            Cur_RMSE_SLC = Compute_rmse(Real, Mod)
            print(f' RMSE : {Cur_RMSE_SLC} mm')
            RMSE_SLC.append(Cur_RMSE_SLC)
        if i == 0:
            if SLC:
                lab = 'SLC'
            else:
                lab = 'melt'
            if Indep != True:
                plt.plot(x, Melts, label = f'Reference {lab}') #{Concat_Oc_names(Oc_tar)}')
            else:
            #plt.figure()

                plt.plot(x, Melts, label = f'Reference {lab}')# {Concat_Oc_names(Oc_tar)}')
            
            
            
        if Nothing == False:
            plt.plot(x, Modded_Melts, label = label)
            if Display_title :
                if Title == None:
                    plt.title('Modeling melt rates of {} \n (NN trained on {})'.format(
                        Concat_Oc_names(Oc_tar), Concat_Oc_names(Oc_train)))
                else:
                    plt.title(Title, fontsize = SIZE)
        else:
            if Display_title :
                if Indep == True:
                    plt.title(f'Integrated melt rates over the iceshelf \n under {Concat_Oc_names(Oc_tar)} scenario')
                else:
                    plt.title(f'Integrated melt rates over the iceshelf', fontsize = SIZE)


    f.tight_layout()
    if Axis_type == 'Log':
        plt.gca().set_yscale('log')
        print('Log y')
    plt.xlabel('Time (month)', fontsize = SIZE)
    if SLC:
        plt.ylabel('SLC(mm)', fontsize = SIZE)
    else:
        plt.ylabel('Mass loss(Gt/yr)', fontsize = SIZE)
    sns.despine()
    #print(Concat_Oc_names(Oc_tar))
    #print(Concat_Oc_names(Oc_train))
    if Display_label :
        plt.legend(fontsize = SIZE - 2)
    plt.gca().tick_params(labelsize = SIZE - 2)
    print(f'{Oc_tar} : {RMSE} Gt/yr \n')
    if save:
        plt.savefig(os.path.join(PWD, 'Image_output', 'Melt_time_fct_M_{}_{}={}_Ex{}.png'.format(int(time.time()), 
                    Concat_Oc_names(Oc_train), Concat_Oc_names(Oc_tar), Save_name)),facecolor='white', bbox_inches = "tight", dpi = 300)
    if SLC:
        return RMSEs, RMSE_SLC
    return RMSEs
def Plot_Melt_to_Modded_melt(save = False, Save_name = '',Compute_at_ind = False,Display_label= True,Display_title = True, **kwargs):
    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, *_ = Compute_RMSE_from_model_ocean(Compute_at_ind = Compute_at_ind, **kwargs)
    fig, ax = plt.subplots()
    dim = (8.75/3*3 * 0.75, 8.75/3*2 * 0.75)
    fig.set_size_inches(dim)
    Vmin = min(np.append(Melts, Modded_melts))
    Vmax = max(np.append(Melts, Modded_melts))
    for v in np.unique(Oc_mask):
        idx = np.where(Oc_mask == v)
        if Display_label:
            label =  f'R = {np.round(np.corrcoef(Modded_melts[idx],Melts[idx])[0,1], 4)}'
        else:
            label = ''
        ax.scatter(Modded_melts[idx], Melts[idx], s = 3, alpha = 0.6 ,label = f'Ocean{int(v)} {label}')
        ax.legend(Oc_mask)
    linex, liney = [Vmin, Vmax], [Vmin, Vmax]
    ax.plot(linex, liney, c = 'red', ls = '-', linewidth = 0.8)
    ax.legend(loc = 'upper left', fontsize = SIZE - 2)
    plt.gca().tick_params(labelsize = SIZE - 2)
    plt.xlabel('NN emulated melt (Gt/yr)', fontsize = SIZE)
    plt.ylabel('Reference melt (Gt/yr)', fontsize = SIZE)
    sns.despine()
    if Display_title:
        plt.title('Modeling melt rates of {} \n (NN trained on {})'.format(Concat_Oc_names(Oc_tar), Concat_Oc_names(Oc_tr)), fontsize = SIZE)
    plt.show()
    print(Compute_rmse(Melts, Modded_melts))
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 'Line_real_Modeled_N{}_Tr{}_Tar{}_{}_Ex_{}'.format(np.unique(Neurs), 
                    Concat_Oc_names(Oc_tr), Concat_Oc_names(Oc_tar), int(time.time()), Save_name)), facecolor = 'white', dpi = 300)
    return RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask


    
def Plot_loss_model(save = False, ind = 0,Forbid_key = [],Second_axis = [],Title = True, Desired_length = None, **kwargs):
    #Models_p, _ = Get_model_path_condition(**kwargs)
    Models_p = Get_model_path_json(**kwargs)
    fig, ax = plt.subplots()
    dim = (8.75/3*3 * 0.75, 8.75/3*2 * 0.75)
    fig.set_size_inches(dim)
    if ind >= len(Models_p):
        ind = len(Models_p) - 1
    Model_p = Models_p[ind]
    hist = pd.read_pickle(Model_p + '/TrainingHistory')
    #plt.plot(hist['loss'])
    data = Get_model_attributes(Model_p)
    for k in hist.keys():
        if k not in Forbid_key and k not in Second_axis:
            if Desired_length != None:
                ax.plot(hist[k][:Desired_length], label = k)                
            else:
                ax.plot(hist[k], label = k)
            
    if Second_axis != []:
        ax2 = ax.twinx()
        for k in Second_axis:
            if Desired_length != None:
                ax2.plot(hist[k][:Desired_length], label = k, color = 'slategrey')
            else:
                ax2.plot(hist[k], label = k, color = 'slategrey')
        #ax2.legend(loc = 'upper left')
        ax2.set_ylabel("Learning rate",color="slategrey", fontsize = SIZE)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=0, fontsize = SIZE)
        ax2.tick_params(labelsize = SIZE - 2)
    else:
        ax.legend(fontsize = SIZE, loc = 'upper right')
    print(Model_p.split('/')[-1])
    if Title:
        plt.title('Loss graph for model : {}'.format(Model_p.split('/')[-1]))
    ax.set_xlabel('Epoch', fontsize = SIZE)
    ax.tick_params(labelsize = SIZE - 2)
    sns.despine()
    plt.show()
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            f"Loss_graph_M_{data['Neur_seq']}_{data['Uniq_id']}.png"), facecolor='white', bbox_inches='tight', dpi = 300)
    return hist

def Plot_Loss_against_loss(save = False, ind = 0, Desired_comparaison = [], Second_axis = [], Title = True, Mods = [], label = [], Desired_length = None):

    li = []
    fig, ax = plt.subplots()
    dim = (8.75/3*3 * 0.75, 8.75/3*2 * 0.75)
    fig.set_size_inches(dim)
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    Uniqs = np.unique(label)
    
    for mod in Mods:
        Model_p = Get_model_path_json(index = ind, **mod)[0]
        print(Model_p)
        data = Get_model_attributes(Model_p)
        Hist = pd.read_pickle(Model_p + '/TrainingHistory')
        Cur_li = []
        for k in Hist.keys():
            if k in Desired_comparaison:
                if Desired_length == None:
                    Cur_li.append(Hist[k])
                else:
                    Cur_li.append(Hist[k][:Desired_length])
        li.append(Cur_li)
    #print(li)
    Plotted_labels = []
    for i, Data in enumerate(li):
        if type(Data[0]) != list:
            ax.plot(Data, label = Desired_comparaison[i])
        else:
            for j,d in enumerate(Data):
                #print(d)
                if label == []:
                    ax.plot(d, label = f'{Desired_comparaison[j]}')
                else:
                    if label[i] not in Plotted_labels:
                        Plotted_labels.append(label[i])
                        ax.plot(d, label = f'{label[i]}', c = colors[int(np.where(Uniqs == label[i])[0])])
                    else:
                        ax.plot(d, c = colors[int(np.where(Uniqs == label[i])[0])])
                    
    ax.legend(fontsize = SIZE-3)
    ax.tick_params(labelsize = SIZE - 2)
    ax.set_xlabel('Epoch', fontsize = SIZE)
    sns.despine()
    
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            f"Multiple_Loss_graph_M_{int(time.time())}.png"), facecolor='white', bbox_inches='tight', dpi = 300)
    
#    if Second_axis != []:
#        ax2 = ax.twinx()
#        for k in Second_axis:
#            ax2.plot(hist[k], label = k, color = 'slategrey')
        #ax2.legend(loc = 'upper left')
#        ax2.set_ylabel("Learning rate",color="slategrey")
#        h1, l1 = ax.get_legend_handles_labels()
#        h2, l2 = ax2.get_legend_handles_labels()
#        ax.legend(h1+h2, l1+l2, loc=0)
        
        
        
        
def Plotting_side_by_side(ind = 0,save = False, **kwargs):
    Dataset, name, T, Oc_tar, Oc_tr = Compute_dataset_for_plot(ind = ind, **kwargs)
    cmap = plt.get_cmap('seismic')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1, ax2 = axes
    vmin = float(min(Dataset.Mod_melt.min(), Dataset.meltRate.min()))
    vmax = float(max(Dataset.Mod_melt.max(), Dataset.meltRate.max()))
    norm = MidpointNormalize( midpoint = 0 )
    a = Dataset.Mod_melt.plot(ax = ax1,add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
    ax1.set_title('Modeled melt rates'.format(T))
    Dataset.meltRate.plot(ax = ax2, add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
    ax2.set_title('"Real" melt rates'.format(T))
    plt.suptitle('Modeled vs "real" melt rates for t = {} month \n (NN trained on {} applied to {})'.format(T, Concat_Oc_names(Oc_tr), Oc_tar), y=1.035)
    #plt.tight_layout()
    cbar = plt.colorbar(a, cmap = cmap, ax = axes, label = 'Melt rate (m/s)', location = 'bottom', extend='both', fraction=0.16, pad=0.15)#, shrink = 0.6
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            'Side_side_M_{}_t={}.png'.format(name, T)), facecolor='white', bbox_inches='tight')
    return np.array([Dataset.Mod_melt.min(), Dataset.meltRate.min()]), np.array([Dataset.Mod_melt.max(), Dataset.meltRate.max()])


def Compute_dataset_for_plot(ind, Epoch = 4, Ocean_trained = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', 
             Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1, Compute_at_t = 1, T = 0, Extra_n = [], Exact = 1):
    Models_p, _ = Get_model_path_condition(Epoch, Ocean_trained, Type_trained, Exact = Exact, Extra_n = Extra_n)
    if ind >= len(Models_p):
        ind = len(Models_p) - 1
    Model_p = Models_p[ind]
    print(Model_p)
    Model_name = Model_p.split('/')[-1]
    EpochM, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
    Dataset = Compute_datas(Model_p, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = 1, T = T)
    if Dataset != None:
        return Dataset, Model_name, T, Ocean_target, Ocean_trained
    else:
        print('Not found')
        
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def Concat_Oc_names(Strs):
    if type(Strs) == list:
        return 'Ocean' + '-'.join(i[-1] if 'Ocean' in i else re.findall('CPL_(\w+)_rst', i)[0] for i in Strs)
    else:
        return Strs
    
#def Get_model_path_json(Var, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', 
#Exact = 0, Choix = 0, Neur = None)
def plot_N_side(Model_fn, Attribs : list, ind = 0, Oc_tar = 'Ocean1'
        ,Type_tar = 'COM_NEMO-CNRS', message = 0, T = [0], save = False, Title = []):
    
    Titles = {"iceDraft" : "iceD", "temperatureYZ" : "T-YZ", "salinityYZ" : "S-YZ", }
    s_to_yr = 3600 * 24 * 365
    cmap = plt.get_cmap('seismic')
    nTime = len(list(T))
    fig, axes_t = plt.subplots(nrows=nTime, ncols=len(Attribs) + 1, figsize=(10 * len(Attribs), 3 * nTime), sharex=True, sharey=True)
    plt.subplots_adjust(hspace = 0.15)
    for ind_t, t in enumerate(T):
        if nTime == 1:
            axes = axes_t
        else:
            axes = axes_t[ind_t]
        Datasets, Vars = [], []
        for Index, Att in enumerate(Attribs):
            Files = Get_model_path_json(**Att)
            #print(Files)
            File = Files[ind]
            Config = Get_model_attributes(File)
            Choix, Epoch = Config['Choix'], Config['Epoch']
#(Model,Model_path, Choix, Ocean_target, Type_tar, Epoch, message,
            Model = Fetch_model(os.path.join(File, f'model_{Epoch}.h5'))
            Dataset = Compute_datas(Model, File, Choix, Oc_tar, 
                            Type_tar, Epoch, message, Compute_at_t = t, Method = Config.get('Method_data'))
            Dataset[['Mod_melt', 'meltRate']] = Dataset[['Mod_melt', 'meltRate']] * s_to_yr
            Datasets.append(Dataset)
            Vars.append(Config['Var_X'])
        mins, maxs = [], []
        for d in Datasets:
            mins.append(float(d.Mod_melt.min()))
            maxs.append(float(d.Mod_melt.max()))
        vmin = min(mins)
        vmax = max(maxs)
        norm = MidpointNormalize( midpoint = 0 )
        for i, d in enumerate(Datasets):
            if i == 0:
                A = d.Mod_melt.plot(ax = axes[0],add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
            else:
                d.Mod_melt.plot(ax = axes[i], add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
            if i == 0:
                
                axes[i].set_ylabel(f"T = {t} month")
            else:
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
            if ind_t == 0:
                axes[i].set_title( f" Var trained : {'_'.join( [ Titles[n] for n in Vars[i] if Config.get('Method_data') == None])} \n {Title[i] if Title != [] else ''}")
                #axes[i].set_title(f"{} = month", loc = 'left')
            else:
                axes[i].set_title('')
            if ind_t != 0 or i != 0:
                axes[i].set_xlabel('')
        d.meltRate.plot(ax = axes[-1], add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
        axes[-1].set_title('')
        axes[-1].set_ylabel('')
        axes[-1].set_xlabel('')

        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        cbar = plt.colorbar(A, cmap = cmap, ax = axes, label = 'Melt rate (m/yr)', location = 'right', extend='both', anchor = (-0.3, 0), ticks = ticks)#, fraction=0.16, pad=0.15)
    #return Datasets
    #fig.supxlabel('x')
    #fig.supylabel('y')
    fig.text(0.45, 0.095, 'X', ha='center')
    fig.text(0.095, 0.5, 'Y', va='center', rotation='vertical')
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            'N_side_M_{}_{}.png'.format(Oc_tar, int(time.time()))), facecolor='white', bbox_inches='tight')
    return Datasets

def plot_N_side_exp(Model_fn, Attribs : list, ind = 0, Oc_tar = 'Ocean1',Type_tar = 'COM_NEMO-CNRS', 
        message = 0, T = [0], save = False, Title = [], sharing = False, Only_reference = False, One_profile = False, Single_type = 'Mean', Display_label = True, Dimensions = None, size = None):
    Titles = {"iceDraft" : "iceD", "temperatureYZ" : "T-YZ", "salinityYZ" : "S-YZ", }
    s_to_yr = 3600 * 24 * 365
    cmap = plt.get_cmap('seismic')
    if One_profile:
        nTime = 1
    else:
        nTime = len(list(T))
    size = len(Attribs) + 1
    if Only_reference:
        size = 1
    if Dimensions != None:
        x, y = Dimensions
    else:
        x, y = 10, 3
        
    if sharing == True:
        fig, axes_t = plt.subplots(nrows=nTime, ncols=size, figsize=(x * len(Attribs), y * nTime), sharex=True, sharey=True)
    else:
        print(sharing)
        fig, axes_t = plt.subplots(nrows=nTime, ncols=size, figsize=(x * len(Attribs), y * nTime), sharex=sharing, sharey=sharing)
   # else:
#        fig, axes_t = plt.subplots(nrows=nTime, ncols=size, figsize=(10 * len(Attribs), 3 * nTime), sharex=False, sharey=False)
    plt.subplots_adjust(hspace = 0.15)
    Datasets = []
    for Index, Att in enumerate(Attribs):
        Files = Get_model_path_json(**Att)
        #print(Files)
        File = Files[ind]
        Config = Get_model_attributes(File)
        Choix, Epoch = Config['Choix'], Config['Epoch']
#(Model,Model_path, Choix, Ocean_target, Type_tar, Epoch, message,
        Model = Fetch_model(os.path.join(File, f'model_{Epoch}.h5'))
        print(f"Started computing for {File}")
        if One_profile:
            Dataset = Compute_datas(Model, File, Choix, Oc_tar, 
                        Type_tar, Epoch, message, Compute_at_t = 'ALL', Method = Config.get('Method_data'))
            
            if Single_type == 'Sum':
                Modded = Dataset.Mod_melt.sum(dim = 'date', skipna= True)
                Real = Dataset.meltRate.sum(dim = 'date', skipna= True)
            if Single_type == 'Mean':
                Modded = Dataset.Mod_melt.mean(dim = 'date', skipna= True)
                Real = Dataset.meltRate.mean(dim = 'date', skipna= True)
                
            Dataset = xr.Dataset()
            Dataset = Dataset.assign(Mod_melt = Modded)
            Dataset = [ Dataset.assign(meltRate = Real) ]
        else:
            
            Dataset = Compute_datas(Model, File, Choix, Oc_tar, 
                        Type_tar, Epoch, message, Compute_at_t = T, Method = Config.get('Method_data'))
        Datasets.append(Dataset)
        print(f"Finished computing for {File}")
    #return Datasets
    
    Plotting_grid = []
    VMins = []
    VMaxs = []
    for Mods in zip(*Datasets):
        Current_grid = []
        Mins, Maxs = [], []
        for Mod in Mods:
            Mod['Mod_melt'] = Mod['Mod_melt'] * s_to_yr
            Mod['meltRate'] = Mod['meltRate'] * s_to_yr
            Current_grid.append(Mod)
            Mins.append(float(Mod.Mod_melt.min()))
            Maxs.append(float(Mod.Mod_melt.max()))
        Plotting_grid.append(Current_grid)
        VMins.append(min(Mins))
        VMaxs.append(max(Maxs))
    for t, Datasets in enumerate(Plotting_grid):
        if nTime == 1:
            axes = axes_t
        else:
            axes = axes_t[t]
        norm = MidpointNormalize( midpoint = 0 )
        vmin, vmax = VMins[t], VMaxs[t]
        if Only_reference:
            d = Datasets[-1]
            d = d.assign_coords({'x':  d.x/1000,
                             'y':  d.y/1000
                            })
            d.meltRate.plot(ax = axes, add_colorbar=True, robust=False, cmap = cmap, norm = norm, extend='min')
            continue
        for i, d in enumerate(Datasets):
            d = d.assign_coords({'x':  d.x/1000,
                             'y':  d.y/1000
                            })
            if i == 0:
                A = d.Mod_melt.plot(ax = axes[0],add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
                
            else:
                d.Mod_melt.plot(ax = axes[i], add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')

            if i == 0:
                if One_profile == True:
                    if Display_label:
                        axes[i].set_ylabel('y(km)', fontsize = SIZE-2)
                        axes[i].set_xlabel('x(km)', fontsize = SIZE-2)
                    else:
                        #axes[i].set_xticklabels([])
                        #axes[i].set_yticklabels([])
                        axes[i].set_xlabel('')
                        axes[i].set_ylabel('')
                        
                    axes[i].tick_params(labelsize = SIZE - 2)
                else:
                    axes[i].set_ylabel(f"T = {T[t]} month")
                    axes[i].set_xlabel('')
            else:
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
                axes[i].set_xticklabels([])
                axes[i].set_yticklabels([])
                axes[i].tick_params(labelsize = SIZE - 2)
            if Title != []:
                if t == 0:
                    axes[i].set_title(f"{Title[i]}", fontsize = SIZE)
                    pass
                #axes[i].set_title( f" Var trained : {'_'.join( [ Titles[n] for n in Vars[i] if Config.get('Method_data') == None])} \n {Title[i] if Title != [] else ''}")
                #axes[i].set_title(f"{} = month", loc = 'left')
                else:
                    axes[i].set_title('')
            if t != 0 or i != 0:
                axes[i].set_xlabel('')
        d.meltRate.plot(ax = axes[-1], add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
        if Title != [] and t == 0:
            axes[-1].set_title(f"{Title[i+1]}", fontsize = SIZE)
        else:
            axes[-1].set_title('')
        axes[-1].set_xticklabels([])
        axes[-1].set_yticklabels([])
        axes[-1].tick_params(labelsize = SIZE - 2)
        axes[-1].set_ylabel('')
        axes[-1].set_xlabel('')
        ticks = np.round(np.linspace(vmin, vmax, 5, endpoint=True), decimals = 1)
        #divider = make_axes_locatable(axes[-1])
        #cax = divider.append_axes('right', size = '5%')
        cbar = plt.colorbar(A, cmap = cmap, ax = axes, label = 'Melt rate (m/yr)', location = 'right', extend='both', anchor = (-0.3, 0), ticks = ticks)#, fraction=0.16, pad=0.15)
        #cbar = plt.colorbar(A, cmap = cmap, ax = axes, label = 'Melt rate (m/yr)', extend='both', anchor = (-0.3, 0), ticks = ticks, cax = cax)#, fraction=0.16, pad=0.15)
        #tick_locator = ticker.MaxNLocator(nbins = 4)
        #cbar.locator = tick_locator
        #cbar.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        #cbar.update_ticks()
        cbar.ax.tick_params(labelsize=SIZE-2)
        if Dimensions == None:
            cbar.set_label(label = 'Melt rate (m/yr)', size = SIZE)
        else:
            cbar.set_label(label = 'Melt rate (m/yr)', size = SIZE)
    #fig.supxlabel('x')
    #fig.supylabel('y')
    #fig.text(0.45, 0.095, 'X', ha='center')
    #fig.text(0.095, 0.5, 'Y', va='center', rotation='vertical')
    if Dimensions == None:
        sns.despine()
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            'N_side_M_{}_{}.png'.format(Oc_tar, int(time.time()))), facecolor='white', bbox_inches='tight')
    return Datasets

def Plot_Models_per_epoch(NN_attrib = {}, Neurones = [],Attribs = [], **kwargs):

    Dicts = []
    RMSEs = []
    Ts = []
    Params = []
    Neurs = []
    for Neur in Neurones:
        Models = Get_model_path_json(Neur = Neur, **NN_attrib)
        for Mod_p in Models:
            Sub_mods = glob.glob(os.path.join(Mod_p, ''))
            for Mod in Sub_mods:
                Param, T, RMSE, Neurs = Compute_RMSEs_Total_Param(Models_paths = Mod, **kwargs)
                Name = Mod.split('/')[-1]
                Epoch = re.findall('model_(\d+).h5', Name)[0]
                Config = Get_model_attributes(Mod_p)
                Dict = {}
                for Attrib in Attribs:
                    Dict[Attrib] = Config[Attrib]
                Dicts.append(Dict)
                RMSEs.append(RMSE)
                Params.append(Param)
                

def Plot_spatial_RMSE(Model_fn, NN_attrib, Oc_tar = 'Ocean1', ind = 0,
        Type_tar = 'COM_NEMO-CNRS', message = 0, save = False, Title = [], Type = 'Mean'):
    #Titles = {"iceDraft" : "iceD", "temperatureYZ" : "T-YZ", "salinityYZ" : "S-YZ"}
    s_to_yr = 3600 * 24 * 365
    if Type == 'CumSum':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots()
    
    cmap = plt.get_cmap('seismic')
    
    Datasets = []
    File = Get_model_path_json(index = ind, **NN_attrib)[0]
    Config = Get_model_attributes(File)
    Choix, Epoch = Config['Choix'], Config['Epoch']
    Model = Fetch_model(os.path.join(File, f'model_{Epoch}.h5'))
    print(f"Started computing for {File}")
    Dataset = Compute_datas(Model, File, Choix, Oc_tar, 
                    Type_tar, Epoch, message, Compute_at_t = 'ALL', Method = Config.get('Method_data'))
    print(f"Finished computing for {File}")
    if Type == 'Mean':
        Diff = (Dataset.Mod_melt - Dataset.meltRate).mean(dim = 'date', skipna= True)
    elif Type == 'Abs_mean':
        Diff = abs(Dataset.Mod_melt - Dataset.meltRate).mean(dim = 'date', skipna= True)
    if Type != 'CumSum':
        Diff = Diff.assign_coords({'x':  Diff.x/1000,
                                 'y': Diff.y/1000
                                }) * s_to_yr
        vmin = float(Diff.min())
        vmax = float(Diff.max())
        norm = MidpointNormalize( midpoint = 0 )
        #norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        A = Diff.plot(ax = ax, add_colorbar=False, robust=False, cmap = cmap, norm = norm, vmin = vmin, vmax = vmax,extend='both')
        cbar = plt.colorbar(A, cmap = cmap, label = 'Melt rate (m/yr)', location = 'right', extend='both', anchor = (0, -0.2),
        norm = norm, ticks = ticks)
        #cbar = plt.colorbar(A, cmap = cmap, label = 'Melt rate (m/yr)', location = 'right', extend='both', anchor = (0, -0.2))#, ticks = ticks)
    
    else:
        Summed_melts = Dataset.meltRate.sum(dim = 'date', skipna= True)
        Summed_melts = Summed_melts.assign_coords({'x':  Summed_melts.x/1000,
                         'y': Summed_melts.y/1000
                        }) * s_to_yr
        Summed_modded_melts = Dataset.Mod_melt.sum(dim = 'date', skipna= True)
        Summed_modded_melts = Summed_modded_melts.assign_coords({'x':  Summed_modded_melts.x/1000,
                         'y': Summed_modded_melts.y/1000
                        }) * s_to_yr
        vmax = float(Summed_modded_melts.max())
        vmin = float(Summed_modded_melts.min())
        norm = MidpointNormalize( midpoint = 0 )
        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        
        A = Summed_modded_melts.plot(ax = axes[0],add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
                
        Summed_melts.plot(ax = axes[1], add_colorbar=False, robust=False, cmap = cmap, norm = norm, vmin = vmin, vmax = vmax,extend='both')
        
        cbar = plt.colorbar(A, cmap = cmap, label = 'Melt rate (m/yr)', location = 'right', extend='both', anchor = (0, -0.2),
        norm = norm, ticks = ticks)
        
    Oc_trained = Concat_Oc_names(Config.get('Dataset_train'))
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            'Spatial_error_{}_M_{}_{}_{}.png'.format(Type, Oc_tar,Oc_trained, Config.get('Uniq_id'))), facecolor='white', bbox_inches='tight')
    
    return Dataset