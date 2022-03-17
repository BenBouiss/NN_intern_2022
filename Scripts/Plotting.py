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


def Plot_RMSE_to_param(save = False, **kwargs):
    RMSEs, Params, _, _, Neurs, _, Oc_tr, Oc_tar, Ep, t = Compute_data_for_plotting(**kwargs)
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
def Plot_total_RMSE_param(save = False, message_p = 1, **kwargs):

    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, _, t = Compute_data_for_plotting(**kwargs)
    if len(Params) == len(Melts):
        df = pd.DataFrame()
        df['Melts'] = Melts
        df['Modded_melts'] = Modded_melts
        df['Params'] = Params
        df['Neurs'] = Neurs
        RMSE = []
        Neurs = []
        Param = np.unique(Params)
        for P in Param:
            Cur = df.loc[df.Params == P]
            RMSE.append(Compute_rmse(np.array(Cur.Melts), np.array(Cur.Modded_melts)))
            Neur.append(np.unique(Cur.Neurs))
    else:
        RMSE = RMSEs
        Param = Params
    plt.scatter(Param, RMSE)
    return Param, RMSE, Neur
    
def Plot_Melt_time_function(ind = 0, save = False, **kwargs):
    RMSE, _, Melts, Modded_Melts, _, _, name, Oc_train, Oc_tar = Compute_data_for_plotting(index = ind, **kwargs)
    name = name.split('/')[-1]
    x = np.arange(1, len(Modded_Melts) + 1)
    plt.plot(x, Melts, label = 'Real melt')
    plt.plot(x, Modded_Melts, label = 'Modeled melt')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Mass lost(Gt/yr)')
    #print(Concat_Oc_names(Oc_tar))
    #print(Concat_Oc_names(Oc_train))
    plt.title('Modeling melt rates of {} \n (NN trained on {})'.format(Concat_Oc_names(Oc_tar), Concat_Oc_names(Oc_train)))
    plt.legend()
    print(RMSE)
    if save:
        plt.savefig(os.path.join(PWD, 'Image_output', 'Melt_time_fct_M_{}_{}={}.png'.format(name, 
                    Concat_Oc_names(Oc_train), Concat_Oc_names(Oc_tar))),facecolor='white')
        
def Plot_Melt_to_Modded_melt(save = False,Compute_at_ind = False, **kwargs):
    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, *_ = Compute_data_for_plotting(Compute_at_ind = Compute_at_ind, **kwargs)
    fig, ax = plt.subplots()
    Vmin = min(np.append(Melts, Modded_melts))
    Vmax = max(np.append(Melts, Modded_melts))
    for v in np.unique(Oc_mask):
        idx = np.where(Oc_mask == v)
        ax.scatter(Modded_melts[idx], Melts[idx], s = 3, alpha = 0.6 ,label = 'Ocean{}'.format(v))
        ax.legend(Oc_mask)
    linex, liney = [Vmin, Vmax], [Vmin, Vmax]
    ax.plot(linex, liney, c = 'red', ls = '-', linewidth = 0.8)
    ax.legend(loc = 'upper left')
    plt.xlabel('Modeled melt rate (Gt/yr)')
    plt.ylabel('"Real" melt rate(Gt/yr)')
    plt.title('Modeling melt rates of {} \n (NN trained on {})'.format(Concat_Oc_names(Oc_tar), Concat_Oc_names(Oc_tr)))
    plt.show()
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 'Line_real_Modeled_N{}_Tr{}_Tar{}'.format(Neurs, 
                    Concat_Oc_names(Oc_tr), Concat_Oc_names(Oc_tar))), facecolor = 'white')
    return RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask

def Get_model_path_condition(Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = 0, Extra_n = []):
    path = os.path.join(PWD, 'Auto_model', Type_trained, '_'.join(Ocean) if type(Ocean) == list else Ocean)
    Model_paths = glob.glob(path + '/Ep_*'.format(Epoch))
    if Exact == 0:
        N_paths = [p for p in Model_paths if int(re.findall('Ep_(\d+)', p.split('/')[-1])[0]) >= Epoch]
    else:
        N_paths = [p for p in Model_paths if int(re.findall('Ep_(\d+)', p.split('/')[-1])[0]) == Epoch
                  and re.findall('Ex_(\w+)', p.split('/')[-1]) == Extra_n ]
    #print('For condition Epoch = {}, list of all matching files : {}'.format(Epoch, [p.split('/')[-1] for p in N_paths]))
    return N_paths, path

def Get_model_path_json(Var = None, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = 0, Extra_n = None, Choix = None, Neur = None, Batch_size = None):
    if type(Ocean) != list:
        Ocean = [Ocean]
    path = os.path.join(PWD, 'Auto_model', Type_trained, '_'.join(Ocean))
    if Exact == 1:
        Model_paths = glob.glob(path + '/Ep_{}*'.format(Epoch))
    else:
        Model_paths = glob.glob(path + '/Ep_*')
    #Mod = list(Model_paths)
    for f in list(Model_paths):
        if os.path.isfile(f + '/config.json'):
            with open(f + '/config.json') as json_file:
                data = json.load(json_file)
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
    return Model_paths
    
def Plot_loss_model(save = False, ind = 0, **kwargs):
    #Models_p, _ = Get_model_path_condition(**kwargs)
    Models_p = Get_model_path_json(**kwargs)
    fig, ax = plt.subplots()
    if ind >= len(Models_p):
        ind = len(Models_p) - 1
    Model_p = Models_p[ind]
    hist = pd.read_pickle(Model_p + '/TrainingHistory')
    plt.plot(hist['loss'])
    with open(Model_p + '/config.json') as json_file:
                data = json.load(json_file)
    for k in hist.keys():
        plt.plot(hist[k], label = k)
    plt.legend(loc = 'upper right')
    plt.title('Loss graph for model : {}'.format(Model_p.split('/')[-1]))
    plt.xlabel('Epoch')
    plt.show()
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            f"Loss_graph_M_{data['Neur_seq']}_{data['Uniq_id']}.png"), facecolor='white', bbox_inches='tight')
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
    Dataset = Compute_data_from_model(Model_p, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = 1, T = T)
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
        return 'Ocean' + '-'.join(i[-1] for i in Strs)
    else:
        return Strs
    
#def Get_model_path_json(Var, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', 
#Exact = 0, Choix = 0, Neur = None)
def plot_N_side(Model_fn, Attribs : list, ind = 0, Oc_tar = 'Ocean1'
        ,Type_tar = 'COM_NEMO-CNRS', message = 0, T = [0], save = False):
    
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
            with open(File + '/config.json') as json_file:
                Config = json.load(json_file)
            Choix, Epoch = Config['Choix'], Config['Epoch']

            Dataset = Compute_data_from_model(File, Choix, Oc_tar, 
                                    Type_tar, Epoch, message, Compute_at_t = True, T = t)
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
                axes[i].set_title('_'.join(Vars[i]))
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
    #fig.supxlabel('x')
    #fig.supylabel('y')
    fig.text(0.45, 0.095, 'X', ha='center')
    fig.text(0.095, 0.5, 'Y', va='center', rotation='vertical')
    if save:
        fig.savefig(os.path.join(PWD, 'Image_output', 
            'N_side_M_{}_t={}.png'.format(Oc_tar, '_'.join(str(T)))), facecolor='white', bbox_inches='tight')
    return Datasets