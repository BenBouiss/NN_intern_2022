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

PWD = os.getcwd() 

def Getpath_dataset(Dataset, Oc_mod_type):
    Bet_path = '/bettik/bouissob/'
    return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))

def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
class model_NN():
    def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = ''):
        self.Neur_seq = Neur_seq
        self.Epoch = Epoch
        self.Var_X = Var_X
        self.Var_Y = Var_Y
        self.Dataset_train = Dataset_train
        self.Oc_mod_type = Oc_mod_type
        self.activ_fct = activ_fct
        self.Choix = Norm_Choix
        self.Uniq_id = int(time.time())
        self.verbose = verbose
        self.batch_size = batch_size
        self.Extra_n = Extra_n
        self.Js = dict(self.__dict__)
    def Init_mod(self, Shape):
        Orders = self.Neur_seq.split('_')
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(Shape))
        for Order in Orders:
            self.model.add(tf.keras.layers.Dense(int(Order), activation = self.activ_fct))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam',
                     loss = 'mse',
                    metrics = ['mae', 'mse'])
        
    def Prepare_data(self, Indexs):
        li = []
        for Data in self.Dataset_train:
            li.append(pd.read_csv(Getpath_dataset(Data, self.Oc_mod_type)))
        df = pd.concat(li, ignore_index= True)
        X = df[self.Var_X]
        Y = df[self.Var_Y]
        del df, li
        if Indexs == None:
            X_train = X.sample(frac = 0.8)
            X_valid = X.drop(X_train.index)

            Y_train = Y.loc[X_train.index]
            Y_valid = Y.drop(X_train.index)
            self.Js['Similar_training'] = 0
        else:
            ind_p = os.path.join(os.getcwd(), 'Auto_model/tmp/', '_'.join(self.Dataset_train))
            inds = glob.glob(ind_p + '/*.csv')
            Inds = np.loadtxt(inds[0]).astype(int)
            X_valid = X.loc[Inds]
            Y_valid = Y.loc[Inds]
            X_train = X.drop(Inds)
            Y_train = Y.drop(Inds)
            self.Js['Similar_training'] = inds[0].replace(ind_p + '/ind_', '').replace('.csv', '')
            
        if self.Choix == 0:
            self.meanX, self.stdX = X_train.mean(), X_train.std() 
            self.meanY, self.stdY = Y_train.mean(), Y_train.std() 

            self.X_train, self.X_valid = (np.array((X_train - self.meanX)/self.stdX), 
                                          np.array((X_valid - self.meanX)/self.stdX))

            self.Y_train, self.Y_valid = (np.array((Y_train - self.meanY)/self.stdY), 
                                          np.array((Y_valid - self.meanY)/self.stdY))
        elif self.Choix == 1:
            self.maxX, self.minX = X_train.max(), X_train.min()
            self.maxY, self.minY = Y_train.max(), Y_train.min()
            self.X_train, self.X_valid = (np.array((X_train - self.minX)/(self.maxX - self.minX)), 
                                          np.array((X_valid - self.minX)/(self.maxX - self.minX)))
            self.Y_train, self.Y_valid = (np.array((Y_train - self.minY)/(self.maxY - self.minY)), 
                                          np.array((Y_valid - self.minY)/(self.maxY - self.minY)))

    def train(self, Indexs = None):
        Shape = len(self.Var_X)
        self.Init_mod(Shape)
        self.Prepare_data(Indexs)
        self.Data_save()
        saver = CustomSaver(path = self.Path, Epoch_max = self.Epoch)
        Start = time.perf_counter()
        Mod = self.model.fit(self.X_train, self.Y_train,
                   callbacks=[saver],
                   epochs = self.Epoch,
                   batch_size = self.batch_size,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose)
        self.Js['Training_time'] = time.perf_counter() - Start
        del self.X_train, self.Y_train, self.X_valid, self.Y_valid
        self.Model_save(Mod)
        
    def Data_save(self):
        self.Name = 'Ep_{}_N_{}_Ch_{}-{}_Ex_{}/'.format(self.Epoch, self.Neur_seq, self.Choix, self.Uniq_id, self.Extra_n)
        self.Path = os.path.join(PWD, 'Auto_model', self.Oc_mod_type, '_'.join(self.Dataset_train), self.Name)
        Make_dire(self.Path)
        if self.Choix == 0 :
            self.meanX.to_pickle(self.Path + 'MeanX.pkl')
            self.stdX.to_pickle(self.Path + 'StdX.pkl')
            np.savetxt(self.Path + 'MeanY.csv', np.array(self.meanY).reshape(1, ))
            np.savetxt(self.Path + 'StdY.csv', np.array(self.stdY).reshape(1, ))
        elif self.Choix == 1:
            self.maxX.to_pickle(self.Path + 'MaxX.pkl')
            self.minX.to_pickle(self.Path + 'MinY.pkl')
            np.savetxt(self.Path + 'MaxY.csv', np.array(self.maxY).reshape(1, ))
            np.savetxt(self.Path + 'MinY.csv', np.array(self.minY).reshape(1, ))
            
    def Model_save(self, Mod):
        self.model.save(self.Path + 'model_{}.h5'.format(self.Epoch))
        hist = self.model.history
        #json.dump(hist, open(self.path + 'History_log.json'), 'w')
        with open(self.Path + 'TrainingHistory', 'wb') as file_p:
            pickle.dump(Mod.history, file_p)
        with open(self.Path + 'config.json', 'w') as f:
            json.dump(self.Js, f)
        

class Sequencial_training():
    def __init__(self, Model):
        self.Model = Model
        self.Tot = []
    def Train_more_Models(self, Model_list = ['COM_NEMO-CNRS'], **kwargs):
        for Mod_name in Model_list:
            self.training(Oc_mod_type = Mod_name, **kwargs)
    def training(self, training_extent = 1, verbose = 1, Verify = 1,message = 1,
                 Standard_train = ['32_64_64_32'], Exact = 0,Similar_training = 0, **kwargs):

        #Neur_seqs.extend(Hyp_param_list(0, training_extent))
        if training_extent == 0:
            Neur_seqs = Standard_train
        else:
            Neur_seqs = []
            [Neur_seqs.extend(Better_neur_gen(i+1)) for i in range(training_extent)]
            if Verify == 1:
                Neur_seqs = self.Verify_current_regiment(Neur_seqs, self.Model, Exact, message, **kwargs)
        if Neur_seqs == []:
            print('No need for training for config : {} epoch and training_extent = {}'.format(self.Model(**kwargs).Epoch, training_extent))
        else:
            print('Projected training regiment :\n {}'.format(Neur_seqs))
        
        if Similar_training == 1:
            Indexs = 1
            self.Generate_training_index(**kwargs)
        else:
            Indexs = None
        Time_elap = float(0)
        for ind, Neur in enumerate(Neur_seqs):
            Model = self.Model(Neur_seq = Neur, verbose = verbose, **kwargs)
            print("Starting training for neurone : {}, {}/{} (Previous step : {:.3f} s)".format(Neur, ind, len(Neur_seqs), Time_elap))
            Start = time.perf_counter()
            Model.train(Indexs)
            Time_elap = time.perf_counter() - Start
    def Neur_seq_preview(self, training_extent):
        Neur_seqs = []
        [Neur_seqs.extend(Better_neur_gen(i+1)) for i in range(training_extent)]
        return Neur_seqs
    
    def Verify_current_regiment(self, Neur_seqs, Model, Exact,message, **kwargs):
        Template = Model(**kwargs)
        #Current_Files, _ = Get_model_path_condition(Template.Epoch, '_'.join(Template.Dataset_train), Template.Oc_mod_type, Exact = 0)
        Current_Files = Get_model_path_json(Template.Var_X, Template.Epoch, '_'.join(Template.Dataset_train), 
                    Template.Oc_mod_type, Exact = Exact, Choix = Template.Choix, Extra_n = Template.Extra_n)
        for file in Current_Files:
            name = file.split('/')[-1]
            EpochM, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', name)[0]
            if (Neur in Neur_seqs) and os.path.isfile(file + '/model_{}.h5'.format(EpochM)):
                Neur_seqs.remove(Neur)
                if message == 1:
                    print('Neur removed : {} because of {}'.format(Neur, file))
        return Neur_seqs
    
    def Test_verify(self, Ext = 4, **kwargs):
        Neur_seqs = self.Neur_seq_preview(Ext)
        N_Neur = self.Verify_current_regiment(Neur_seqs,self.Model ,**kwargs)
        return N_Neur
    
    def Generate_training_index(self, **kwargs):
        Template = self.Model(**kwargs)
        Path = os.getcwd() + '/Auto_model/tmp/'+ '_'.join(Template.Dataset_train) + '/'
        Make_dire(Path)
        if len(glob.glob(Path + '*.csv')) == 0 :
            li = []
            for data in Template.Dataset_train:
                li.append(pd.read_csv(Getpath_dataset(data, Template.Oc_mod_type)))
            df = pd.concat(li, ignore_index= True)
            index = df.sample(frac = 0.2).index
            np.savetxt(Path + 'ind_{}.csv'.format(int(time.time())), index.to_numpy())
        
        
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, path, Epoch_max):
        self.path = path
        self.Epoch_max = Epoch_max
    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 != self.Epoch_max:
            self.model.save(self.path + "model_{}.h5".format(epoch + 1))
            
            
                        
            
def Hyp_param_list2(Ind, Max): ### Initial fct ###
    List = ['1', '4', '16', '32', '64']
    string = []
    Possible = List[min(4, Max - 1 + Ind) :min(Max + 1 + Ind, len(List))]
    if Ind == Max:
        return Possible
    else:
        Next = Hyp_param_list(Ind + 1, Max)
        return ['_'.join([j, i]) for i in Next for j in Possible]
    
def Verify_string_tuple(Seqs, extent):
    List = ['1', '4', '8','16', '32', '64', '96', '128']
    seqsT = list(Seqs)
    Min = int(Seqs[0][0])
    for ind, seq in enumerate(Seqs):
        removed = 0
        int_list = [int(i) for i in seq]
        Max = int_list[0]
        for ints in int_list:
            if ints<Max:
                seqsT.remove(seq)
                removed = 1
                break
            else:
                Max = ints
        if extent >= 3 and removed == 0:
            if int_list.count(Min) < int(extent)/2 and int_list.count(Min)!=0:
                seqsT.remove(seq)
    return seqsT
            
def Better_neur_gen(Extent):
    List = ['1', '4', '8','16', '32', '64', '96', '128'] 
    li = List[min(Extent-1 + int(Extent/2), min(len(List)-2, len(List) - 4 + int(Extent/5))):Extent + 3 + int(2/Extent) + int(Extent/5)]
    permut = list(itertools.product(li, repeat = Extent))
    Seq = Verify_string_tuple(permut, Extent)
    return ['_'.join(i) for i in Seq]
    
    
def Fetch_data(Ocean_target, Type_tar):
    li = []
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
        #for f in files:
        #    os.remove(f)
        #os.rmdir(model_path)
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

def Normalizer(d, mod_p, Choix, Data_Norm):
    if str(Choix) == '0':
        MeanX, MeanY, StdX, StdY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MeanX)/StdX).reshape(-1, len(Var_X), )
    if str(Choix) == '1':
        MaxX, MinX, MaxY, MinY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MinX)/(MaxX - MinX)).reshape(-1, len(Var_X), )
    return X, Var_X
                     
                     
def Compute_rmse(Tar, Mod):
    return np.sqrt(np.mean((Mod-Tar)**2))
                     
def Compute_data_from_model(Model_path, Choix, Ocean_target, Type_tar, Epoch, message, Compute_at_t = False, T = 0):
    Yr_t_s = 3600 * 24 * 365 # s/yr
    Rho = 920 #Kg/m**3
    Horiz_res = 2 #km/pix
    S = (Horiz_res*10**3) ** 2  #m**2/pix
    Data = Fetch_data(Ocean_target, Type_tar)
    Model = Fetch_model(Model_path, name = 'model_{}.h5'.format(Epoch))
    if Model == None:
        return None
    tmx = int(Data.loc[len(Data) - 1].date + 1)
    Data_norm = Fetch_model_data(Model_path, Choix)
    Var_X = Data_norm[len(Data_norm) - 1]
    if message:
        print('Data variables used : {}'.format(' '.join(Var_X)))
    Melts, Modded_melts = [], []
    if not Compute_at_t :
        for t in range(tmx):
            if message :
                if (t+1)%int(tmx/5) == 0:
                    print('Starting {} / {}'.format(t+1, tmx) , end='\r')
            Cur = Data.loc[Data.date == t].reset_index(drop = True)
            X, Var_X = Normalizer(Cur, Model_path, Choix, Data_norm)
            if str(Choix) == '0':
                Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
                Cur['Mod_melt'] = Y
            if str(Choix) == '1':
                Y = np.array(Model(X) * (Data_norm[2] - Data_norm[3]) + Data_norm[3])
                Cur['Mod_melt'] = Y
                
            Melts.append(Cur['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
            Modded_melts.append(Cur['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
        Melts = np.array(Melts)
        Modded_melts = np.array(Modded_melts)
        return Melts, Modded_melts, Compute_rmse(Melts, Modded_melts), Model.count_params()
    else:
        Cur = Data.loc[Data.date == T].reset_index(drop = True)
        X, Var_X = Normalizer(Cur, Model_path, Choix, Data_norm)
        if str(Choix) == '0':
            Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
            Cur['Mod_melt'] = Y
        if str(Choix) == '1':
            Y = np.array(Model(X) * (Data_norm[2] - Data_norm[3]) + Data_norm[3])
            Cur['Mod_melt'] = Y
            
        Data = Cur.set_index(['date', 'y', 'x'])
        Dataset = Data.to_xarray()
        return Dataset
    
def Compute_data_for_plotting(Epoch = 4, Ocean_trained = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Multi_comp = 0,
             Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1, index = None, Time = False, NN_attributes = {}):
    #Models_paths, path = Get_model_path_condition(Epoch, Ocean_trained, Type_trained, Exact, Extra_n)
    Models_paths = Get_model_path_json(Epoch = Epoch, Ocean = Ocean_trained, Type_trained = Type_trained, **NN_attributes)
    if index != None:
        if index >= len(Models_paths):
            index = len(Models_paths) - 1
        Models_paths = [Models_paths[index]]
        print(f'{Models_paths} \n', end = '\r')
    RMSEs, Params, Neurs, Melts, Modded_melts, Oc_mask, t = [], [], [], [], [], [], []
    #print(path + '/Ep_{}'.format(Epoch))
    if type(Ocean_target) != list:
        Ocean_target = [Ocean_target]
    for Oc in Ocean_target:
        Oc_m = int(Oc[-1])
        for ind, model_p in enumerate(Models_paths):
            print('Starting {}/{} model {}'.format(ind + 1, len(Models_paths), model_p.split('/')[-1]), end = '\r')
            Model_name = model_p.split('/')[-1]
            EpochM, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
            Comp = Compute_data_from_model(model_p, Choix, Oc, Type_tar, Epoch, message)
            if Comp != None:
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
                if Time:
                    with open(model_p + '/config.json') as json_file:
                        data = json.load(json_file)
                    t.append(data['Training_time'])
    if index == None:
        return np.array(RMSEs), np.array(Params), Melts, Modded_melts, Neurs, Oc_mask, Ocean_trained, Ocean_target, Epoch, t
    else:
        return np.array(RMSEs), np.array(Params), Melts, Modded_melts, Neurs, Oc_mask, model_p, Ocean_trained, Ocean_target
    
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
        plt.savefig(os.path.join(os.getcwd(), 'Image_output', 'RMSE_param_Ep{}_Tr{}_Tar{}_{}'.format(Ep, 
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
        plt.savefig(os.path.join(os.getcwd(), 'Image_output', 'Melt_time_fct_M_{}_{}={}.png'.format(name, 
                    Concat_Oc_names(Oc_train), Concat_Oc_names(Oc_tar))),facecolor='white')
        
def Plot_Melt_to_Modded_melt(save = False, **kwargs):
    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask, Oc_tr, Oc_tar, _ = Compute_data_for_plotting(**kwargs)
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
        fig.savefig(os.path.join(os.getcwd(), 'Image_output', 'Line_real_Modeled_N{}_Tr{}_Tar{}'.format(Neurs, 
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
    
def Plot_loss_model(ind = 0, **kwargs):
    #Models_p, _ = Get_model_path_condition(**kwargs)
    Models_p = Get_model_path_json(**kwargs)
    if ind >= len(Models_p):
        ind = len(Models_p) - 1
    Model_p = Models_p[ind]
    hist = pd.read_pickle(Model_p + '/TrainingHistory')
    plt.plot(hist['loss'])
    for k in hist.keys():
        plt.plot(hist[k], label = k)
    plt.legend()
    plt.title('Loss graph for model : {}'.format(Model_p.split('/')[-1]))
    plt.show()
    
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
        fig.savefig(os.path.join(os.getcwd(), 'Image_output', 
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
        fig.savefig(os.path.join(os.getcwd(), 'Image_output', 
            'N_side_M_{}_t={}.png'.format(name, T)), facecolor='white', bbox_inches='tight')
    return Datasets
#    cmap = plt.get_cmap('seismic')
#    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#    ax1, ax2 = axes
#   vmin = float(min(Dataset.Mod_melt.min(), Dataset.meltRate.min()))
#  vmax = float(max(Dataset.Mod_melt.max(), Dataset.meltRate.max()))
#    norm = MidpointNormalize( midpoint = 0 )
#    a = Dataset.Mod_melt.plot(ax = ax1,add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
#    ax1.set_title('Modeled melt rates'.format(T))
#    Dataset.meltRate.plot(ax = ax2, add_colorbar=False, robust=False, vmin = vmin, vmax = vmax, cmap = cmap, norm = norm, extend='min')
#    ax2.set_title('"Real" melt rates'.format(T))
#    plt.suptitle('Modeled vs "real" melt rates for t = {} month \n (NN trained on {} applied to {})'.format(T, Concat_Oc_names(Oc_tr), Oc_tar), y=1.035)
    #plt.tight_layout()
#    cbar = plt.colorbar(a, cmap = cmap, ax = axes, label = 'Melt rate (m/s)', location = 'bottom', extend='both', fraction=0.16, pad=0.15)#, shrink = 0.6
        