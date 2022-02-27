import os
import numpy as np
import tensorflow as tf
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re 
import sys
import json

def Getpath_dataset(Dataset, Oc_mod_type):
    Bet_path = '/bettik/bouissob/'
    return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))

def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
class model_NN():
    def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32):
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
        
    def Prepare_data(self):
        li = []
        for Data in self.Dataset_train:
            li.append(pd.read_csv(Getpath_dataset(Data, self.Oc_mod_type)))
        df = pd.concat(li, ignore_index= True)
        X = df[self.Var_X]
        Y = df[self.Var_Y]
        del df, li
        X_train = X.sample(frac = 0.8)
        X_valid = X.drop(X_train.index)
        
        Y_train = Y.loc[X_train.index]
        Y_valid = Y.drop(X_train.index)
        
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

    def train(self):
        Shape = len(self.Var_X)
        self.Init_mod(Shape)
        self.Prepare_data()
        self.Data_save()
        saver = CustomSaver(path = self.Path, Epoch_max = self.Epoch)
        self.model.fit(self.X_train, self.Y_train,
                   callbacks=[saver],
                   epochs = self.Epoch,
                   batch_size = self.batch_size,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose)
        del self.X_train, self.Y_train, self.X_valid, self.Y_valid
        self.Model_save()
        
    def Data_save(self):
        self.Name = 'Ep_{}_N_{}_Ch_{}-{}/'.format(self.Epoch, self.Neur_seq, self.Choix, self.Uniq_id)
        self.Path = os.path.join(os.getcwd(), 'Auto_model', self.Oc_mod_type, '_'.join(self.Dataset_train), self.Name)
        pwd = os.getcwd()
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
            
    def Model_save(self):
        self.model.save(self.Path + 'model_{}.h5'.format(self.Epoch))
        hist = self.model.history
        json.dump(hist, opne(self.path + 'History_log.json'), 'r')

        

class Sequencial_training():
    def __init__(self, Model, Epoch = 2):
        self.Model = Model
        self.Epoch = Epoch
        self.Tot = []
    def training(self, training_extent, verbose = 1, **kwargs):
        Neur_seqs = []
        [Neur_seqs.extend(Hyp_param_list(0, i+1 )) for i in range(training_extent)]
        #Neur_seqs.extend(Hyp_param_list(0, training_extent))
        print('Projected training regiment :\n {}'.format(Neur_seqs))
        for ind, Neur in enumerate(Neur_seqs):
            Model = self.Model(Neur_seq = Neur, Epoch = self.Epoch, verbose = verbose, **kwargs)
            
            print('Starting training for neurone : {}, {}/{}'.format(Neur, ind, len(Neur_seqs)))
            Model.train()
            
    def Neur_seq_preview(self, training_extent):
        Neur_seqs = []
        [Neur_seqs.extend(Hyp_param_list(0, i)) for i in range(training_extent + 1)]
        return Neur_seqs
    
        
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, path, Epoch_max):
        self.path = path
        self.Epoch_max = Epoch_max
    def on_epoch_end(self, epoch, logs={}):
        if epoch != self.Epoch_max:
            self.model.save(self.path + "model_{}.h5".format(epoch))
            
            
                        
            
def Hyp_param_list(Ind, Max):
    List = ['1', '4', '16', '32', '64']
    string = []
    Possible = List[min(4, Max - 1 + Ind) :min(Max + 1 + Ind, len(List))]
    if Ind == Max:
        return Possible
    else:
        Next = Hyp_param_list(Ind + 1, Max)
        return ['_'.join([j, i]) for i in Next for j in Possible]
    
    
    
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
        return None
                     
                     
def Fetch_model_data(model_path, Choix):
    if Choix == '0':
        StdY = np.loadtxt(model_path + '/' + 'StdY.csv')
        MeanY = np.loadtxt(model_path + '/' + 'MeanY.csv')
        MeanX = pd.read_pickle(model_path + '/' + 'MeanX.pkl')
        StdX = pd.read_pickle(model_path + '/' + 'StdX.pkl')
    return MeanX, MeanY, StdX, StdY, MeanX.index

def Normalizer(d, mod_p, Choix, Data_Norm):
    if Choix == '0':
        MeanX, MeanY, StdX, StdY, Var_X = Data_Norm
        X = d[Var_X]
        X = np.array((X - MeanX)/StdX).reshape(-1, 5, )
    return X, Var_X
                     
                     
def Compute_rmse(Tar, Mod):
    return np.sqrt(np.mean((Mod-Tar)**2))
                     
def Compute_data_from_model(Model_path, Choix, Ocean_target, Type_tar, Epoch, message):
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
    for t in range(tmx):
        if message :
            if (t+1)%int(tmx/5) == 0:
                print('Starting {} / {}'.format(t+1, tmx) , end='\r')
        Cur = Data.loc[Data.date == t].reset_index(drop = True)
        X, Var_X = Normalizer(Cur, Model_path, Choix, Data_norm)
        if Choix == '0':
            Y = np.array((Model(X) * Data_norm[3]) + Data_norm[1])
            Cur['Mod_melt'] = Y
        Melts.append(Cur['meltRate'].sum() * Yr_t_s * Rho * S / 10**12)
        Modded_melts.append(Cur['Mod_melt'].sum() * Yr_t_s * Rho * S / 10**12)
    Melts = np.array(Melts)
    Modded_melts = np.array(Modded_melts)
    return Melts, Modded_melts, Compute_rmse(Melts, Modded_melts), Model.count_params()

def Compute_data_for_plotting(Epoch = 14, Ocean_trained = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', 
             Ocean_target = 'Ocean1', Type_tar = 'COM_NEMO-CNRS', message = 1):
    pwd = os.getcwd()
    path = os.path.join(pwd, 'Auto_model', Type_trained, '_'.join(Ocean_trained) if type(Ocean_trained) == list else Ocean_trained)
    Models_paths = glob.glob(path + '/Ep_{}*'.format(Epoch))
    RMSEs, Params, Neurs, Melts, Modded_melts, Oc_mask = [], [], [], [], [], []
    print(path + '/Ep_{}'.format(Epoch))
    if type(Ocean_target) != list:
        Ocean_target = [Ocean_target]
    for Oc in Ocean_target:
        Oc_m = int(Oc[len(Oc) - 1])
        for ind, model_p in enumerate(Models_paths):
            print('Starting {}/{} model'.format(ind + 1, len(Models_paths)), end = '\r')
            Model_name = model_p.split('/')
            Model_name = Model_name[len(Model_name) - 1]
            Epoch, Neur, Choix = re.findall('Ep_(\d+)_N_(\w+)_Ch_(\d+)', Model_name)[0]
            Comp = Compute_data_from_model(model_p, Choix, Oc, Type_tar, Epoch, message)
            if Comp != None:
                Melt, Modded_melt, RMSE, Param = Comp
                RMSEs.append(RMSE)
                Params.append(Param)
                Neurs.append(Neur)
                Melts = np.append(Melts, Melt)
                Modded_melts = np.append(Modded_melts, Modded_melt)
                Oc_mask = np.append(Oc_mask, np.full_like(Melt, Oc_m))
    return np.array(RMSEs), np.array(Params), Melts, Modded_melts, Neurs, Oc_mask
    
def Plot_RMSE_to_param(**kwargs):
    RMSEs, Params, _, _, Neurs, _ = Compute_data_for_plotting(**kwargs)
    plt.scatter(Params, RMSEs)
    return RMSEs, Params, Neurs
    
def Plot_Melt_to_Modded_melt(**kwargs):
    RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask = Compute_data_for_plotting(**kwargs)
    fig, ax = plt.subplots()

    for v in np.unique(Oc_mask):
        idx = np.where(Oc_mask == v)
        ax.scatter(Modded_melts[idx], Melts[idx], s = 1.5, label = 'Ocean{}'.format(v))
        #ax.legend(Oc_mask)
    ax.legend(loc = 'upper right')
    plt.show()
    return RMSEs, Params, Melts, Modded_melts, Neurs, Oc_mask