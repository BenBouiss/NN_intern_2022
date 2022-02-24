import os
import numpy as np
import tensorflow as tf
import time
import glob
import pandas as pd

def Getpath_dataset(Dataset, Oc_mod_type):
    Bet_path = '/bettik/bouissob/'
    return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))

def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
class model_NN():
    def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1):
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
                   batch_size = 32,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose)
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
        self.model.save(self.Path + 'model.h5')

        

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
            
    