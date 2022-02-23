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
    def __init__(self, Epoch = 2, Neur_seq = '32/64/64/32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', 
             Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish'):
        self.Neur_seq = Neur_seq
        self.Epoch = Epoch
        self.Var_X = Var_X
        self.Var_Y = Var_Y
        self.Dataset_train = Dataset_train
        self.Oc_mod_type = Oc_mod_type
        self.activ_fct = activ_fct
        
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
        
        self.meanX, self.stdX = X_train.mean(), X_train.std() 
        self.meanY, self.stdY = Y_train.mean(), Y_train.std() 

        self.X_train, self.X_valid = (np.array((X_train - self.meanX)/self.stdX), 
                                      np.array((X_valid - self.meanX)/self.stdX))
        
        self.Y_train, self.Y_valid = (np.array((Y_train - self.meanY)/self.stdY), 
                                      np.array((Y_valid - self.meanY)/self.stdY))
        
    def train(self):
        Shape = len(self.Var_X)
        self.Init_mod(Shape)
        self.Prepare_data()
        self.model.fit(self.X_train, self.Y_train,
                   epochs = self.Epoch,
                   batch_size = 32,
                   validation_data = (self.X_valid, self.Y_valid))
        self.Model_save()
        
    def Model_save(self):
        pwd = os.getcwd()
        Uniq_id = int(time.time())

        Name = 'Ep_{}_{}-{}/'.format(self.Epoch, self.Neur_seq, Uniq_id)
        Path = os.path.join(pwd, 'Auto_model', self.Oc_mod_type, '_'.join(self.Dataset_train), Name)
        Make_dire(Path)
        self.model.save(Path + 'Model.h5')
        self.meanX.to_pickle(Path + 'MeanX.pkl')
        self.stdX.to_pickle(Path + 'StdX.pkl')
        np.savetxt(Path + 'MeanY.csv', np.array(self.meanY).reshape(1, ))
        np.savetxt(Path + 'StdY.csv', np.array(self.stdY).reshape(1, ))
        
        