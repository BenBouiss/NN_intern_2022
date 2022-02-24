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
             Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish',
                Norm_Choix = 0):
        self.Neur_seq = Neur_seq
        self.Epoch = Epoch
        self.Var_X = Var_X
        self.Var_Y = Var_Y
        self.Dataset_train = Dataset_train
        self.Oc_mod_type = Oc_mod_type
        self.activ_fct = activ_fct
        self.Choix = Norm_Choix
        
    def Init_mod(self, Shape):
        Orders = self.Neur_seq.split('/')
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
        self.model.fit(self.X_train, self.Y_train,
                   epochs = self.Epoch,
                   batch_size = 32,
                   validation_data = (self.X_valid, self.Y_valid))
        self.Model_save()
        
    def Model_save(self):
        pwd = os.getcwd()
        Uniq_id = int(time.time())

        Name = 'Ep_{}_{}_Ch_{}-{}/'.format(self.Epoch, self.Neur_seq, self.Choix, Uniq_id)
        Path = os.path.join(pwd, 'Auto_model', self.Oc_mod_type, '_'.join(self.Dataset_train), Name)
        Make_dire(Path)
        self.model.save(Path + 'Model.h5')
        if self.Choix == 0 :
            self.meanX.to_pickle(Path + 'MeanX.pkl')
            self.stdX.to_pickle(Path + 'StdX.pkl')
            np.savetxt(Path + 'MeanY.csv', np.array(self.meanY).reshape(1, ))
            np.savetxt(Path + 'StdY.csv', np.array(self.stdY).reshape(1, ))
        elif self.Choix == 1:
            self.maxX.to_pickle(Path + 'MaxX.pkl')
            self.minX.to_pickle(Path + 'MinY.pkl')
            np.savetxt(Path + 'MaxY.csv', np.array(self.maxY).reshape(1, ))
            np.savetxt(Path + 'MinY.csv', np.array(self.minY).reshape(1, ))
        

class Sequencial_training():
    def __init__(self, Model, Epoch = 2):
        self.Model = Model
        self.Epoch = Epoch
    def training(self, training_extent):
        Neur_seqs = []
        #[Neur_seqs.extend(Hyp_param_list(0, i+1 )) for i in range(training_extent)]
        Neur_seqs.extend(Hyp_param_list(0, self.training_extent))
        for Neur in Neur_seqs:
            print('Starting training for neurone : {}'.format(Neur))
            self.Model.Neur_seq = Neur
            self.Model.Epoc = self.Epoch
            self.Model.train()
            
            
def Hyp_param_list(Ind, Max):
    List = ['1', '4', '16', '32', '64']
    string = []
    Possible = List[min(4, Max - 1 + Ind) :min(Max + 1 + Ind, len(List))]
    if Ind == Max:
        return Possible
    else:
        Next = Hyp_param_list(Ind + 1, Max)
        return ['_'.join([j, i]) for i in Next for j in Possible]
            
    