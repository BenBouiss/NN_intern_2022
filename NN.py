import os
import numpy as np
import tensorflow as tf
import time
import glob

def Getpath_dataset(Dataset, Oc_mod_type):
    Bet_path = '/bettik/bouissob/'
    return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))

def GetPath_Model(Oc_mod):
    pwd = os.getcwd()
    
def Init_mod(Sequence_str, Shape):
    Orders = Sequence_str.split('/')
    model = tf.keras.models.Sequential()
    model.add(.tf.keras.layers.Input(Shape))
    for Order in Orders:
        model.add(tf.keras.layers.Dense(int(Order), activation = 'swish'))
    model.add(tf.keras.layers.Dense(1))
    return model.compile(optimizer='adam',
                 loss = 'mse',
                metrics = ['mae', 'mse'])

def Train_NN(Nbr_Epoch = 3, Sequence_str = '32/64/64/32', Dataset_training = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS',
               Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'Icedraft']):
    
    
    Model = Init_mod(Sequence_str, Shape = len(Var_X))
    for Dataset in Dataset_training:
        
    Model = model.fit(X_train, Y_train,
                   epochs = Nbr_Epoch,
                   batch_size = 32,
                   validation_data = (X_valid, Y_valid))

class model_NN():
    def __init__(Epoch = 2, )