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
import random 
import tensorflow_model_optimization as tfmot
import tempfile
import math 
import gc

PWD = os.getcwd()
PWD = '/home/bouissob/Code'
print(f'PWD : {PWD}')
Bet_path = '/bettik/bouissob/'


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



def Getpath_dataset(Dataset, Oc_mod_type, Method = None):
    Bet_path = '/bettik/bouissob/'
    if Method == None:
        return os.path.join(Bet_path, 'Data', 'data_{}_{}.csv'.format(Dataset, Oc_mod_type))
    elif Method == 4 or Method == 2 or Method == 5:
        return Bet_path + f'Method_Data/{Oc_mod_type}/Method_{Method}/{Dataset}_lite.csv'
    else:
        return Bet_path + f'Method_Data/{Oc_mod_type}/Method_{Method}/{Dataset}.csv'
    
    
def Make_dire(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

def Generate_Var_name(Str, Extent):
    Min, Max = Extent
    return [f"{Str}_{i}" for i in np.arange(Min, Max)]

def Convert_from_big(li : list):
    if 'Big_T' in li:
        li.remove('Big_T')
        for i in range(40):
            li.append(f'T_{i}')
    if 'Big_S' in li:
        li.remove('Big_S')
        for i in range(40):
            li.append(f'S_{i}')
    return li


class model_NN():
    def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5, Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8, Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, Fraction = None, Fraction_save = None,
Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2, min_delta = 0.007, Pruning = False, Pruning_type = 'Constant', initial_sparsity = 0, target_sparsity = 0.5, Random_seed = None, Spatial_cutting = False, Coordinate_box = None, Spatial_cutting_type = 'box', Time_cutting = False, Time_cutting_type = None, Similar_training = False, Time_percent = 1, Time_period = 1, Downsample_Multiplier = 1):
        self.Neur_seq = Neur_seq
        self.Epoch = Epoch
        self.Var_X = list(Var_X)
        self.Var_Y = [Var_Y] if type(Var_Y) != list else Var_Y
        self.Dataset_train = Dataset_train
        self.Oc_mod_type = Oc_mod_type
        self.activ_fct = activ_fct
        self.Choix = Norm_Choix
        self.Uniq_id = int(time.time())
        self.verbose = verbose
        self.batch_size = batch_size
        self.Extra_n = Extra_n
        self.Cutting = Better_cutting
        self.Drop = Drop
        self.Random_seed = Random_seed
        if Drop != None:
            self.Default_drop = Default_drop
        self.Method_data = Method_data
        self.Method_extent = Method_extent
        self.Scaling_lr = Scaling_lr
        self.TensorBoard_logs = TensorBoard_logs
        if self.Scaling_lr == True:
            self.Frequence_scaling_change = Frequence_scaling_change
            self.Scaling_change = Scaling_change
            self.Scaling_type = Scaling_type
            self.LR_min = LR_min
            if self.Scaling_type == 'Plateau':
                self.LR_Patience = LR_Patience
                self.LR_Factor = LR_Factor
                self.min_delta = min_delta
        self.Spatial_cutting = Spatial_cutting
        self.Time_cutting = Time_cutting
        self.Similar_training = Similar_training
        if self.Spatial_cutting == True:
            self.Coordinate_box = Coordinate_box
            self.Spatial_cutting_type = Spatial_cutting_type
        self.Downsample_Multiplier = Downsample_Multiplier
        if self.Time_cutting == True:
            self.Time_cutting_type = Time_cutting_type
            self.Time_percent = Time_percent 
            self.Time_period = Time_period
        #self.Multi_thread = Multi_thread
        #self.Workers = Workers
        self.Hybrid = Hybrid
        if Hybrid:
            self.Epoch_lim = Epoch_lim
        self.Fraction = Fraction
        if Method_data == 4 or Method_data == 2:
            Big_Var = []
            if 'Big_T' in self.Var_X:
                self.Var_X.remove('Big_T')
                All_name = Generate_Var_name('T', Method_extent)
                Big_Var.append(All_name)
                self.Var_X.extend(All_name)
            if 'Big_S' in self.Var_X:
                self.Var_X.remove('Big_S')
                All_name = Generate_Var_name('S', Method_extent)
                Big_Var.append(All_name)
                self.Var_X.extend(All_name)
            self.Big_Var = Big_Var
        
        self.Fraction_save = Fraction_save
        self.Pruning = Pruning
        if self.Pruning == True:
            self.Pruning_type = Pruning_type
            if self.Pruning_type == 'Polynomial':
                self.initial_sparsity = initial_sparsity
            self.target_sparsity = target_sparsity
        self.Js = dict(self.__dict__)
        self.Callback = []

    def load_dataset(self, Ocean, Model_type, Method, Var_to_load):
        path = Getpath_dataset(Ocean, Model_type, Method)
        return pd.read_csv(path, usecols = Var_to_load)
    
    def Fetch_dataset(self):
        dfT = pd.DataFrame()
        Coord_var = ['x', 'y', 'date']
        Var_to_load = self.Var_X + self.Var_Y + Coord_var
        for Ocean in self.Dataset_train:
            for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),key= lambda x: -x[1])[:10]:
                print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
            print(f'Loading {Ocean} dataset')
            df = self.load_dataset(Ocean, self.Oc_mod_type, self.Method_data, Var_to_load)
            df = self.Apply_modif_to_dataset(df)
            dfT = pd.concat([dfT, df], ignore_index= True)
            del df
        return dfT
    def Apply_modif_to_dataset(self, li : list):
        if self.Time_cutting :
            li = self.Apply_Time_Cutting(li)
        if self.Spatial_cutting:
            li = self.Apply_Spatial_Cutting(li)
        if self.Fraction != None:
            li = li.sample(frac = self.Fraction)
        return li
    
    def Apply_Time_Cutting(self, li : list):
        if self.Time_cutting_type == 'S2End-percent':
            TMAX = int(max(li.date))
            tmax = (1 - self.Time_percent) * TMAX
            tmin = (self.Time_percent) * TMAX
            li = li.loc[(li.date <= tmin) | (li.date >= tmax) ]
        
        if self.Time_cutting_type == 'period':
            li = li.loc[li.date % self.Time_period == 0]
        return li
    
    def Apply_Spatial_Cutting(self, li : list):
        print('Applying spatial cutting')
        if self.Spatial_cutting_type == 'box':
            xmin, xmax = self.Coordinate_box[0:2]
            ymin, ymax = self.Coordinate_box[2:4]
            li = li.loc[(li.x >= xmin) & (li.x <= xmax) & (li.y >= ymin) & (li.y <= ymax)]
        if self.Spatial_cutting_type.lower() == 'downsample':
            X = self.Downsample(self.Downsample_Multiplier, l = np.unique(li.x))
            Y = self.Downsample(self.Downsample_Multiplier, l = np.unique(li.y))
            li = li.loc[li.x.isin(X) & li.y.isin(Y)]
        return li
    
    def Downsample(self, Multiplier : int, l : list):
        new_li = np.array([elem for i, elem in enumerate(l) if i % Multiplier == 0])
        return new_li
    
    
    def Extract_stat_variables(self, df):
        if self.Choix == 0:
            mean = df.mean().to_frame().T
            std = df.std().to_frame().T
            return mean, std
        if self.Choix == 1:
            Max = df.max().to_frame().T
            Min = df.min().to_frame().T
            return Max, Min
        if self.Choix == 2:
            iqr = (df.quantile(0.75) - df.quantile(0.25)).to_frame().T
            med = df.median().to_frame().T
            return iqr, med
    def Extract_stat_variables_from_big(self, df):
        arr = df.to_numpy()
        if self.Choix == 0:
            mean = arr.mean()
            std = arr.std()
            return mean, std
        if self.Choix == 1:
            Max = arr.max()
            Min = arr.min()
            return Max, Min
        if self.Choix == 2:
            iqr = np.quantile(arr, q = 0.75) - np.quantile(arr, q = 0.25)
            med = np.median(arr)
            return iqr, med
    def Normalise(self):
        Var_drop = [Var for Vars in self.Big_Var for Var in Vars]
        Arr = self.X_train.drop(Var_drop, axis = 1) 
        Stat1, Stat2 = self.Extract_stat_variables(Arr)
        del Arr
        
        for i, Var in enumerate(self.Big_Var):
            Cur = self.Extract_stat_variables_from_big(self.X_train[Var])
            Stat1[Var] = Cur[0]
            Stat2[Var] = Cur[1]
        if self.Choix == 0:
            self.meanX = Stat1.mean()
            self.stdX = Stat2.mean()
            self.meanY, self.stdY = self.Y_train.mean(), self.Y_train.std()
        elif self.Choix == 1:
            self.maxX, self.minX = Stat1.mean(), Stat2.mean()
            self.maxY, self.minY = self.Y_train.max(), self.Y_train.min()
        elif self.Choix == 2:
            self.iqrX = Stat1.mean()
            self.MedX = Stat2.mean()
            self.MedY = self.Y_train.median()
            self.iqrY = self.Y_train.quantile(0.75) - self.Y_train.quantile(0.25)

    def Normalize_Train_valid_ds(self):
        self.Normalise()
        if self.Choix == 0:
            self.X_train, self.X_valid = (np.array((self.X_train - self.meanX)/self.stdX), 
                                          np.array((self.X_valid - self.meanX)/self.stdX))

            self.Y_train, self.Y_valid = (np.array((self.Y_train - self.meanY)/self.stdY), 
                                          np.array((self.Y_valid - self.meanY)/self.stdY))
        
        elif self.Choix == 1:
            self.X_train, self.X_valid = (np.array((self.X_train - self.minX)/(self.maxX - self.minX)), 
                                          np.array((self.X_valid - self.minX)/(self.maxX - self.minX)))
            
            self.Y_train, self.Y_valid = (np.array((self.Y_train - self.minY)/(self.maxY - self.minY)), 
                                          np.array((self.Y_valid - self.minY)/(self.maxY - self.minY)))
        elif self.Choix == 2:
            self.X_train, self.X_valid = (np.array((self.X_train - self.MedX)/(self.iqrX)),
                                          np.array((self.X_valid - self.MedX)/(self.iqrX)))
            
            self.Y_train, self.Y_valid = (np.array((self.Y_train - self.MedY)/(self.iqrY)),
                                          np.array((self.Y_valid - self.MedY)/(self.iqrY)))
            
        self.Training_sample_size = len(self.Y_train)
        self.Js['Training_sample_size'] = self.Training_sample_size
    def Separate_X_Y(self):
        print('Separate X Y datasets')
        Y = self.li[self.Var_Y]
        #li.drop(columns=self.Var_Y, inplace = True)
        X = self.li[self.Var_X]
        del self.li
        gc.collect()
        if self.Similar_training == True:
            self.Select_dataset_for_similar_train(X, Y)
        else:
            self.X_train = X.sample(frac = 0.8, random_state = self.Random_seed)
            self.X_valid = X.drop(self.X_train.index)
            del X
            gc.collect()
            self.Y_train = Y.loc[self.X_train.index]
            self.Y_valid = Y.drop(self.X_train.index)
            del Y
            gc.collect()
            #return X_train, X_valid, Y_train, Y_valid
    
    def Select_dataset_for_similar_train(self, X, Y):
        ind_p = os.path.join(PWD, 'Auto_model/tmp/', '_'.join(self.Dataset_train))
        inds = glob.glob(ind_p + '/ind_*.csv')
        Inds = np.loadtxt(inds[0]).astype(int)
        self.X_valid = X.loc[X.index.isin(Inds)]
        self.Y_valid = Y.loc[Y.index.isin(Inds)]
        self.X_train = X.loc[~X.index.isin(Inds)]
        self.Y_train = Y.loc[~Y.index.isin(Inds)]
        #print(len(self.X_train), len(self.X_valid))
        #print(len(self.Y_train), len(self.Y_valid))
        self.Js['Similar_training'] = inds[0].replace(ind_p + '/ind_', '').replace('.csv', '')
        #return X_train, X_valid, Y_train, Y_valid
    def Save_stat_variables(self):
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
            self.minX.to_pickle(self.Path + 'MinX.pkl')
            np.savetxt(self.Path + 'MaxY.csv', np.array(self.maxY).reshape(1, ))
            np.savetxt(self.Path + 'MinY.csv', np.array(self.minY).reshape(1, ))
        elif self.Choix == 2:
            self.MedX.to_pickle(self.Path + 'MedX.pkl')
            self.iqrX.to_pickle(self.Path + 'iqrX.pkl')
            np.savetxt(self.Path + 'MedY.csv', np.array(self.MedY).reshape(1, ))
            np.savetxt(self.Path + 'iqrY.csv', np.array(self.iqrY).reshape(1, ))
            
    def DataProcessing(self):
        self.li = self.Fetch_dataset()
        self.Separate_X_Y()
        self.Normalize_Train_valid_ds()
        self.Save_stat_variables()

    def Neural_network_training(self):
        #Prepare all the data :
        #Fetching, Normalising, creating self.X_train ..... self.Y_valid
        self.DataProcessing()
        
        #Initialise NN structure and all the hyperparameter
        self.Init_NN_structure()
        self.Gather_callbacks()
        
        # Execute training
        Start = time.perf_counter()
        if self.Hybrid == True:
            # Not yet implemented
            Mod = self.Execute_hybrid_train()
        else:
            Mod = self.Execute_train()
        self.Js['Training_time'] = time.perf_counter() - Start
        print(f"Training done after {int(self.Js['Training_time'])} s")
        self.Model_save(Mod)
        
    def Init_NN_structure(self):
        '''
    This functions initialise the NN structure, neurons, activation function and dropout layers.
    
    Parameters
    ----------
    arg : all contained in self
        - Neur_seq : str : Is a string containing the NN neurons and layers structure 
            ex : "8_16" is a NN composed of two dense layers between the input and ouput layer. The first dense layer having 8 neurons and second 16 neurons.
        - activ_fct : Sets the activation function to be used by all dense layers of the NN(except the output). Any function can be selected from the keras library. Default here : self.activ_fct = 'swish'    
        
        - Drop = Default : Adds dropout layers between the dense layers with a constant self.Default_drop dropout rate(ex:0.1 : 10% of dropping connections)
        - Drop = Scaling : Adds dropout layers between the dense layers with a changing dropout rate chaning from 0.9(90%) at the first dropout layer to a minimum of 0.5(50%). Values can be change to observe changes in NN behaviour.
    
    Returns:
            self.model: NN model to be used for the training process.
    '''
        Orders = self.Neur_seq.split('_')
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(len(self.Var_X)))
        Drop_seq = []   
        for i, Order in enumerate(Orders):
            if int(Order)!=0:
                self.model.add(tf.keras.layers.Dense(int(Order), activation = self.activ_fct))
                if self.Drop != None or self.Hybrid == True:
                    if i < len(Orders) - 1 : 
                        if self.Drop == 'Default' or self.Hybrid == True:
                            self.model.add(tf.keras.layers.Dropout(self.Default_drop))
                            Drop_seq.append(self.Default_drop)
                        elif self.Drop == 'Scaling':
                            Dropout = max(0.5, 0.9 - i * 0.15)
                            self.model.add(tf.keras.layers.Dropout(Dropout))
                            Drop_seq.append(Dropout)
        self.model.add(tf.keras.layers.Dense(len(self.Var_Y)))
        #print(self.model.layers)
        if self.Drop == 'Scaling':
            optimizer = tf.keras.optimizers.Adam(lr=0.01)
            self.model.compile(optimizer=optimizer,
                     loss = 'mse',
                    metrics = ['mae', 'mse'])
        else:
            self.model.compile(optimizer='adam',
                     loss = 'mse',
                    metrics = ['mae', 'mse'])
        # Nice to have values in the config files.
        self.Js['Drop_seq'] = Drop_seq
        self.Js['Param'] = self.model.count_params()
    
    def Gather_callbacks(self):
    '''
    This functions initialise all of the required callbacks for the training process. All callbacks are then stored inside the self.Callback list.
    
    Special case : When using the hybrid training method(Training in 2 parts with and without dropout) the self.Callback will be a list of dimension 2 for the respective training parts as the lr schedulling method does not do well with dropout.
    
    Parameters
    ----------
    arg : all contained in self
        - Scaling_lr = True : Adds a learning scheduling method to the NN training using the self.Add_lr_schedulling() function
        - Pruning = True : Adds a pruning method to the NN training using the self.Add_pruning() function
        - TensorBoard_logs = True : Creates a tensorboard file for the NN training.
    
    '''
        saver = CustomSaver(path = self.Path, Epoch_max = self.Epoch, Fraction = self.Fraction_save)
        self.Callback.append(saver)
        
        if self.Scaling_lr == True:
            self.Add_lr_schedulling()
        if self.Pruning == True:
            self.Add_pruning()
        if self.TensorBoard_logs == True:
            logdir="Auto_model/logs/fit/" + str(self.Uniq_id)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            self.Callback.append(tensorboard_callback)
    def Add_lr_schedulling(self):
    '''
    This functions initialise all of the required callbacks for the training process. All callbacks are then stored inside the self.Callback list.

    Parameters
    ----------
    arg : all contained in self
        - Scaling_type = Linear : Adds a linear lr schedulling method meaning the lr gets reduced every (self.Frequence_scaling_change)th number of epoch.
        - Scaling_type == 'Plateau' : Adds a lr schedduling method using the ReduceLROnPlateau keras's callback with a lr reduced depending on the variation of the valiation loss with respect to previous epoch's end.

    '''
        if self.Scaling_lr == True:
            if self.Scaling_type == 'Linear':
                New_lr = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            elif self.Scaling_type == 'Plateau':
                New_lr = tf.keras.callbacks.ReduceLROnPlateau(Patience = self.LR_Patience, factor = 1 / self.LR_Factor, monitor='val_loss', min_lr = self.LR_min, verbose = 1, min_delta=self.min_delta)
            
            self.Callback.append(New_lr)

    
    def scheduler(self, epoch, lr):
        
        if not self.Hybrid:
            if (epoch+1) % self.Frequence_scaling_change == 0 and lr/self.Scaling_change >= self.LR_min:
                return lr / self.Scaling_change
            if lr/self.Scaling_change < self.LR_min:
                return self.LR_min
            else:
                return lr
        else:
            if (epoch+1) % self.Frequence_scaling_change == 0 and self.Epoch_init == self.Epoch_lim:
                return lr / self.Scaling_change
            else:
                return lr
            
    def Add_pruning(self):
        logdir = tempfile.mkdtemp()
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        #self.model = prune_low_magnitude(self.model, **pruning_params)
#        self.Training_sample_size Correspond au nombre de points present dans le dataset d'entrainement
        Frequency = math.ceil(self.Training_sample_size / self.batch_size)
        if self.Pruning_type == 'Constant':
            
            
            pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                self.target_sparsity, 1, end_step=-1, frequency=Frequency
        )}

        if self.Pruning_type == 'Polynomial':
            Total_step = Frequency * self.Epoch
            pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.initial_sparsity,
                                                               final_sparsity=self.target_sparsity,
                                                               begin_step=0,
                                                               end_step=Total_step)}

        self.model = prune_low_magnitude(self.model, **pruning_params)
        self.Callback.append(tfmot.sparsity.keras.UpdatePruningStep())
        self.Callback.append(tfmot.sparsity.keras.PruningSummaries(log_dir=logdir))
        print('Engaging pruning procedure!')
                

    def Execute_hybrid_train(self):
        pass
    
    def Execute_train(self):
        Mod = self.model.fit(self.X_train, self.Y_train,
                   callbacks=self.Callback,
                   epochs = self.Epoch,
                   batch_size = self.batch_size,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose)
        return Mod
    
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
            self.minX.to_pickle(self.Path + 'MinX.pkl')
            np.savetxt(self.Path + 'MaxY.csv', np.array(self.maxY).reshape(1, ))
            np.savetxt(self.Path + 'MinY.csv', np.array(self.minY).reshape(1, ))
        elif self.Choix == 2:
            self.MedX.to_pickle(self.Path + 'MedX.pkl')
            self.iqrX.to_pickle(self.Path + 'iqrX.pkl')
            np.savetxt(self.Path + 'MedY.csv', np.array(self.MedY).reshape(1, ))
            np.savetxt(self.Path + 'iqrY.csv', np.array(self.iqrY).reshape(1, ))
                                                   
    def Model_save(self, Mod):
        if self.Pruning == True:
            self.model = tfmot.sparsity.keras.strip_pruning(self.model)

        self.model.save(self.Path + 'model_{}.h5'.format(self.Epoch))
        hist = self.model.history
        #json.dump(hist, open(self.path + 'History_log.json'), 'w')
        if not self.Hybrid :
            with open(self.Path + 'TrainingHistory', 'wb') as file_p:
                pickle.dump(Mod.history, file_p)
        else:
            d = {}
            for k in self.histories[0].keys():
                d[k] = np.concatenate(list(d[k] for d in self.histories))
            with open(self.Path + 'TrainingHistory', 'wb') as file_p:
                pickle.dump(d, file_p)
        with open(self.Path + 'config.json', 'w') as f:
            json.dump(self.Js, f)
        
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self,Epoch_lim):
        self.Epoch_lim = Epoch_lim
    
    def on_epoch_begin(self, epoch, logs = None):
        for i,layer in enumerate(self.model.layers):
            if layer.name[:7] == 'dropout' and (epoch+1) == self.Epoch_lim:
                self.model.layers[i].rate = 0
                
                
class Sequencial_training():
    def __init__(self, Model):
        self.Model = Model
        self.Tot = []
    def Train_more_Models(self, Model_list = ['COM_NEMO-CNRS'], **kwargs):
        for Mod_name in Model_list:
            self.training(Oc_mod_type = Mod_name, **kwargs)
    def training(self, training_extent = 0, verbose = 1, Verify = 1,message = 1,
                 Standard_train = ['32_64_64_32'], Exact = 0, Similar_training = False, **kwargs):

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
        
        if Similar_training == True:
            self.Generate_training_index(**kwargs)
        Time_elap = float(0)
        for ind, Neur in enumerate(Neur_seqs):
            Model = self.Model(Neur_seq = Neur, verbose = verbose,Similar_training = Similar_training, **kwargs)
            print("Starting training for neurone : {}, {}/{} (Previous step : {:.3f} s)".format(Neur, ind, len(Neur_seqs), Time_elap))
            Start = time.perf_counter()
            Model.Neural_network_training()
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
        Path = PWD + '/Auto_model/tmp/'+ '_'.join(Template.Dataset_train) + '/'
        Make_dire(Path)
        #print(PWD)
        #print(Path, glob.glob(Path + '*'))
        if len(glob.glob(Path + '*.csv')) == 0 :
            df = Template.Fetch_data(Template.Dataset_train, Template.Oc_mod_type)
            index = df.sample(frac = 0.2).index
            np.savetxt(Path + 'ind_{}.csv'.format(int(time.time())), index.to_numpy())
        
        
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, path, Epoch_max, Epoch_init = 0, Fraction = None):
        self.path = path
        self.Epoch_max = Epoch_max
        self.Epoch_init = Epoch_init
        self.Fraction = Fraction
    def on_epoch_end(self, epoch, logs={}):
        if self.Fraction == None or (self.Fraction != None and (epoch + 1) % self.Fraction == 0):
            if epoch + 1 != self.Epoch_max:
                self.model.save(self.path + "model_{}.h5".format(epoch + 1 + self.Epoch_init))
            

                        
    
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

def Convert_big_to_var(Var, data):
    if data.get('Big_Var') is not None and Var != None:
        if len(data.get('Big_Var')) != 0:
            #print(data.get('Big_Var'), type(data.get('Big_Var')), print(data.get('Uniq_id')))
            if 'Big_T' in Var:
                Var.remove('Big_T')
                if 'T_0' in data.get('Var_X'):
                    Var.extend(data['Big_Var'][0])
            if 'Big_S' in Var:
                Var.remove('Big_S')
                if 'S_0' in data.get('Var_X'):
                    #print(data.get('Uniq_id'))
                    Var.extend(data['Big_Var'][min(len(data['Big_Var'])-1, 1)])
                
                #print(f'After transf {Var}')
    return Var
    
    
    return 
def Get_model_path_json(Var = None, Epoch = None, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = True, 
            Extra_n = None, Choix = None, Neur = None, Batch_size = None, index = 0, Cutting = None, Drop = None, 
            Method_data = None, Scaling_lr = None , Pick_Best = False, Hybrid = None, return_all = False, Activation_fct = None, Uniq_id = None, Specific_epoch = None, Scaling_type = None, Pruning = None):
    if type(Ocean) != list:
        Ocean = [Ocean]
    path = os.path.join(PWD, 'Auto_model', Type_trained, '_'.join(Ocean))
    if Exact == True and Epoch != None:
        Model_paths = glob.glob(path + '/Ep_{}*'.format(Epoch))
    else:
        Model_paths = glob.glob(path + '/Ep_*')
    #Mod = list(Model_paths)
    for f in list(Model_paths):
        data = Get_model_attributes(f)
        if data != None:

            if Uniq_id != None:
                if Uniq_id != data.get('Uniq_id'):
                    Model_paths.remove(f)
                    continue
                    
            if ((Choix != None and data['Choix'] != int(Choix)) or (Var != None and sorted(data['Var_X']) != sorted(Convert_big_to_var(list(Var), data)))):
                Model_paths.remove(f)
                #print(f"{f} removed because either Choix or Var")
                continue
            if (Neur != None and data['Neur_seq'] != Neur) or (Batch_size != None and Batch_size != data['batch_size']):
                Model_paths.remove(f)
                #print(f"{f} removed because either Neur or Batch")
                continue
            if (Extra_n != None and data['Extra_n'] != Extra_n) :
                Model_paths.remove(f)
                #print(f"{f} removed because Extra_n")
                continue
            if Epoch != None:
                if (Exact == True and data['Epoch'] != Epoch) or (Exact == False and data['Epoch'] < Epoch):
                    Model_paths.remove(f)
                    #print(f"{f} removed because Epoch")
                    continue
            if Cutting != None:
                if (data.get('Cutting') is None and Cutting != '') or (data.get('Cutting') is not None and data['Cutting'] != Cutting) or (Cutting == '' and (data.get('Cutting') is not None and data['Cutting'] != Cutting)):
                    Model_paths.remove(f)
                    #print(f"{f} removed because either Cutting")
                    continue
            if Drop != None:
                if (data.get('Cutting') is None and Drop != '') or (data.get('Drop') is not None and data['Drop'] != Drop):
                    Model_paths.remove(f)
                    #print(f"{f} removed because either Drop")
                    continue
            if Method_data != None:
                if (data.get('Method_data') != Method_data and Method_data != '') or (Method_data == '' and data.get('Method_data') is not None):
                    Model_paths.remove(f)
                    continue
            if Scaling_lr != None:
                #print(data.get('Scaling_lr'), Scaling_lr, f)
                if data.get('Scaling_lr') is None and Scaling_lr == True or (data.get('Scaling_lr') is not None and Scaling_lr != data.get('Scaling_lr')):
                    Model_paths.remove(f)
                    #print(f"{f} removed because Scaling")
                    continue
            if Scaling_type != None:
                if data.get('Scaling_type') is None and data.get('Scaling_lr') == True or (data.get('Scaling_type') is not None and Scaling_type != data.get('Scaling_type')):
                    Model_paths.remove(f)
                    #print(f"{f} removed because Scaling type")
                    continue
                
                
            if Hybrid != None:
                if data.get('Hybrid') is None and Hybrid == True or data.get('Hybrid') is not None and Hybrid != data.get('Hybrid'):
                    Model_paths.remove(f)
                    continue
            if Activation_fct != None:
                if data.get('activ_fct') != Activation_fct:
                    Model_paths.remove(f)
                    continue
            if Pruning != None:
                if data.get('Pruning') is None and Pruning == True or data.get('Pruning') is not None and Pruning != data.get('Pruning'):
                    Model_paths.remove(f)
                    continue
        else:
            Model_paths.remove(f)
    #print(f"Validated paths : {Model_paths}")
    if return_all:
        return Model_paths
        
    if index != None:
        if index >= len(Model_paths):
            index = len(Model_paths) - 1
        Model_paths = [Model_paths[index]]
    if Model_paths != []:
        data = Get_model_attributes(Model_paths[0])
        #print(Var)
        #print(data['Var_X'])
        if Pick_Best == True:
            hist = pd.read_pickle(Model_paths[0] + '/TrainingHistory')
            Best = np.argmin(hist['val_mse'])
            Model_paths = [f"{Model_paths[0]}/model_{Best + 1}.h5"]
        elif Specific_epoch != None :
            Model_paths = [f"{Model_paths[0]}/model_{Specific_epoch}.h5"]
    return Model_paths

def Get_model_path_json_exp(Var = None, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = 0, 
            Extra_n = None, Choix = None, Neur = None, Batch_size = None, index = None):
    if type(Ocean) != list:
        Ocean = [Ocean]
    path = os.path.join(PWD, 'Auto_model', Type_trained, '_'.join(Ocean))
    if Exact == 1:
        Model_paths = glob.glob(path + '/Ep_{}*'.format(Epoch))
    else:
        Model_paths = glob.glob(path + '/Ep_*')
    #Mod = list(Model_paths)
    for f in list(Model_paths):
        data = Get_model_attributes(f)
        if data != None:
            if ((Choix != None and data['Choix'] != int(Choix)) or (Var != None and sorted(data['Var_X']) != sorted(Var))):
                Model_paths.remove(f)
                #print(f"{f} removed because either Choix or Var")
                continue
    #print(f"Validated paths : {Model_paths}")
    if index != None:
        if index >= len(Model_paths):
            index = len(Model_paths) - 1
        Model_paths = [Model_paths[index]]
    return Model_paths

def Get_model_attributes(model_p):
    if os.path.isfile(model_p + '/config.json'):
        with open(model_p + '/config.json') as json_file:
            data = json.load(json_file)
        return data
    else:
        return None
