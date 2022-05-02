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

PWD = os.getcwd()
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

class model_NN():
    def __init__(self, Epoch = 2, Neur_seq = '32_64_64_32', Dataset_train = ['Ocean1'], Oc_mod_type = 'COM_NEMO-CNRS', Var_X = ['x', 'y', 'temperatureYZ', 'salinityYZ', 'iceDraft'], Var_Y = 'meltRate', activ_fct = 'swish', Norm_Choix = 0, verbose = 1, batch_size = 32, Extra_n = '', Better_cutting = False, Drop = None, Default_drop = 0.5, Method_data = None, Method_extent = [0, 40], Scaling_lr = False, Scaling_change = 2, Frequence_scaling_change = 8, Multi_thread = False, Workers = 1, TensorBoard_logs = False, Hybrid = False, Fraction = 1, Fraction_save = None,
Epoch_lim = 15, Scaling_type = 'Linear', LR_Patience = 2, LR_min = 0.0000016, LR_Factor = 2):
        self.Neur_seq = Neur_seq
        self.Epoch = Epoch
        self.Var_X = list(Var_X)
        self.Var_Y = Var_Y
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
        self.Default_drop = Default_drop
        self.Method_data = Method_data
        self.Method_extent = Method_extent
        self.Scaling_lr = Scaling_lr
        self.TensorBoard_logs = TensorBoard_logs
        if self.Scaling_lr == True:
            self.Frequence_scaling_change = Frequence_scaling_change
            self.Scaling_change = Scaling_change
            self.Scaling_type = Scaling_type
            if self.Scaling_type == 'Plateau':
                self.LR_Patience = LR_Patience
                self.LR_min = LR_min
                self.LR_Factor = LR_Factor
        self.Multi_thread = Multi_thread
        self.Workers = Workers
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
        self.Js = dict(self.__dict__)
        
    def Init_mod(self, Shape):
        Orders = self.Neur_seq.split('_')
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(Shape))
        Drop_seq = []
        #self.model.add(tf.keras.layers.Dropout(self.Default_drop))
        if self.Drop == 'Scaling':
            optimizer = tf.keras.optimizers.Adam(lr=0.01) #x10
            
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
        self.Js['Drop_seq'] = Drop_seq
        self.model.add(tf.keras.layers.Dense(1))
        if self.Drop == 'Scaling':
            self.model.compile(optimizer=optimizer,
                     loss = 'mse',
                    metrics = ['mae', 'mse'])
        else:
            self.model.compile(optimizer='adam',
                     loss = 'mse',
                    metrics = ['mae', 'mse'])
        self.Js['Param'] = self.model.count_params()
        
    def Fetch_data(self, Datasets, Oc_mod_type):
        li = []
        for ind, Data in enumerate(Datasets):

            if self.Method_data == None:
                Current = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data))
                if self.Cutting == 'Same_t':
                    Current = Current.loc[Current.date <= 250]
                elif '%' in self.Cutting:
                    percent = int(re.findall('(\d+)%', self.Cutting)[0])/100
                    print(self.Cutting, percent)
                    tmx = int(max(np.array(Current.date)))
                    Current = Current.loc[Current.date <= int(percent * tmx) | Current.date >= int((1 - percent) * tmx)]
                li.append(Current)
                del Current
            else:
                print(f"Getting dataset : {Data}")
                print(f"Dataset used : {Getpath_dataset(Data, Oc_mod_type, self.Method_data)}")
                if (self.Method_data != 4 and self.Method_data != 2):
                    if ind == 0:
                        df = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data))
                    else:
                        df = pd.concat([df, pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data))], ignore_index= True)
                else:
                    if ind == 0:
                        X = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data), usecols = self.Var_X)
                        Y = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data), usecols = [self.Var_Y])
                        if '%' in self.Cutting:
                            percent = int(re.findall('(\d+)%', self.Cutting)[0])/100
                            print(self.Cutting, percent)
                            Current = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data))
                            tmx = int(max(np.array(Current.date)))
                            Current = Current.loc[(Current.date <= int(percent * tmx)) | (Current.date >= int((1 - percent) * tmx))]
                            X = Current[self.Var_X]
                            Y = Current[self.Var_Y]
                        
                    else:
                        if self.Cutting == False:
                            X = pd.concat([X, pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data), usecols = self.Var_X)], ignore_index= True)
                            Y = pd.concat([Y, pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data), usecols = [self.Var_Y])], ignore_index= True)
                        else:
                            Current = pd.read_csv(Getpath_dataset(Data, Oc_mod_type, self.Method_data))
                            if '%' in self.Cutting:
                                percent = int(re.findall('(\d+)%', self.Cutting)[0])/100
                                print(self.Cutting, percent)
                                tmx = int(max(np.array(Current.date)))
                                Current = Current.loc[(Current.date <= int(percent * tmx)) | (Current.date >= int((1 - percent) * tmx))]
                            X = pd.concat([X, Current[self.Var_X]], ignore_index= True)
                            Y = pd.concat([Y, Current[self.Var_Y]], ignore_index= True)
                            
                for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),key= lambda x: -x[1])[:10]:
                    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                print(f"Finished dataset : {Data}")
            
        if self.Method_data == 4 or self.Method_data == 2:
            if self.Fraction != 1:
                X = X.sample(frac = self.Fraction)
                Y = Y.loc[X.index]
            return X, Y
        
        if self.Method_data == None:
            df = pd.concat(li, ignore_index= True)
        del li
        return df

    def Prepare_data(self, Indexs):
        if self.Method_data == 4 or self.Method_data == 2:
            X, Y = self.Fetch_data(self.Dataset_train, self.Oc_mod_type)
            df = []
        else:
            df = self.Fetch_data(self.Dataset_train, self.Oc_mod_type)
            X = df[self.Var_X]
            Y = df[self.Var_Y]
        del df
        print('Check index')
        if Indexs == None:
            X_train = X.sample(frac = 0.8)
            X_valid = X.drop(X_train.index)
            Y_train = Y.loc[X_train.index]
            Y_valid = Y.drop(X_train.index)
            self.Js['Similar_training'] = 0
        else:
            ind_p = os.path.join(PWD, 'Auto_model/tmp/', '_'.join(self.Dataset_train))
            if self.Cutting != False:
                inds = glob.glob(ind_p + '/Cut*.csv')
            else:
                inds = glob.glob(ind_p + '/ind_*.csv')
            Inds = np.loadtxt(inds[0]).astype(int)
            X_valid = X.loc[Inds]
            Y_valid = Y.loc[Inds]
            X_train = X.drop(Inds)
            Y_train = Y.drop(Inds)
            self.Js['Similar_training'] = inds[0].replace(ind_p + '/ind_', '').replace('.csv', '')
        del X, Y
        print('Begin Norma')
        if self.Choix == 0:
            if self.Method_data != 4 and self.Method_data != 2 and self.Method_data != 5:
                self.meanX, self.stdX = X_train.mean(), X_train.std() 
            else:
                #Arr = X_train.drop(self.Big_Var, axis = 1)
                Means, Stds = [], []
                Var_drop = [Var for Vars in self.Big_Var for Var in Vars]
                Arr = X_train.drop(Var_drop, axis = 1) 
                M = Arr.mean().to_frame().T
                Std = Arr.std().to_frame().T
                for i, Var in enumerate(self.Big_Var):
                    M[Var] = X_train[Var].to_numpy().mean()
                    Std[Var] = X_train[Var].to_numpy().std()
                self.meanX = M.mean()
                self.stdX = Std.mean()
                del Arr, M, Std
            
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
        elif self.Choix == 2:
            self.MedX, self.MedY = X_train.median(), Y_train.median()
            self.iqrX = X_train.quantile(0.75) - X_train.quantile(0.25)
            self.iqrY = Y_train.quantile(0.75) - Y_train.quantile(0.25)
            self.X_train, self.X_valid = (np.array((X_train - self.MedX)/(self.iqrX)),
                                          np.array((X_valid - self.MedX)/(self.iqrX)))
            self.Y_train, self.Y_valid = (np.array((Y_train - self.MedY)/(self.iqrY)),
                                          np.array((Y_valid - self.MedY)/(self.iqrY)))
    def train(self, Indexs = None):
        Shape = len(self.Var_X)
        self.Init_mod(Shape)
        self.Prepare_data(Indexs)
        self.Data_save()
        saver = CustomSaver(path = self.Path, Epoch_max = self.Epoch, Fraction = self.Fraction_save)
        Callbacks = []
        Callbacks.append(saver)
        if self.Scaling_lr == True:
            if self.Scaling_type == 'Linear':
                New_lr = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            elif self.Scaling_type == 'Plateau':
                New_lr = tf.keras.callbacks.ReduceLROnPlateau(Patience = self.LR_Patience, factor = 1 / self.LR_Factor, monitor='val_loss', min_lr = self.LR_min, verbose = 1, min_delta=0.007)
            Callbacks.append(New_lr)
        if self.TensorBoard_logs == True:
            logdir="Auto_model/logs/fit/" + str(self.Uniq_id)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            Callbacks.append(tensorboard_callback)
            
        Start = time.perf_counter()
        if not self.Hybrid:
            Mod = self.model.fit(self.X_train, self.Y_train,
                   callbacks=Callbacks,
                   epochs = self.Epoch,
                   batch_size = self.batch_size,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose,
                   use_multiprocessing = self.Multi_thread,
                   workers = self.Workers)
            
        else:
            Epochs = [self.Epoch_lim, self.Epoch-self.Epoch_lim]
            self.histories = []
            for i,Ep in enumerate(Epochs):
                if i == 0:
                    self.Epoch_init = 0
                else:
                    self.Epoch_init = Epochs[i-1]
                Callbacks[0] = CustomSaver(path = self.Path, Epoch_max = self.Epoch, Epoch_init = self.Epoch_init, Fraction = self.Fraction_save)
                Mod = self.model.fit(self.X_train, self.Y_train,
                   callbacks=Callbacks,
                   epochs = Ep,
                   batch_size = self.batch_size,
                   validation_data = (self.X_valid, self.Y_valid),
                   verbose = self.verbose,
                   use_multiprocessing = self.Multi_thread,
                   workers = self.Workers)
                if i == 0:
                    for i,layer in enumerate(self.model.layers):
                        if layer.name[:7] == 'dropout':
                            self.model.layers[i].rate = 0
                    clone = tf.keras.models.clone_model(self.model)
                    clone.set_weights(self.model.get_weights())
                    self.model = clone
                    self.model.compile(optimizer='adam',
                     loss = 'mse',
                        metrics = ['mae', 'mse'])
                self.histories.append(Mod.history)
                
        self.Js['Training_time'] = time.perf_counter() - Start
        del self.X_train, self.Y_train, self.X_valid, self.Y_valid
        self.Model_save(Mod)
        
    def scheduler(self, epoch, lr):
        if not self.Hybrid:
            if (epoch+1) % self.Frequence_scaling_change == 0:
                return lr / self.Scaling_change
            else:
                return lr
        else:
            if (epoch+1) % self.Frequence_scaling_change == 0 and self.Epoch_init == self.Epoch_lim:
                return lr / self.Scaling_change
            else:
                return lr
        
        
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
        Path = PWD + '/Auto_model/tmp/'+ '_'.join(Template.Dataset_train) + '/'
        Make_dire(Path)
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
                Var.extend(data['Big_Var'][0])
            if 'Big_S' in Var:
                Var.remove('Big_S')
                Var.extend(data['Big_Var'][1])
                #print(f'After transf {Var}')
    return Var
    
    
    return 
def Get_model_path_json(Var = None, Epoch = 4, Ocean = 'Ocean1', Type_trained = 'COM_NEMO-CNRS', Exact = 0, 
            Extra_n = None, Choix = None, Neur = None, Batch_size = None, index = None, Cutting = None, Drop = None, 
            Method_data = None, Scaling_lr = None , Pick_Best = False, Hybrid = None, return_all = False):
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

            if ((Choix != None and data['Choix'] != int(Choix)) or (Var != None and sorted(data['Var_X']) != sorted(Convert_big_to_var(list(Var), data)))):
                Model_paths.remove(f)
                #print(f"{f} removed because either Choix or Var")
                continue
            if (Neur != None and data['Neur_seq'] != Neur) or (Batch_size != None and Batch_size != data['batch_size']):
                Model_paths.remove(f)
                #print(f"{f} removed because either Neur or Batch")
                continue
            if (Exact != 1 and data['Epoch'] < Epoch) or (Extra_n != None and data['Extra_n'] != Extra_n):
                Model_paths.remove(f)
                #print(f"{f} removed because either Epoch or Extra_n")
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
            if Hybrid != None:
                if data.get('Hybrid') is None and Hybrid == True or data.get('Hybrid') is not None and Hybrid != data.get('Hybrid'):
                    Model_paths.remove(f)
        else:
            Model_paths.remove(f)
    #print(f"Validated paths : {Model_paths}")
    if return_all:
        return Model_paths
        
    if index != None:
        if index >= len(Model_paths):
            index = len(Model_paths) - 1
        Model_paths = [Model_paths[index]]
    data = Get_model_attributes(Model_paths[0])
    #print(Var)
    #print(data['Var_X'])
    if Pick_Best == True:
        hist = pd.read_pickle(Model_paths[0] + '/TrainingHistory')
        Best = np.argmin(hist['val_mse'])
        Model_paths = [f"{Model_paths[0]}/model_{Best + 1}.h5"]
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
