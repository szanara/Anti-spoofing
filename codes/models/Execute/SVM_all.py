#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import*


# In[2]:


import numpy as np
import pandas as pd

# In[3]:


import os




path_develop = '/home/cz/mds3/visao/Dataset/DEVELOP/'
path_evaluation = '/home/cz/mds3/visao/Dataset/EVALUATION/'
path_train =  '/home/cz/mds3/visao/Dataset/TRAIN/'





path2save = '/home/cz/mds3/visao/results'




path2save_im = '/home/cz/mds3/visao/figures/TrainTest_same'




develops = os.listdir(path_develop)
evaluations = os.listdir(path_evaluation)
trains = os.listdir(path_train)

results = []

for descriptor_train, descriptor_dev, descriptor_eval in zip(trains, develops, evaluations):
    print(descriptor_train, descriptor_dev, descriptor_eval)
    
    image_save = os.path.join(path2save_im, descriptor_train)
    os.makedirs(image_save,exist_ok=True)
    
    X_train = np.load(path_train+f'/{descriptor_train}/X_train.npy', allow_pickle=True)
    y_train =np.load(path_train+f'/{descriptor_train}/y_train.npy', allow_pickle=True)
    X_develop = np.load(path_develop+f'{descriptor_dev}/X_train.npy', allow_pickle=True)
    y_develop =np.load(path_develop+f'{descriptor_dev}/y_train.npy', allow_pickle=True)
    X_test = np.load(path_evaluation+f'{descriptor_eval}/X_train.npy', allow_pickle=True)
    y_test =np.load(path_evaluation+f'{descriptor_eval}/y_train.npy', allow_pickle=True)
    
    
    svm = SvmModel(X_train = X_train,
               X_test = X_test,
               y_train= y_train, 
               y_test = y_test,
               X_dev = X_develop,
               y_dev = y_develop)
    FAR_test, FRR_test, HTER, training_time = svm.SVM(plot_roc=[True,image_save],
                                                      plot_Testconfusion=[True,image_save])
    
    results.append([descriptor_train, descriptor_dev, descriptor_eval, FAR_test, FRR_test, HTER, training_time])
# Convertendo a lista para um DataFrame do pandas
results_df = pd.DataFrame(results, columns=['descriptor_train', 'descriptor_dev', 'descriptor_eval', 'FAR_test', 'FRR_test', 'HTER', 'training_time'])

# Salvando o DataFrame como um arquivo CSV
results_df.to_csv(path2save+'/results_TrainTest_same.csv', index=False)

