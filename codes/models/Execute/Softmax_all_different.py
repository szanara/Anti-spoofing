#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import*


# In[2]:


import numpy as np
import pandas as pd 

# In[3]:


import os


# In[11]:



# In[9]:


path_develop = '/home/cz/mds3/visao/Dataset/DEVELOP/'
path_evaluation = '/home/cz/mds3/visao/Dataset/TEST/'
path_train =  '/home/cz/mds3/visao/Dataset/TRAIN/'

path_pca_develop = '/home/cz/mds3/visao/pca/different/'
path_pca_evaluation = '/home/cz/mds3/visao/pca/different/'
path_pca_train =  '/home/cz/mds3/visao/pca/different/'

# In[12]:


path2save = '/home/cz/mds3/visao/results'


# In[2]:


path2save_im = '/home/cz/mds3/visao/figures/TrainTest_same/SOFTMAX/different'


# In[1]:

not_pca = ['CoALBP','RI-CoALBP']
descriptors_PCA = ['BSIF', 'LPQ', 'RI-LBP', 'UniLBP' ]


# In[10]:


develops = os.listdir(path_develop)
evaluations = os.listdir(path_evaluation)
trains = os.listdir(path_train)

results = []

for descriptor in descriptors_PCA:
    print(descriptor)
    
    image_save = os.path.join(path2save_im, descriptor)
    os.makedirs(image_save, exist_ok=True)
    X_train = np.load(path_pca_train+f'{descriptor}/train_pca_37.npy', allow_pickle=True)
    y_train =np.load(path_train+f'{descriptor}/y_train.npy', allow_pickle=True)
    X_develop = np.load(path_pca_develop+f'{descriptor}/dev_pca_37.npy', allow_pickle=True)
    y_develop =np.load(path_develop+f'{descriptor}/y_train.npy', allow_pickle=True)
    X_test = np.load(path_pca_evaluation+f'{descriptor}/test_pca_37.npy', allow_pickle=True)
    y_test =np.load(path_evaluation+f'{descriptor}/y_train.npy', allow_pickle=True)

    trainer = ModelTrainer(X_train = X_train,
                            y_train = y_train,
                            X_dev =X_develop,
                            y_dev =y_develop, 
                            X_test = X_test,
                            y_test =y_test)
    FAR_test, FRR_test, HTER,EER, training_time =trainer.training_model(epochs=100,
                            batch_size=5, 
                            input_shape=X_train.shape, 
                         plot_roc=[True,image_save],
                         plot_Testconfusion=[True,image_save])
    
    results.append([descriptor, FAR_test, FRR_test, HTER,EER, training_time])

for descriptor in not_pca:
    print(descriptor)
    
    image_save = os.path.join(path2save_im, descriptor)
    os.makedirs(image_save, exist_ok=True)
    X_train = np.load(path_train+f'/{descriptor}/X_train.npy', allow_pickle=True)
    y_train =np.load(path_train+f'/{descriptor}/y_train.npy', allow_pickle=True)
    X_develop = np.load(path_develop+f'{descriptor}/X_train.npy', allow_pickle=True)
    y_develop =np.load(path_develop+f'{descriptor}/y_train.npy', allow_pickle=True)
    X_test = np.load(path_evaluation+f'{descriptor}/X_train.npy', allow_pickle=True)
    y_test =np.load(path_evaluation+f'{descriptor}/y_train.npy', allow_pickle=True)

    trainer = ModelTrainer(X_train = X_train,
                            y_train = y_train,
                            X_dev =X_develop,
                            y_dev =y_develop, 
                            X_test = X_test,
                            y_test =y_test)
    FAR_test, FRR_test, HTER,EER, training_time =trainer.training_model(epochs=100,
                            batch_size=5, 
                            input_shape=X_train.shape, 
                         plot_roc=[True,image_save],
                         plot_Testconfusion=[True,image_save])
    
    results.append([descriptor, FAR_test, FRR_test, HTER,EER, training_time])


# Convertendo a lista para um DataFrame do pandas
results_df = pd.DataFrame(results, columns=['descriptor_train', 'FAR_test', 'FRR_test', 'HTER', 'EER', 'training_time'])

# Salvando o DataFrame como um arquivo CSV
results_df.to_csv(path2save+'/results_TrainTest_different_SOFTMAX.csv', index=False)

