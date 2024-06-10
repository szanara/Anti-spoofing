
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model_proposal import*
import pandas as  pd

# In[2]:


import numpy as np


# In[3]:


import os


# In[11]:


import time


# In[9]:


path_develop1 = '/home/cz/mds3/visao/Dataset/DEVELOP/RI-LBP/'
path_evaluation1 = '/home/cz/mds3/visao/Dataset/TEST/RI-LBP/'
path_train1 =  '/home/cz/mds3/visao/Dataset/TRAIN/RI-LBP/'

path_develop2 = '/home/cz/mds3/visao/Dataset/DEVELOP/SIFT/'
path_evaluation2 = '/home/cz/mds3/visao/Dataset/TEST/SIFT/'
path_train2 =  '/home/cz/mds3/visao/Dataset/TRAIN/SIFT/'



# In[12]:


path2save = '/home/cz/mds3/visao/results/'


# In[2]:


path2save_im = '/home/cz/mds3/visao/figures/TrainTest_same/SVM/proposal'


# In[10]:



results = []



image_save = os.path.join(path2save_im, 'PROPOSAL')
os.makedirs(image_save,exist_ok=True)

X_train_ri = np.load(path_train1+f'/X_train.npy', allow_pickle=True)
y_train_ri =np.load(path_train1+f'/y_train.npy', allow_pickle=True)
X_develop_ri = np.load(path_develop1+f'/X_train.npy', allow_pickle=True)
y_develop_ri =np.load(path_develop1+f'/y_train.npy', allow_pickle=True)
X_test_ri = np.load(path_develop1+f'/X_train.npy', allow_pickle=True)
y_test_ri =np.load(path_develop1+f'/y_train.npy', allow_pickle=True)
X_train_sift = np.load(path_train2+f'/X_train.npy', allow_pickle=True)
y_train_sift =np.load(path_train2+f'/y_train.npy', allow_pickle=True)
X_develop_sift = np.load(path_develop2+f'/X_train.npy', allow_pickle=True)
y_develop_sift =np.load(path_develop2+f'/y_train.npy', allow_pickle=True)
X_test_sift = np.load(path_develop2+f'/X_train.npy', allow_pickle=True)
y_test_sift =np.load(path_develop2+f'/y_train.npy', allow_pickle=True)

print('xTEST',X_test_ri.shape, X_test_sift.shape)
svm = SvmModelProposal(X_train1 = X_train_ri,
            X_test1 = X_test_ri, 
            y_train1= y_train_ri,  
            y_test1 = y_test_ri,  
            X_dev1 = X_develop_ri,
            y_dev1 = y_develop_ri,
            X_train2 = X_train_sift,
            X_test2 = X_test_sift, 
            y_train2 = y_train_sift, 
             y_test2 = y_test_sift,  
              X_dev2 = X_develop_sift,
              y_dev2 = y_develop_sift )
FAR_test, FRR_test, HTER, EER, training_time = svm.SVM()

results.append(['RI-LBP+SIFT', FAR_test, FRR_test, HTER, EER, training_time])
# Convertendo a lista para um DataFrame do pandas
results_df = pd.DataFrame(results, columns=['descriptor', 'FAR_test', 'FRR_test', 'HTER', 'EER','training_time'])

# Salvando o DataFrame como um arquivo CSV
results_df.to_csv(path2save+'/results_TrainTest_different_SVM_proposal.csv', index=False)

