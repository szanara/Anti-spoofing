import numpy as np
import os 
from model import *

path = '/home/cz/mds3/visao/Dataset/'
path2save='/home/cz/mds3/visao/pca'


path_develop = '/home/cz/mds3/visao/Dataset/DEVELOP/'
path_evaluation = '/home/cz/mds3/visao/Dataset/EVALUATION/'
path_train =  '/home/cz/mds3/visao/Dataset/TRAIN/'

develops = os.listdir(path_develop)
evaluations = os.listdir(path_evaluation)
trains = os.listdir(path_train)

path_test = '/home/cz/mds3/visao/Dataset/TEST/'
tests = os.listdir(path_test)
path2save2='/home/cz/mds3/visao/pca/different/'

for descriptor_train, descriptor_dev, descriptor_eval in zip(trains, develops, tests):
    print(descriptor_train, descriptor_dev, descriptor_eval)
    saving = os.path.join(path2save, descriptor_train)
    os.makedirs(saving,exist_ok=True)
    if descriptor_train not in ['CoALBP','RI-CoALBP' ]:
        X_train = np.load(path_train+f'/{descriptor_train}/X_train.npy', allow_pickle=True)
        y_train =np.load(path_train+f'/{descriptor_train}/y_train.npy', allow_pickle=True)
        X_dev = np.load(path_develop+f'{descriptor_dev}/X_train.npy', allow_pickle=True)
        y_dev =np.load(path_develop+f'{descriptor_dev}/y_train.npy', allow_pickle=True)
        X_test = np.load(path_test+f'{descriptor_eval}/X_train.npy', allow_pickle=True)
        y_test =np.load(path_test+f'{descriptor_eval}/y_train.npy', allow_pickle=True)

        n_components = min(len(X_train), len(X_test), len(X_dev))

        pca = PCA_features(X_train =X_train,
                        X_dev = X_dev,
                        X_test =  X_test,
                        n_components = n_components,
                        path2save=saving
                        )

        X_train, X_dev, X_test = pca.obtaining()