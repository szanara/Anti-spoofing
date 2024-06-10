import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import time
from sklearn.feature_selection import SelectKBest, f_classif

class SvmModelProposal:
    def __init__(self, X_train1, X_test1, y_train1, y_test1,X_dev1, y_dev1, X_train2, X_test2, y_train2, y_test2,X_dev2, y_dev2 ) -> None:
        self.X_train_ri = X_train1
        self.X_dev_ri = X_dev1
        self.y_train_ri = y_train1
        self.y_dev_ri = y_dev1
        self.X_train_sift = X_train2
        self.X_dev_sift = X_dev2
        self.y_train_sift = y_train2
        self.y_dev_sift = y_dev2
        self.X_test_ri = X_test1
        self.X_test_sift = X_test2
        self.y_test_ri= y_test1
        self.y_test_sift= y_test2

    
    def SVM(self, ):
        
        svm1 = SVC(kernel='linear', probability=True, verbose=True)
        start_time = time.time()
        svm1.fit(self.X_train_ri, self.y_train_ri)
        end_time = time.time()
        training_time1 = end_time - start_time

        svm2 = SVC(kernel='linear', probability=True, verbose=True)
        start_time = time.time()
        svm2.fit(self.X_train_sift, self.y_train_sift)
        end_time = time.time()
        training_time2 = end_time - start_time

        training_time = training_time1+training_time2

        print(f"Tempo de treinamento: {training_time} segundos")

        # Predict probabilities in the dev set
        y_dev_scores_ri = svm1.predict_proba(self.X_dev_ri)[:, 1]

        # Calculating EER
        fpr, tpr, thresholds = roc_curve(self.y_dev_ri, y_dev_scores_ri)
        fnr = 1 - tpr
        eer_threshold_ri = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]


        # Predict probabilities in the dev set
        y_dev_scores_sift = svm2.predict_proba(self.X_dev_sift)[:, 1]

        # Calculating EER
        fpr, tpr, thresholds = roc_curve(self.y_dev_sift, y_dev_scores_sift)
        fnr = 1 - tpr
        eer_threshold_sift = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

        # Selecionar as 3 melhores features usando ANOVA F-test
        selector = SelectKBest(f_classif, k=3)
        X_train_selected = selector.fit_transform(self.X_train_ri, self.y_train_ri)
        X_test_selected = selector.transform(self.X_test_ri)

        # Treinar o modelo SVM com as features selecionadas
        svm_retrain = SVC(kernel='linear', probability=True)
        svm_retrain.fit(X_train_selected, self.y_train_ri)
        

        # Aplying  EER treshold in develop set
        y_dev_pred_ri = (y_dev_scores_ri >= eer_threshold_ri).astype(int)
        y_dev_pred_sift = (y_dev_scores_sift >= eer_threshold_sift).astype(int)

        # Calculating  FAR and FRR  in the dev set using the EER treshold
        tn, fp, fn, tp = confusion_matrix(self.y_dev_ri, y_dev_pred_ri).ravel()
        FRR_ri = fn / (fn + tp)  # False Rejection Rate
        FAR_ri = fp / (fp + tn)  # False Acceptance Rate

        tn, fp, fn, tp = confusion_matrix(self.y_dev_sift, y_dev_pred_sift).ravel()
        FRR_sift = fn / (fn + tp)  # False Rejection Rate
        FAR_sift = fp / (fp + tn)  # False Acceptance Rate

        FAR = (FAR_sift +FAR_ri)/2
        FRR = (FRR_sift + FRR_ri)/2
        eer_threshold = (eer_threshold_sift + eer_threshold_ri)/2
        print(f"Development Set - EER Threshold: {eer_threshold}, FAR: {FAR}, FRR: {FRR}")
        #print(self.X_test1.shape)
        #print(self.X_test1)

        y_test_scores_ri1 = svm1.predict_proba(self.X_test_ri)[:, 1]
        y_test_scores_ri2 = svm_retrain.predict_proba(self.X_test_sift)[:, 1]
        y_test_scores_sift1 = svm2.predict_proba(self.X_test_sift)[:, 1]
        y_test_scores_sift2 = svm2.predict_proba(X_test_selected)[:, 1]


        y_test_scores_d1 = (y_test_scores_ri1 +y_test_scores_sift2)/2
        y_test_scores_d2 = (y_test_scores_ri2 +y_test_scores_sift1)/2

        
        #   Aplying the  EER threshol in the new data set 
        y_test_pred_d1 = (y_test_scores_d1 >= eer_threshold).astype(int)
        y_test_pred_d2 = (y_test_scores_d2 >= eer_threshold).astype(int)

        # Calculating FAR and FRR in the new data set
        tn1, fp1, fn1, tp1 = confusion_matrix(self.y_test_ri, y_test_pred_d1).ravel()
        FRR_test1 = fn1 / (fn1 + tp1)  # False Rejection Rate
        FAR_test1 = fp1 / (fp1 + tn1)  # False Acceptance Rate

    
        tn2, fp2, fn2, tp2 = confusion_matrix(self.y_test_sift, y_test_pred_d2).ravel()
        FRR_test2 = fn2 / (fn2 + tp2)  # False Rejection Rate
        FAR_test2 = fp2 / (fp2 + tn2)  # False Acceptance Rate

        FRR_test = (FRR_test1 +FRR_test2)/2
        FAR_test = (FAR_test1 + FAR_test2)/2

       

        # Calculating HTER
        HTER = (FAR_test + FRR_test) / 2

        print(f"Test Set - FAR: {FAR_test}, FRR: {FRR_test}, HTER: {HTER}")
        return FAR_test, FRR_test, HTER, eer_threshold,training_time
