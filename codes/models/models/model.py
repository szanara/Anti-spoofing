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

class SvmModel:
    def __init__(self, X_train, X_test, y_train, y_test,X_dev, y_dev) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.X_dev = X_dev
        self.y_train = y_train
        self.y_test = y_test
        self.y_dev = y_dev

    
    def SVM(self, plot_roc:list, plot_Testconfusion: list):
        plot_roc,path2save = plot_roc
        plot_Testconfusion, path2save = plot_Testconfusion
        svm = SVC(kernel='linear', probability=True, verbose=True)
        start_time = time.time()
        svm.fit(self.X_train, self.y_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Tempo de treinamento: {training_time} segundos")
        # Predict probabilities in the dev set
        y_dev_scores = svm.predict_proba(self.X_dev)[:, 1]

        # Calculating EER
        fpr, tpr, thresholds = roc_curve(self.y_dev, y_dev_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        
        if plot_roc:
            # Plotting  ROC
            print(' ***** PRINTING ROC CURVE FROM DEVELOV SET *****')
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Chance')
            plt.xlabel('False Predicted Rate (FPR)')
            plt.ylabel('True Predicted Rate(TPR)')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(path2save+'/Roc Curve.jpg', dpi=300)
            plt.show()
        
        # Aplying  EER treshold in develop set
        y_dev_pred = (y_dev_scores >= eer_threshold).astype(int)

        # Calculating  FAR and FRR  in the dev set using the EER treshold
        tn, fp, fn, tp = confusion_matrix(self.y_dev, y_dev_pred).ravel()
        FRR = fn / (fn + tp)  # False Rejection Rate
        FAR = fp / (fp + tn)  # False Acceptance Rate
        print(f"Development Set - EER Threshold: {eer_threshold}, FAR: {FAR}, FRR: {FRR}")

        y_test_scores = svm.predict_proba(self.X_test)[:, 1]

        #   Aplying the  EER threshol in the new data set 
        y_test_pred = (y_test_scores >= eer_threshold).astype(int)

        # Calculating FAR and FRR in the new data set
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_test_pred).ravel()
        FRR_test = fn / (fn + tp)  # False Rejection Rate
        FAR_test = fp / (fp + tn)  # False Acceptance Rate


        if plot_Testconfusion:
            print(' ***** PRINTING CONFUSION MATRIX FROM TEST SET *****')
            cm = confusion_matrix(self.y_test, y_test_pred)
            cm_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
            # Ploting confusion matrix 
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative Prediction', ' Positive Prediction'],
                        yticklabels=['True Negative', 'True Positive'])
            plt.xlabel('Predicted Class ')
            plt.ylabel('True Class ')
            plt.title('COnfusion Matrix')
            plt.savefig(path2save+'/Confusion Matrix.jpg', dpi=300)
            plt.show()

        # Calculating HTER
        HTER = (FAR_test + FRR_test) / 2

        print(f"Test Set - FAR: {FAR_test}, FRR: {FRR_test}, HTER: {HTER}")
        return FAR_test, FRR_test, HTER, eer_threshold,training_time

class PCA_features:
    def __init__(self, X_train, X_test, X_dev, n_components,path2save) -> None:
        self.X_train = X_train
        self.X_test = X_test 
        self.X_dev = X_dev
        self.components = n_components
        self.path = path2save
    
    def obtaining(self,):
        pca = PCA(n_components=self.components)
        print('Reducing X_train')
        dados_train = pca.fit_transform(self.X_train)
        print('Reducing X_test')
        dados_test = pca.fit_transform(self.X_test)
        print('Reducing X_dev')
        dados_dev = pca.fit_transform(self.X_dev)
        np.save(self.path+f'/train_pca_{self.components}.npy',dados_train)
        np.save(self.path+f'/test_pca_{self.components}.npy',dados_test)
        np.save(self.path+f'/dev_pca_{self.components}.npy',dados_dev)
        return dados_train, dados_dev, dados_test

class SoftmaxModel(nn.Module):
    def __init__(self, input_shape):
        super(SoftmaxModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_shape[1] // 8), 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 2)
        #self.softmax = nn.Softmax(dim=1)  # Specify dim for softmax
        
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(nn.ReLU()(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        #x = self.softmax(x)
        return x
    

class ModelTrainer:
    def __init__(self, X_train, y_train, X_dev, y_dev, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_test = X_test
        self.y_test = y_test

    def training_model(self, epochs, batch_size, input_shape, plot_roc:list, plot_Testconfusion: list):
        plot_roc,path2save = plot_roc
        plot_Testconfusion, path2save = plot_Testconfusion
        model = SoftmaxModel(input_shape)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        x_train = torch.tensor(np.expand_dims(self.X_train, axis=-1), dtype=torch.float32)
        x_train = x_train.permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, channels]
        y_train = torch.tensor(self.y_train, dtype=torch.long)
        x_dev = torch.tensor(np.expand_dims(self.X_dev, axis=-1), dtype=torch.float32)
        x_dev = x_dev.permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, channels]
        y_dev = torch.tensor(self.y_dev, dtype=torch.long)
        x_test = torch.tensor(np.expand_dims(self.X_test, axis=-1), dtype=torch.float32)
        x_test = x_test.permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, channels]
        y_test = torch.tensor(self.y_test, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_dev, y_dev = x_dev.to(device), y_dev.to(device)
        x_test , y_test = x_test.to(device), y_test.to(device)
        start_time = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                batch_x_train = x_train[i:i+batch_size]
                batch_y_train = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_x_train)
                loss = criterion(outputs, batch_y_train)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                del batch_x_train, batch_y_train, outputs, loss
                torch.cuda.empty_cache()


            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (len(x_train) / batch_size)}")
        end_time = time.time()

        
        training_time = end_time - start_time
        print(f"Tempo de treinamento: {training_time} segundos")
        # Prediction on dev set
        with torch.no_grad():
            outputs_dev = model(x_dev)
            y_dev_scores = nn.functional.softmax(outputs_dev, dim=1)[:, 1].cpu().numpy()

        # Calculate EER
        fpr, tpr, thresholds = roc_curve(self.y_dev, y_dev_scores)
        fnr = 1 - tpr
        eer_threshold =  thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

        if plot_roc:
            # Plotting  ROC
            print(' ***** PRINTING ROC CURVE FROM DEVELOV SET *****')
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Chance')
            plt.xlabel('False Predicted Rate (FPR)')
            plt.ylabel('True Predicted Rate(TPR)')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(path2save+'/Roc Curve.jpg', dpi=300)
            plt.show()
         # Aplying the threshold EER in the dev set
        y_dev_pred = (y_dev_scores >= eer_threshold).astype(int)

        # Calcule FAR and FRR  in the dev set using the threshld EER
        tn, fp, fn, tp = confusion_matrix(self.y_dev, y_dev_pred).ravel()
        FRR = fn / (fn + tp)  # False Rejection Rate
        FAR = fp / (fp + tn)  # False Acceptance Rate
        print(f"Development Set - EER Threshold: {eer_threshold}, FAR: {FAR}, FRR: {FRR}")


        # Probs in test set
        with torch.no_grad():
            outputs_test = model(x_test)
            y_test_scores = nn.functional.softmax(outputs_test, dim=1)[:, 1].cpu().numpy()

        # Applying the threshold EER in test set
        y_test_pred = (y_test_scores >= eer_threshold).astype(int)

        # Calculate FAR and FRR in the test set
        tn, fp, fn, tp = confusion_matrix(y_test.cpu(), y_test_pred).ravel()
        FRR_test = fn / (fn + tp)  # False Rejection Rate
        FAR_test = fp / (fp + tn)  # False Acceptance Rate

        if plot_Testconfusion:
            print(' ***** PRINTING CONFUSION MATRIX FROM TEST SET *****')
            cm = confusion_matrix(self.y_test, y_test_pred)
            cm_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
            # Ploting confusion matrix 
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative Prediction', ' Positive Prediction'],
                        yticklabels=['True Negative', 'True Positive'])
            plt.xlabel('Predicted Class ')
            plt.ylabel('True Class ')
            plt.title('COnfusion Matrix')
            plt.savefig(path2save+'/Confusion Matrix.jpg', dpi=300)
            plt.show()

        # Calculate HTER
        HTER = (FAR_test + FRR_test) / 2

        print(f"Test Set - FAR: {FAR_test}, FRR: {FRR_test}, HTER: {HTER}")
        return FAR_test, FRR_test, HTER, eer_threshold,training_time







       
    """def training_model(self, epochs, batch_size, input_shape, plot_roc=True, plot_Testconfusion=True):
        model = SoftmaxModel(input_shape)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        x_train = torch.tensor(np.expand_dims(self.X_train, axis=-1), dtype=torch.float32)
        y_train = torch.tensor(self.y_train, dtype=torch.long)
        x_dev = torch.tensor(np.expand_dims(self.X_dev, axis=-1), dtype=torch.float32)
        y_dev = torch.tensor(self.y_dev, dtype=torch.long)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_dev, y_dev = x_dev.to(device), y_dev.to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Prediction on dev set
        with torch.no_grad():
            outputs_dev = model(x_dev)
            y_dev_scores = nn.functional.softmax(outputs_dev, dim=1)[:, 1].cpu().numpy()

        # Calculate EER
        fpr, tpr, thresholds = roc_curve(self.y_dev, y_dev_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

        if plot_roc:
            # Plotting  ROC
            print(' ***** PRINTING ROC CURVE FROM DEVELOV SET *****')
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Chance')
            plt.xlabel('False Predicted Rate (FPR)')
            plt.ylabel('True Predicted Rate(TPR)')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

        # Aplying the threshold EER in the dev set
        y_dev_pred = (y_dev_scores >= eer_threshold).astype(int)

        # Calcule FAR and FRR  in the dev set using the threshld EER
        tn, fp, fn, tp = confusion_matrix(self.y_dev, y_dev_pred).ravel()
        FRR = fn / (fn + tp)  # False Rejection Rate
        FAR = fp / (fp + tn)  # False Acceptance Rate
        print(f"Development Set - EER Threshold: {eer_threshold}, FAR: {FAR}, FRR: {FRR}")

        # Probs in test set
        y_test_scores = model.predict(x_test).ravel()

        # Applying the threshold  EER in test set
        y_test_pred = (y_test_scores >= eer_threshold).astype(int)

        # Calcule FAR and FRRin the test set
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_test_pred).ravel()
        FRR_test = fn / (fn + tp)  # False Rejection Rate
        FAR_test = fp / (fp + tn)  # False Acceptance Rate
        if plot_Testconfusion:
            print(' ***** PRINTING CONFUSION MATRIX FROM TEST SET *****')
            cm = confusion_matrix(self.y_test, y_test_pred)
            cm_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
            # Ploting confusion matrix 
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative Prediction', ' Positive Prediction'],
                        yticklabels=['True Negative', 'True Positive'])
            plt.xlabel('Predicted Class ')
            plt.ylabel('True Class ')
            plt.title('COnfusion Matrix')
            plt.show()

        # Calcule HTER
        HTER = (FAR_test + FRR_test) / 2

        print(f"Test Set - FAR: {FAR_test}, FRR: {FRR_test}, HTER: {HTER}")
        return FAR_test, FRR_test, HTER"""