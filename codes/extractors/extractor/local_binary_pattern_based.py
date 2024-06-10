import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


class LbpBased:
    def __init__(self, channel) -> None:
        self.channel = channel
        self.radius = 1
        self.n_points = 8 * self.radius  # Deve ser 8 * radius, não apenas 8

    def Uniform_lbp(self):
        lbp_uniform = local_binary_pattern(self.channel, self.n_points, self.radius, method='uniform')
        return lbp_uniform

    def RI_lbp(self):
        lbp_ror = local_binary_pattern(self.channel, self.n_points, self.radius, method='ror')
        return lbp_ror

    def coALBP(self):
        # Calcular LBP
        lbp = local_binary_pattern(self.channel, self.n_points, self.radius, method='uniform')
        
        # Definir parâmetros da matriz de co-ocorrência
        distances = [1]  # Distância para a co-ocorrência
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ângulos para a co-ocorrência
        max_value = int(lbp.max() + 1)
        
        # Calcular matriz de co-ocorrência
        glcm = graycomatrix(lbp.astype(np.uint8), distances, angles, levels=max_value, symmetric=True, normed=True)
        
        # Extrair propriedades da matriz de co-ocorrência
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        ASM = graycoprops(glcm, 'ASM')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        
        # Concatenar características
        coALBP_features = np.hstack([contrast, dissimilarity, homogeneity, ASM, energy, correlation])
        
        return coALBP_features,lbp
    
    def RIcoALBP(self):
        # Calcular LBP
        lbp = local_binary_pattern(self.channel, self.n_points, self.radius, method='ror')
        
        # Definir parâmetros da matriz de co-ocorrência
        distances = [1]  # Distância para a co-ocorrência
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ângulos para a co-ocorrência
        max_value = int(lbp.max() + 1)
        
        # Calcular matriz de co-ocorrência
        glcm = graycomatrix(lbp.astype(np.uint8), distances, angles, levels=max_value, symmetric=True, normed=True)
        
        # Extrair propriedades da matriz de co-ocorrência
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        ASM = graycoprops(glcm, 'ASM')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        
        # Concatenar características
        coALBP_features = np.hstack([contrast, dissimilarity, homogeneity, ASM, energy, correlation])
        
        return coALBP_features,lbp


    def visualize_uniform(self):
        lbp_uniform = self.Uniform_lbp()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(lbp_uniform, cmap='gray')
        plt.title('Uniform LBP')
        plt.show()

    def visualize_ri(self):
        lbp_ror = self.RI_lbp()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.imshow(lbp_ror, cmap='gray')
        plt.title('RI-LBP')
        plt.show()

    def visualize_co(self):
        coALBP_features, lbp_image = self.coALBP()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(self.channel, cmap='gray')
        ax[0].set_title('Original Image')
        
        ax[1].imshow(lbp_image, cmap='gray')
        ax[1].set_title('LBP Image')

        for a in ax:
            a.axis('off')

        plt.show()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(coALBP_features)), coALBP_features)
        plt.title('Co-occurrence of Adjacent Local Binary Patterns (coALBP) Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.show()

    def visualize_co_barrs(self):
        lbp_co, lbp= self.coALBP()
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(lbp_co)), lbp_co)
        plt.title('Co-occurrence of Adjacent Local Binary Patterns (coALBP) Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.show()
    
    def visualize_ri_co(self):
        coALBP_features, lbp_image = self.RIcoALBP()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(self.channel, cmap='gray')
        ax[0].set_title('Original Image')
        
        ax[1].imshow(lbp_image, cmap='gray')
        ax[1].set_title('LBP Image')

        for a in ax:
            a.axis('off')

        plt.show()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(coALBP_features)), coALBP_features)
        plt.title('Co-occurrence of Adjacent Local Binary Patterns (coALBP) Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.show()
    
    def visualize_ri_co_barrs(self):
        lbp_co, lbp= self.RIcoALBP()
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(lbp_co)), lbp_co)
        plt.title('Co-occurrence of Adjacent Local Binary Patterns (coALBP) Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.show()