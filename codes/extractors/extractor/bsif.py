
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

class BSIF:
    def __init__(self, channel) -> None:
        self.channel = channel


    def load_filters(self, bit):
        # Carregar filtros pré-treinados de acordo com a quantidade de bits
        filters = sio.loadmat(f'/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/BsifFilters/ICAtextureFilters_7x7_{bit}bit.mat')['ICAtextureFilters']
        return filters

    def apply_filters(self, filters):
        # Aplicar filtros à imagem
        h, w = self.channel.shape
        bsif_features = np.zeros((h, w, filters.shape[2]), dtype=np.uint8)
        for i in range(filters.shape[2]):
            filtered_image = cv2.filter2D(self.channel, -1, filters[:,:,i])
            bsif_features[:,:,i] = filtered_image > 0  # Binarizar as respostas
        return bsif_features

    def bsif(self):
        all_bsif_features = []
        for bit in range(5,13):  
            filters = self.load_filters(bit)
            bsif_features = self.apply_filters(filters)
            #print(bsif_features.shape)
            all_bsif_features.append(bsif_features)
        all_bsif_features= np.concatenate(all_bsif_features,axis=2)
        return all_bsif_features


    def binary_codes_to_integers(self):
        """
        Convert binary codes to integer values for visualization.
        
        Parameters:
        binary_codes (ndarray): Binary codes of shape (image_height, image_width, num_filters).
        
        Returns:
        int_codes (ndarray): Integer codes of shape (image_height, image_width).
        """
        binary_codes = self.bsif()
        int_codes = np.zeros(binary_codes.shape[:2], dtype=int)
        for i in range(binary_codes.shape[2]):
            int_codes += (binary_codes[:, :, i].astype(int) << i)
        return int_codes

    def visualize_bsif_codes(self):
        """
        Visualize the integer BSIF codes.
        
        Parameters:
        int_codes (ndarray): Integer BSIF codes of shape (image_height, image_width).
        """
        int_codes = self.binary_codes_to_integers()
        plt.imshow(int_codes, cmap='gray')
        plt.colorbar()
        plt.title('BSIF Codes')
        plt.show()
