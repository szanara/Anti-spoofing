import numpy as np
import cv2
import matplotlib.pyplot as plt



class SIFT:
    def __init__(self, channel) -> None:
        self.channel = channel

    def Sift(self):
        sift = cv2.SIFT_create()
        #kp = sift.detect(self.channel, None)
        keypoints, descriptors = sift.detectAndCompute(self.channel, None)

        img_with_keypoints = cv2.drawKeypoints(self.channel, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Extrair as características dos keypoints e salvar como array
        keypoints_array = np.array([[keypoint.pt[0], keypoint.pt[1], keypoint.size] for keypoint in keypoints])
        
        return descriptors, keypoints_array

    def visualize_sift(self, img_with_keypoints, keypoints_array):
    
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with Keypoints')
        plt.show()

        # Plotar os keypoints separadamente
        plt.figure(figsize=(10, 10))
        plt.imshow(self.channel, cmap='gray')
        plt.scatter(keypoints_array[:, 0], keypoints_array[:, 1], c='r', s=keypoints_array[:, 2]*10, edgecolors='k')
        plt.title('Keypoints plottedo n  image')
        plt.show()