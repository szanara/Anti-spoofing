import cv2
import os 
import matplotlib.pyplot as plt


# Load the pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ReadImage:
    def __init__(self, path, filename) -> None:
        self.path = path
        self.name = filename
    
    def read_file(self):
        file_name = os.path.join(self.path, self.name )
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def visualize_image(self):
        image = self.read_file()
        plt.imshow(image)
        plt.title(self.name)
        plt.show()



class ProcessingImage:
    def __init__(self, image) -> None:
        self.image = image
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_normalize_face(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("No face detected")
            return None

        for (x, y, w, h) in faces:
            face_region = self.image[y:y+h, x:x+w]
            
            # Resize to 64x64 pixels
            normalized_face = cv2.resize(face_region, (64, 64), interpolation=cv2.INTER_CUBIC)
            normalized_face = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB)
            
            return normalized_face
        
    def convert_to_hsv_ycbcr(self):
        normalized_face = self.detect_and_normalize_face()
        if normalized_face is None:
            return None, None
        
        # Convert RGB image to HSV color space
        hsv_image = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2HSV)
        
        # Convert RGB image to YCbCr color space
        ycbcr_image = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2YCrCb)
        
        return hsv_image, ycbcr_image
    

    def extractor(self):
        self.hsv, self.ycbcr = self.convert_to_hsv_ycbcr()
        if self.hsv is None or self.ycbcr is None:
            return None
        else:
            ch1_hsv, ch2_hsv, ch3_hsv = cv2.split(self.hsv)
            ch1_ycbcr, ch2_ycbcr, ch3_ycbcr = cv2.split(self.ycbcr)
            return [ch1_hsv, ch2_hsv, ch3_hsv],[ch1_ycbcr, ch2_ycbcr, ch3_ycbcr]

    def visualize_channel(self):
        [ch1_hsv, ch2_hsv, ch3_hsv],[ch1_ycbcr, ch2_ycbcr, ch3_ycbcr] = self.extractor()
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        # Plotando os canais HSV
        axs[0, 0].imshow(ch1_hsv, cmap='gray')
        axs[0, 0].set_title('Canal 1 (HSV)')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(ch2_hsv, cmap='gray')
        axs[0, 1].set_title('Canal 2 (HSV)')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(ch3_hsv, cmap='gray')
        axs[0, 2].set_title('Canal 3 (HSV)')
        axs[0, 2].axis('off')

        # Plotando os canais YCbCr
        axs[1, 0].imshow(ch1_ycbcr, cmap='gray')
        axs[1, 0].set_title('Canal 1 (YCbCr)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(ch2_ycbcr, cmap='gray')
        axs[1, 1].set_title('Canal 2 (YCbCr)')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(ch3_ycbcr, cmap='gray')
        axs[1, 2].set_title('Canal 3 (YCbCr)')
        axs[1, 2].axis('off')
        plt.show()

    
        