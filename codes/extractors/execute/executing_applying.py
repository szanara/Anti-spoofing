from applying import*
import os
import numpy as np
sys.path.append('/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/codes/preprocessing/')

from pre_processing import *

path2save = '/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/FeaturesExtracted'

path1 ='/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/datasets_originals/LCCFASD'
pathx = '/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/datasets_originals/MiniDataset'



files1 = os.listdir(pathx)

for dataset in files1:
    #print('dataset',dataset)
    path2 = os.path.join(pathx, dataset)
    files2 = os.listdir(path2)
    tokens = [i for i in range(len(files2))]
    for image, token in zip(files2,tokens):
        read = ReadImage(path2, image)
        image2 = read.read_file()
        process = ProcessingImage(image2)
        if process.extractor() is None:
            continue
        else:
            [ch1_hsv, ch2_hsv, ch3_hsv],[ch1_ycbcr, ch2_ycbcr, ch3_ycbcr] =process.extractor()
            app = ApplyingExtractors( hsv_channels = [ch1_hsv, ch2_hsv, ch3_hsv],
                          ycbcr_channels = [ch1_ycbcr, ch2_ycbcr, ch3_ycbcr]
                        )
            
            #print(dataset)
            savecomplete =path2save+f'/MiniDataset/{dataset}/'
            print(savecomplete)
            os.makedirs(savecomplete, exist_ok=True)
            app.proposal(path2save = savecomplete,
                             token = token) 

files2 = os.listdir(path1)
for dataset in files2:
    print('dataset',dataset)
    path2 = os.path.join(path1, dataset)
    files2 = os.listdir(path2)
    print(files2)
    for conjunto in files2:
        path3 = os.path.join(path2, conjunto)
        files3 = os.listdir(path3)
        #print(path3, files3)
        tokens = [i for i in range(len(files3))]
        for image, token  in zip(files3, tokens):
            read = ReadImage(path3, image)
            image2 = read.read_file()
            process = ProcessingImage(image2)
            if process.extractor() is None:
                continue
            else:
                [ch1_hsv, ch2_hsv, ch3_hsv],[ch1_ycbcr, ch2_ycbcr, ch3_ycbcr] =process.extractor()
                print(ch1_hsv.shape)
                app = ApplyingExtractors( hsv_channels = [ch1_hsv, ch2_hsv, ch3_hsv],
                              ycbcr_channels = [ch1_ycbcr, ch2_ycbcr, ch3_ycbcr]
                            )
                print(conjunto,'conjunto')
                if 'development' in dataset:
                    sa = 'DEVELOP'
                elif 'evaluation' in dataset:
                    sa = 'EVALUATION'
                else:
                    sa = 'TRAIN'
                if conjunto =='spoof':
                    tag = 'Real'
                else:
                    tag = 'Spoof'
                
                savecomplete =path2save+f'/LCC_FASD/{sa}/{tag}'
                print(savecomplete)
                os.makedirs(savecomplete, exist_ok=True)
                returned = app.proposal(path2save = savecomplete,
                                 token = token) 
                
