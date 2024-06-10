import sys

sys.path.append('/home/mds3/Downloads/Visao Computacional-20240430T145425Z-001/ProjetoVisao/codes/extractors/extractor')

from bsif import *
from local_binary_pattern_based import *
from local_phase_quantisation import *
from SIFT import *
import cv2
import numpy as np
import os

class ApplyingExtractors:
    def __init__(self,
                 hsv_channels: list,
                 ycbcr_channels: list ) -> None:
        
        self.hsv = hsv_channels
        self.ycbcr = ycbcr_channels
        
    def extracting_hsv_channel(self):
        #lists to append the extracted features 
        features_hsv_bsif = []
        features_hsv_uniform = []
        features_hsv_ri = []
        features_hsv_co = []
        features_hsv_rico = []
        features_hsv_lpq = []
        
        # extrating for each extractor 

        for channel in self.hsv:

            #BSIF 
            Bsif = BSIF(channel)
            features = Bsif.bsif()
            features_hsv_bsif.append(features)
            
            #LBP Based features extractors
            lbp = LbpBased(channel)
            ## uniform
            features = lbp.Uniform_lbp()
            features_hsv_uniform.append(features)
            ## RI 
            features = lbp.RI_lbp()
            features_hsv_ri.append(features)
            #lbp.visualize_ri()
            #coALBP
            features, lbp_feat = lbp.coALBP()
            features_hsv_co.append(features)
            #riCoALBP
            features, lbp_fea = lbp.RIcoALBP()
            features_hsv_rico.append(features)

            #LPQ
            features = lpq(channel=channel , winSize=7,freqestim=1,mode='im')
            features_hsv_lpq.append(features)
            
            

        # conactenating all channels in hsv
        features_hsv_bsif = np.concatenate(np.array(features_hsv_bsif),axis=1)
        features_hsv_rico = np.concatenate(np.array(features_hsv_rico),axis=1)
        features_hsv_co = np.concatenate(np.array(features_hsv_co),axis=1)
        features_hsv_ri = np.concatenate(np.array(features_hsv_ri),axis=1)
        features_hsv_uniform = np.concatenate(np.array(features_hsv_uniform),axis=1)
        features_hsv_lpq = np.concatenate(np.array(features_hsv_lpq),axis=1)
        
        return features_hsv_bsif, features_hsv_ri, features_hsv_co, features_hsv_rico, features_hsv_uniform, features_hsv_lpq,
    def extracting_ycbcr_channel(self):
       #lists to append the extracted features
       features_ycbcr_bsif = []
       features_ycbcr_uniform = []
       features_ycbcr_ri = []
       features_ycbcr_co = []
       features_ycbcr_rico = []
       features_ycbcr_lpq = []
      # extrating for each extractor


       for channel in self.ycbcr:

           print('BSIF')
           #BSIF
           Bsif = BSIF(channel)
           features = Bsif.bsif()
           features_ycbcr_bsif.append(features)
           print('lbp')
           #LBP Based features extractors
           lbp = LbpBased(channel)
           ## uniform
           features = lbp.Uniform_lbp()
           features_ycbcr_uniform.append(features)
           ## RI
           
           features = lbp.RI_lbp()
           features_ycbcr_ri.append(features)
           #lbp.visualize_ri()
           #coALBP
           features, lbp_feat = lbp.coALBP()
           features_ycbcr_co.append(features)
           #riCoALBP
           features, lbp_fea = lbp.RIcoALBP()
           features_ycbcr_rico.append(features)

           print('lpq')
           #LPQ
           features = lpq(channel=channel , winSize=7,freqestim=1,mode='im')
           features_ycbcr_lpq.append(features)
           
           
       # conactenating all channels in ycbcr
       features_ycbcr_bsif = np.concatenate(np.array(features_ycbcr_bsif),axis=1)
       features_ycbcr_rico = np.concatenate(np.array(features_ycbcr_rico),axis=1)
       features_ycbcr_co = np.concatenate(np.array(features_ycbcr_co),axis=1)
       features_ycbcr_ri = np.concatenate(np.array(features_ycbcr_ri),axis=1)
       features_ycbcr_uniform = np.concatenate(np.array(features_ycbcr_uniform),axis=1)
       features_ycbcr_lpq = np.concatenate(np.array(features_ycbcr_lpq),axis=1)
      
       return features_ycbcr_bsif, features_ycbcr_ri, features_ycbcr_co, features_ycbcr_rico, features_ycbcr_uniform, features_ycbcr_lpq
    

    def extracting_both(self):
        #colecing the features
        features_ycbcr_bsif, features_ycbcr_ri, features_ycbcr_co, features_ycbcr_rico, features_ycbcr_uniform, features_ycbcr_lpq,  = self.extracting_ycbcr_channel()
        features_hsv_bsif, features_hsv_ri, features_hsv_co, features_hsv_rico, features_hsv_uniform, features_hsv_lpq = self.extracting_hsv_channel()
        
        #concatening fatures from the same extractor for both channels

        features_all_rico= np.concatenate((features_hsv_rico,features_ycbcr_rico ), axis=1)
        features_all_co= np.concatenate(( features_hsv_co, features_ycbcr_co ),axis=1)
        features_all_ri= np.concatenate(( features_hsv_ri, features_ycbcr_ri ), axis=1)
        features_all_uniform= np.concatenate((features_hsv_uniform , features_ycbcr_uniform), axis=1)
        features_all_bsif = np.concatenate((features_hsv_bsif, features_ycbcr_bsif ), axis=1)
        features_all_lpq = np.concatenate((features_hsv_lpq, features_ycbcr_lpq), axis=1)
      
        ## turning all of it into flatten vectors
        features_all_rico= features_all_rico.flatten()                      
        features_all_co=  features_all_co.flatten()                         
        features_all_ri=   features_all_ri.flatten()                           
        features_all_uniform= features_all_uniform.flatten()                       
        features_all_bsif = features_all_bsif.flatten()                    
        features_all_lpq = features_all_lpq.flatten()                   
        
        return features_all_rico, features_all_co, features_all_ri, features_all_ri, features_all_uniform, features_all_bsif, features_all_lpq

    def saving_features_extracted(self, path2save: str, token:str):
        features_all_rico, features_all_co, features_all_ri, features_all_ri, features_all_uniform, features_all_bsif, features_all_lpq= self.extracting_both()
        path = path2save
        os.makedirs(os.path.join(path, 'RI-CoALBP'), exist_ok=True)
        os.makedirs(os.path.join(path, 'CoALBP'), exist_ok=True)
        os.makedirs(os.path.join(path, 'RI-LBP'), exist_ok=True)
        os.makedirs(os.path.join(path, 'UniLBP'), exist_ok=True)
        os.makedirs(os.path.join(path, 'BSIF'), exist_ok=True)
        os.makedirs(os.path.join(path, 'LPQ'), exist_ok=True)
    

        PATH_rico =os.path.join(path, 'RI-CoALBP')
        PATH_co = os.path.join(path, 'CoALBP')
        PATH_ri = os.path.join(path, 'RI-LBP')
        PATH_uniform = os.path.join(path, 'UniLBP')
        PATH_bsif = os.path.join(path, 'BSIF') 
        PATH_lpq = os.path.join(path, 'LPQ') 
        
        print('---------- STARTING TO SAVE ----------')
        np.save(PATH_rico+f'/features_all_rico_{token}.npy', features_all_rico)
        np.save(PATH_co+f'/features_all_co_{token}.npy', features_all_co)
        print('------------- 2 saved ---------------')
        np.save(PATH_ri+f'/features_all_ri_{token}.npy', features_all_ri)
        np.save(PATH_uniform+f'/features_all_uniform_{token}.npy', features_all_uniform)
        print('------------- 2 more save --------------')
        np.save(PATH_bsif+f'/features_all_bsif_{token}.npy', features_all_bsif)
        np.save(PATH_lpq+f'/features_all_lpq_{token}.npy', features_all_lpq)
       
        print('---------------- ALL FILES SAVED---------')
    def applying_sift_hsv(self):
        features = []
        
        # extrating for each extractor 

        for i,channel in enumerate(self.hsv):
            print(i)
            sift = SIFT(channel)
            sift_features, keypoints_array = sift.Sift()
            #print(sift_features.shape)
                
            if sift_features is  None:
                break
            else:
                features.append(sift_features)
            # Apenas concatenar e salvar se não houver características vazias
        if len(features)!=0:
            features = np.concatenate(features, axis=0)
            print('sift h',features.shape)
        return features
    
    def applying_sift_ycbcr(self):
        features = []
        
        # extrating for each extractor 

        for i,channel in  enumerate(self.ycbcr):      
            print(i)
            sift = SIFT(channel)
            sift_features, keypoints_array = sift.Sift()
            #print(sift_features.shape)
            #print(channel)   
            if sift_features is  None:
                break
            else:
                features.append(sift_features)
            # Apenas concatenar e salvar se não houver características vazias
        if len(features)!=0:
            features = np.concatenate(features, axis=0)
            print('sift y', features.shape)
        return features
    
    def applying_sift(self):
        
        print('applying hsv')   
        features_hsv_sift = self.applying_sift_hsv()
        print('aoolying y')
        features_ycbcr_sift = self.applying_sift_ycbcr()
        if len(features_hsv_sift)==0:
            return features_ycbcr_sift
        elif len(features_ycbcr_sift)==0:
            return features_hsv_sift
        elif len(features_ycbcr_sift)==0 and  len(features_ycbcr_sift)==0:
            features_all= []  
            print('BOTH EMPTY') 
            return features_all
        else:
            features_all = np.concatenate((features_hsv_sift, features_ycbcr_sift))
            return features_all

    def saving_sift(self,path2save: str, token:str):
        features_all = self.applying_sift()
        os.makedirs(os.path.join(path2save, 'SIFT'), exist_ok=True)
        PATH_sift =os.path.join(path2save, 'SIFT')
        if len(features_all)!=0:
            np.save(PATH_sift+f'/features_all_sift_{token}.npy', features_all)
            print('SIFT saved')
            return features_all
        else:
            print('Not saved. empty set')

    
    
    def applying_rilbp_ycbcr(self):
        features_ext = []
        
        # extrating for each extractor 

        for channel in self.ycbcr:
            
        
            lbp = LbpBased(channel)
            features = lbp.RI_lbp()
            features_ext.append(features)
        print('ri ycbcr', np.array(features_ext).shape)
        return np.array(features_ext)

    def applying_rilbp_hsv(self):
        features_ext = []
        
        # extrating for each extractor 

        for channel in self.hsv:
        
            lbp = LbpBased(channel)
            features = lbp.RI_lbp()
            features_ext.append(features)
        print('ri lbp', np.array(features_ext).shape)
        return features_ext
    def applying_rilbp(self):
        ri_hsv = self.applying_rilbp_hsv()
        ri_ycbcr = self.applying_rilbp_ycbcr()
        ri_all = np.concatenate((ri_hsv, ri_ycbcr),axis=1)
        return ri_all 
    
    def proposal(self, path2save:str, token:str):
        sift_all = self.applying_sift()
        print('SIFTALL',np.array(sift_all).shape)
        os.makedirs(os.path.join(path2save, 'RI-LBP+SIFT'), exist_ok=True)
        PATH_sift =os.path.join(path2save, 'RI-LBP+SIFT')
        if len(sift_all)!=0:
            features_all = features_all.reshape(len(features_all,) 64,64)
            ri_all = self.applying_rilbp()
            ri_all = ri_all.flatten()
            sift_all= sift_all.flatten()
            features_all = np.concatenate((ri_all, sift_all),axis=0)
            features_all=features_all.flatten()
            #print(features_all.shape)
            #np.save(PATH_sift+f'/features_all_risift_{token}.npy', features_all)
            print('PROPOSAL saved')