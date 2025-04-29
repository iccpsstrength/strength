# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import metrics
warnings.filterwarnings("ignore")
import json

class HyperparameterTuning:
    
    def __init__(self, adm_algo, dataframe, zone_id, hyper_param_info):
        
        self.adm_algo         = adm_algo  # ["DBSCAN" or "K-Means"]
        self.dataframe        = dataframe  
        self.zone_id          = zone_id
        self.hyper_param_info = hyper_param_info

    def get_radious_with_max_score(self, radiouses, samples, scores):
        """
        Returns the radious and sample corresponding to the maximum score.
        :param radiouses: List of radiouses
        :param scores: List of score values
        :return: radious, sample corresponding to max score
        """
        
        index_of_max_scores = scores.index(max(scores))
        return radiouses[index_of_max_scores], samples[index_of_max_scores]
    
    def get_ks_with_max_score(self, ks, scores):
        """
        Returns the k corresponding to the maximum score.
        :param ks: List of ks
        :param scores: List of score values
        :return: k corresponding to max score
        """
        
        index_of_max_scores = scores.index(max(scores))
        return ks[index_of_max_scores]
    
    def get_best_hyperparameters(self):
            
        self.dataframe = self.dataframe[['Occupant\'s Zone', 'Zone Arrival Time (Minute)', 'Stay Duration (Minute)']]
        self.features = self.dataframe[self.dataframe['Occupant\'s Zone'] == self.zone_id].iloc[:, 1:].values
        
        if self.adm_algo == "DBSCAN":
            return self.get_best_DBSCAN_hyperparameters()
        
        elif self.adm_algo == "K-Means":
            return self.get_best_KMeans_hyperparameters()
        
    def get_best_DBSCAN_hyperparameters(self):
        
        radiouses = []
        samples   = []
        scores    = []
        
        for sample in range(self.hyper_param_info['min_samples'], self.hyper_param_info['max_samples']):
            for radious in range(self.hyper_param_info['min_radiouses'], self.hyper_param_info['max_radiouses']):
                try:
                    # DBSCAN clustering model
                    adm = DBSCAN(eps = radious, min_samples = sample).fit(self.features)
                    
                    # Get Davies Bouldin score
                    score = davies_bouldin_score(self.features, adm.labels_)
                    
                    # Get labels for each point
                    labels = adm.labels_
                            
                    radiouses.append(radious)
                    samples.append(sample)
                    scores.append(score)
                    
                except:
                    pass
        
        best_hyper_params = {"best_eps" : self.get_radious_with_max_score(radiouses, samples, scores)[0], "best_samples": self.get_radious_with_max_score(radiouses, samples, scores)[1]}
        
        return best_hyper_params


    def get_best_KMeans_hyperparameters(self):
        
        ks      = []
        scores  = []
        
        
        for k in range(self.hyper_param_info['min_ks'], self.hyper_param_info['max_ks']):
            
                try:
                    # DBSCAN clustering model
                    adm = KMeans(n_clusters = k).fit(self.features)
                    
                    # Get Davies Bouldin score
                    score = davies_bouldin_score(self.features, adm.labels_)
                    
                    # Get labels for each point
                    labels = adm.labels_
                            
                    ks.append(k)
                    scores.append(score)
                    
                except:
                    pass
        
        best_hyper_params = {"best_ks" : self.get_ks_with_max_score(ks, scores)}
        
        return best_hyper_params

                                                                               
                                                                                                     