import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import warnings
from z3 import *
warnings.filterwarnings("ignore")
import os
import json


#####################################
########## Fixed Parameters #########
#####################################
NUM_ZONES     = 5
NUM_TIMESLOTS = 1440

class ActualADM:
    """
    A threat analytics for convex-hull representation of the clustering models (i.e., DBSCAN, K-Means)
    Attributes:
        name (str): The name of the calculator.
    """
    
    def __init__(self, adm_algo, dataframe, house_name, occupant_id, num_timeslots, num_zones):
        
        """
        Initializes a new approximate model instance

        Parameters:
            dataframe (DataFrame): Dataset to train the cluster models
        """
        
        self.adm_algo      = adm_algo  # ["DBSCAN" or "K-Means"]
        self.dataframe     = dataframe  
        self.house_name    = house_name
        self.occupant_id   = occupant_id
        self.num_timeslots = num_timeslots
        self.num_zones     = num_zones
        
    def get_best_hyper_params(self, zone_id):
     
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        filename = str(parent_directory) + '/data/hyperparameters/' + str(self.adm_algo) + '_House-' + str(self.house_name) + '_Occupant-' + str(self.occupant_id)  + '_Zone-' + str(zone_id) + '.json'
        
        with open(filename, "r") as json_file:
            json_string = json_file.read()
        
        return json.loads(json_string)

    def clustering_dbscan(self, data, hyperparams):
        """
        DBSCAN clustering from given data samples and specified model parameters

        Parameters:
            data (np array): Data features (i.e., from complete or partial dataset) to train the cluster models
            hyperparameters: Contains epsion and min_samples
        Returns:
            sklearn model: Clustering (i.e., sklearn.cluster.DBSCAN) model
        """
        
        db = DBSCAN(eps = hyperparams["best_eps"], min_samples = hyperparams["best_samples"])
        cluster = db.fit(data)

        return cluster
    
    def clustering_kmeans(self, data, hyperparams):
        """
        DBSCAN clustering from given data samples and specified model parameters

        Parameters:
            data (np array): Data features (i.e., from complete or partial dataset) to train the cluster models
            hyperparameters: Containts number of clusters
        
        Returns:
            sklearn model: Clustering (i.e., sklearn.cluster.KMeans) model
        """
        
        db = KMeans(n_clusters = hyperparams["best_ks"])
        cluster = db.fit(data)

        return cluster

    def convex_hull(self, zone_id, points):
        """
        Convex hull forming from a set of points
        
        Parameters:
            zone (int): Assiciated zone of the points, which are used to generate the convex hull
            points (ndarray of floats): Coordinates of points to construct a convex hull from 
        
        Returns:
            list[tuples]: Convex hull vertices, zone tuple (x-coordinate of the vertice, zone, y-coordinate of the vertice) in counter-clockwise orientation
        """
        
        hull = ConvexHull(points)
        simplices = hull.simplices
        
        vertices = []
        for index in hull.vertices:
            vertices.append((points[index][0], zone_id, points[index][1]))
        vertices.append((points[hull.vertices[0]][0], zone_id, points[hull.vertices[0]][1]))
        
        return vertices


    def get_clusters(self):
        """
        Acquiring cluster boundaries
        
        Parameters:
            dataframe (DataFrame): Dataframe (i.e., complete or partial dataset) to train the cluster models
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
            min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        
        Returns:
            list[dict]: List dictionary (i.e., constaints zone id (int), cluster id (int), vertices)
        """
        clusters = []
            
        self.dataframe = self.dataframe[['Occupant\'s Zone', 'Zone Arrival Time (Minute)', 'Stay Duration (Minute)']]
        
        
        for zone_id in range(self.num_zones):
            try:
                features = self.dataframe[self.dataframe['Occupant\'s Zone'] == zone_id].iloc[:, 1:].values
                hyperparams = self.get_best_hyper_params(zone_id)
                
                if self.adm_algo == "DBSCAN":
                    cluster_model = self.clustering_dbscan(features, hyperparams) 
                
            
                    labels = cluster_model.labels_
                    core_sample_indexes = cluster_model.core_sample_indices_
                
                    # Get the number of clusters
                    num_clusters = len(set(cluster_model.labels_)) - (1 if -1 in cluster_model.labels_ else 0)
                
                    zone_clusters = []
                    #for i in range(len(core_sample_indexes)):
                    #    print(labels[core_sample_indexes[i]])
                
                    for i in range(num_clusters):
                        arr = []
                        for j in range(len(core_sample_indexes)):
                            if labels[core_sample_indexes[j]] == i:
                                index = core_sample_indexes[j]
                                arrival = features[index][0]
                                stay = features[index][1]
                                
                                #if plot_circle_line_intersection(eps, arrival, stay, 546):
                                    
                                    
                                #print(eps, arrival, stay, plot_circle_line_intersection(eps, arrival, stay, 546))
                                arr.append((arrival, stay))
                            
                        zone_clusters.append(arr)
                        
                    clusters.append(zone_clusters)
            except:
                pass

        return clusters


    def is_left(self, x, y, init_x, init_y, final_x, final_y):
        """
        Checking a point is left of a line or not
        
        Parameters:
            x (float): x-coordinate of the point to check
            y (float): y-coordinate of the point to check
            init_x (float): x-coordinate of the starting point of the line
            init_y (float): y-coordinate of the starting point of the line
            final_x (float): x-coordinate of the final point of the line
            final_y (float): y-coordinate of the final point of the line
            
        Returns:
            bool: Whether the point is on the left or not
        """
        return ((final_x - init_x)*(y - init_y) - (final_y - init_y)*(x - init_x)) >= 0
    
    
    def circle_line_intersection(self, radius, center_x, center_y, line_x):
        # Generate y-coordinates of the intersection points
        if np.abs(center_x - line_x) <= radius:
            delta_x = np.abs(center_x - line_x)
            y1 = center_y + np.sqrt(radius**2 - delta_x**2)
            y2 = center_y - np.sqrt(radius**2 - delta_x**2)
            #print(y1,y2)
            if y1 > y2:
                temp = y1
                y1 = y2
                y2 = temp
            return y1, y2
        else:
            #print("No intersection points found!")
            return -1 
 
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    def range_calculation(self):
        """
        '''''''Calculating the ranges (i.e., valid ranges) of of the cluster

        ''''''''Parameters:
            list[dict]: List dictionary (i.e., constaints zone id (int), cluster id (int), vertices)
            
        Returns:
            list[list[list]] and list[list[list]]: returns minimum and maximum valied ranges for a particular zone, time, and cluster
        """
        list_time_min = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
        list_time_max = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
        

        
        clusters = self.get_clusters()
        
        for i in range(self.num_zones):
            try:
                hyperparams = self.get_best_hyper_params(i)
                
                if self.adm_algo == "DBSCAN":
                    self.eps = hyperparams["best_eps"]
                    
                    for j in range(self.num_timeslots):
                        try:
                            current_cluster = clusters[i]
                        except:
                            continue

                        for cluster in current_cluster:
                            max_point = -1
                            min_point = 1500
                            for circle in cluster:
                                #print(self.eps, circle[0], circle[1])
                                if self.circle_line_intersection(self.eps, circle[0], circle[1], j) != -1:
                                    #print("c", i, j, circle[0], circle[1], plot_circle_line_intersection(eps, circle[0], circle[1], j))
                                    if self.circle_line_intersection(self.eps, circle[0], circle[1], j)[0] >= 0:
                                        min_point = min(min_point, self.circle_line_intersection(self.eps, circle[0], circle[1], j)[0])
                                    else:
                                        min_point = 0
                
                                    if self.circle_line_intersection(self.eps, circle[0], circle[1], j)[1] >= 0:
                                        max_point = max(max_point, self.circle_line_intersection(self.eps, circle[0], circle[1], j)[1])
                                    else:
                                        max_point = 0
                                                        
                            if min_point != 1500:
                                if j + min_point > 1440:
                                    list_time_min[i][j].append(1440 - j)
                                else:
                                    list_time_min[i][j].append(int(min_point))
                            if max_point != -1:
                                if j + max_point > 1440:
                                    list_time_max[i][j].append(1440 - j)
                                else:
                                    list_time_max[i][j].append(int(max_point)) 
                
                
                elif self.adm_algo == "K-Means":
                    self.ks = hyperparams["best_ks"]
                    features = self.dataframe[self.dataframe['Occupant\'s Zone'] == i].iloc[:, 1:].values
                    cluster_model = self.clustering_kmeans(features, hyperparams)  
                    centroids = cluster_model.cluster_centers_
                    labels = cluster_model.labels_
                    
                    self.eps = []
                    for unique_label in np.unique(labels):
                        cluster_points = features[labels == unique_label]
                        centroid = centroids[unique_label]
                        distances = [self.euclidean_distance(point, centroid) for point in cluster_points]
                        self.eps.append(max(distances))
                    

                    for j in range(self.num_timeslots):

                        for k in range(len(centroids)):
                            centroid = centroids[k]
                            max_point = -1
                            min_point = 1500                       
    
                            if self.circle_line_intersection(self.eps[k], centroid[0], centroid[1], j) != -1:
                                #print("c", i, j, circle[0], circle[1], plot_circle_line_intersection(eps, circle[0], circle[1], j))
                                if self.circle_line_intersection(self.eps[k], centroid[0], centroid[1], j)[0] >= 0:
                                    min_point = min(min_point, self.circle_line_intersection(self.eps[k], centroid[0], centroid[1], j)[0])
                                else:
                                    min_point = 0
            
                                if self.circle_line_intersection(self.eps[k], centroid[0], centroid[1], j)[1] >= 0:
                                    max_point = max(max_point, self.circle_line_intersection(self.eps[k], centroid[0], centroid[1], j)[1])
                                else:
                                    max_point = 0
                                                    
                            if min_point != 1500:
                                if j + min_point > 1440:
                                    list_time_min[i][j].append(1440 - j)
                                else:
                                    list_time_min[i][j].append(int(min_point))
                            if max_point != -1:
                                if j + max_point > 1440:
                                    list_time_max[i][j].append(1440 - j)
                                else:
                                    list_time_max[i][j].append(int(max_point)) 
            except:
                pass
        return list_time_min, list_time_max
   
    
    def deadlock(self, list_time_min, arrival_time, num_zones):
        
        zones = []
        for zone in range(num_zones):
            try:
                if len(list_time_min[zone][arrival_time]) != 0:
                    zones.append(zone)
            except:
                pass
        if len(zones) == 0:
            return True
        return False
    
    def noise_augmented_range_calculation(self):
        
        list_time_min, list_time_max = self.range_calculation()
        
        num_benign = 0
        num_anomaly = 0
        data = self.dataframe.values


        for i in range(len(data)):
            zone_id = int(data[i][0])
            entrance = int(data[i][1])
            duration = int(data[i][2])
            flag = False
            for j in range(len(list_time_min[zone_id][entrance])):
                if duration >= list_time_min[zone_id][entrance][j] and duration <= list_time_max[zone_id][entrance][j]:
                    flag = True
                    num_benign +=1
            if flag == False:
                num_anomaly += 1
                list_time_min[zone_id][entrance].append(duration)
                list_time_max[zone_id][entrance].append(duration)
                
# =============================================================================
#         while(1):
#             count_pop = 0
#             count_max = 0
#             for arrival_time in range(self.num_timeslots):
#                 for arrival_zone in range(self.num_zones):
#                     for cluster in range(len(list_time_min[arrival_zone][arrival_time])):
#                         min_duration = list_time_min[arrival_zone][arrival_time][cluster]
#                         max_duration = list_time_max[arrival_zone][arrival_time][cluster]
#                         
#                         if self.deadlock(list_time_min, arrival_time + min_duration, self.num_zones) == True or max_duration == 0:
#                             list_time_min[arrival_zone][arrival_time].pop(cluster)
#                             list_time_max[arrival_zone][arrival_time].pop(cluster)
#                             count_pop += 1
#                             break
#                             
#                         for duration in range(list_time_min[arrival_zone][arrival_time][cluster], list_time_max[arrival_zone][arrival_time][cluster] + 1):
#                             if self.deadlock(list_time_min, arrival_time + duration, self.num_zones) == True:
#                                 if arrival_time + duration == 1440:
#                                     continue
#                                 list_time_max[arrival_zone][arrival_time][cluster] = duration - 1
#                                 count_max += 1
#                                 break
#             print(count_pop, count_max)
#             if count_pop == 0 and count_max == 0:
#                 break   
#             
# =============================================================================
        return list_time_min, list_time_max
    
    