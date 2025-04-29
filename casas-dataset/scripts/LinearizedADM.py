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

class LinearizedADM:
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
            
        self.dataframe = self.dataframe[['Occupant\'s Zone', 'Zone Arrival Time (Minute)', 'Stay Duration (Minute)']]
        
        
        list_clusters = []
        
        for zone_id in range(self.num_zones):

            try:  
                features = self.dataframe[self.dataframe['Occupant\'s Zone'] == zone_id].iloc[:, 1:].values
                hyperparams = self.get_best_hyper_params(zone_id)
                
                if self.adm_algo == "DBSCAN":
                    cluster_model = self.clustering_dbscan(features, hyperparams) 
                
                elif self.adm_algo == "K-Means":
                    cluster_model = self.clustering_kmeans(features, hyperparams)             
            
                labels = cluster_model.labels_
                
                vertices = []
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                for cluster in range(n_clusters):
                    points = []
                    for k in range(len(labels)):
                        if labels[k] == cluster:
                            points.append(features[k])
                    points = np.array(points)
        
                    if len(points) >= 3:
                        try:
                            vertices = self.convex_hull(zone_id, points)
                        except:
                            pass
                            #print("Exception")
                    else:
                        pass
                        #print("zone", zone_id, "Problem Found!")
        
                    list_clusters.append({"zone_id": zone_id, "cluster_id": cluster, "points": vertices})
            except:
                pass
    
        return list_clusters


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
    
 
    def range_calculation(self):
        """
        '''''''Calculating the ranges (i.e., valid ranges) of of the cluster

        ''''''''Parameters:
            list[dict]: List dictionary (i.e., constaints zone id (int), cluster id (int), vertices)
            
        Returns:
            list[list[list]] and list[list[list]]: returns minimum and maximum valied ranges for a particular zone, time, and cluster
        """
        
        list_cluster = self.get_clusters()
        
        list_time_min = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
        list_time_max = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
    
        for i in range(len(list_cluster)):
            zone_id = list_cluster[i]["zone_id"]
            min_x_range = self.num_timeslots
            max_x_range = 0
    
            ##################################################################
            ##################### Zone Constraints ###########################
            ##################################################################
            x = Int('x')
            y = Int('y')
    
            points = list_cluster[i]["points"]
            
            # Convert each element in the 2D list to int
            points = [[int(value) for value in sublist] for sublist in points]
            
            
            zone_constraints = []
            
            and_constraints = []
            
            for j in range(len(points) - 1):
        
                and_constraints.append(self.is_left(x, y, points[j][0], points[j][2], points[j + 1][0], points[j + 1][2]))
    
            zone_constraints.append(And(and_constraints))
    
            #print(zone_constraints)
            ####### Minimum value of X range #######
            o = Optimize()
            o.add(zone_constraints)
            o.minimize(x)
            o.check()
            
            min_x_range = int(str(o.model()[x]))
    
            ####### Maximum value of X range #######
            o = Optimize()
            o.add(zone_constraints)
            o.maximize(x)
            o.check()
            #print(o.model()[x])
    
            max_x_range = int(str(o.model()[x]))
            
            for j in range(min_x_range, max_x_range):
                ####### Minimum value of Y range #######
                o = Optimize()
                o.add(zone_constraints)
                o.add(x == j)
                o.minimize(y)
                o.check()
    
                min_y_range = o.model()[y]
                if min_y_range == None:
                    min_y_range = 0
    
                ####### Maximum value of Y range #######
                o = Optimize()
                o.add(zone_constraints)
                o.add(x == j)
                o.maximize(y)
                o.check()
    
                max_y_range = o.model()[y]
                if max_y_range == None:
                    max_y_range = 0
                    
                list_time_min[zone_id][j].append(int(str(min_y_range)))
                list_time_max[zone_id][j].append(int(str(max_y_range)))

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

        return list_time_min, list_time_max
                