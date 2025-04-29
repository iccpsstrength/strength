import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import warnings
from z3 import *
import os
import sys
import json
import time
import copy


class DeadlockElimination:

    def __init__(self, num_timeslots, num_zones, list_time_min, list_time_max):        
        
        self.num_timeslots = num_timeslots
        self.num_zones = num_zones
        self.list_time_min = list_time_min
        self.list_time_max = list_time_max
        self.next_states = dict()
        
    def get_rewards(self):
        
        return [0, 1, 2, 4, 3]
    
    def get_sorted_duration(self, arrival_time, arrival_zone):
        
        sorted_durations = set()
        
        for cluster in range(len(self.list_time_min[arrival_zone][arrival_time])):
            for duration in range(self.list_time_min[arrival_zone][arrival_time][cluster], self.list_time_max[arrival_zone][arrival_time][cluster] + 1):
                sorted_durations.add(int(duration))
        sorted_durations = list(sorted_durations)
        sorted_durations = sorted(sorted_durations)
        if 0 in sorted_durations:
            sorted_durations.remove(0)
        return sorted_durations
    
    
    def get_all_possible_states(self):
        
        #states = ['*-*-*']
        states = []
        
        for arrival_time in range(self.num_timeslots):
            for arrival_zone in range(self.num_zones):
                stay_durations = self.get_sorted_duration(arrival_time, arrival_zone)
                for stay_duration in stay_durations:
                    if stay_duration > 0:
                        states.append(str(arrival_time) + '-' + str(arrival_zone) + '-' + str(stay_duration))
        return states
    
    
    def get_all_possible_actions(self):
        
        actions = []
        
        for zone in range(self.num_zones):
            actions.append(zone)
        return actions
    

    
    def generate_next_states(self, states, state_indexes):
                
        next_states = {}
        count = 0
        for i in range (len(states)): 
            try:
                state = states[i]
                
                arrival_time = int(state.split('-')[0])
                arrival_zone = int(state.split('-')[1])
                state_stay_duration = int(state.split('-')[2])
            
                ##############################################################
                ################ determine current state #####################
                ##############################################################

                arrival_zone_time_stay_durations = self.get_sorted_duration(arrival_time, arrival_zone)
                
                for stay_duration in range (len(arrival_zone_time_stay_durations) - 1):
                    if arrival_zone_time_stay_durations[stay_duration] == state_stay_duration:
                        next_state = str(arrival_time) + '-' + str(arrival_zone) + '-' + str(arrival_zone_time_stay_durations[stay_duration + 1])
                        self.append_to_dict(next_states, i, state_indexes[next_state])
                                    
                actions = self.get_all_possible_actions()
                
                for action in actions:
                    
                    if action != arrival_zone:
                        
                        next_arrival_time = arrival_time + state_stay_duration
                        
                        stay_durations = self.get_sorted_duration(next_arrival_time, action)
                        
                        if len(stay_durations) != 0:
                            
                            next_arrival_zone = action
                            next_stay_duration = stay_durations[0]
                            if next_stay_duration != 0:
                                next_state = str(next_arrival_time) + '-' + str(next_arrival_zone) + '-' + str(next_stay_duration)
                                self.append_to_dict(next_states, i, state_indexes[next_state])
                
            except:
                pass
        return next_states
    
    def deadlock_elimination(self):
        
        start_time = time.time()
        
        states = self.get_all_possible_states()
        
        init_states = len(states)
        
        split_state = lambda s: list(map(int, s.split('-')))
        
        while True:
            memory = np.full((self.num_zones, self.num_timeslots, self.num_timeslots + 1), -1)
        
            for state in states:
                arrival_time, arrival_zone, state_stay_duration = split_state(state)
                try:
                    memory[arrival_zone][arrival_time][state_stay_duration] = 1
                except:
                    pass
        
            self.list_time_min = [[[] for _ in range(self.num_timeslots)] for _ in range(self.num_zones)]
            self.list_time_max = [[[] for _ in range(self.num_timeslots)] for _ in range(self.num_zones)]
        
            for arrival_zone in range(self.num_zones):
                for arrival_time in range(self.num_timeslots):
                    list_time_min_zone = self.list_time_min[arrival_zone]
                    list_time_max_zone = self.list_time_max[arrival_zone]
                    flag = 0
                    for duration in range(1, self.num_timeslots):
                        if flag == 0 and memory[arrival_zone][arrival_time][duration] == 1:
                            list_time_min_zone[arrival_time].append(duration)
                            flag = 1
                            if duration == self.num_timeslots - 1:
                                list_time_max_zone[arrival_time].append(duration)
                                flag = 0
        
                        if flag == 1 and memory[arrival_zone][arrival_time][duration] == -1:
                            list_time_max_zone[arrival_time].append(duration - 1)
                            flag = 0
        
            # Use sets for faster lookups
            at_zones = {}
            for i, state in enumerate(states):
                arrival_time, arrival_zone, state_stay_duration = split_state(state)
                self.append_set_to_dict(at_zones, arrival_time, arrival_zone)
        
            count = 0
            states_to_pop = set()
            for i, state in enumerate(states):
                arrival_time, arrival_zone, state_stay_duration = split_state(state)
                if arrival_time + state_stay_duration == 1440:
                    continue
                if arrival_time + state_stay_duration not in at_zones:
                    count += 1
                    states_to_pop.add(i)
                if state_stay_duration in self.list_time_max[arrival_zone][arrival_time]:
                    if arrival_time + state_stay_duration < self.num_timeslots:
                        if self.list_time_min[arrival_zone][arrival_time + state_stay_duration] != []:
                            count += 1
                            states_to_pop.add(i)
        
            states = [state for i, state in enumerate(states) if i not in states_to_pop]
            final_states = len(states)
            
            state_indexes = {state: i for i, state in enumerate(states)}
            next_states = self.generate_next_states(states, state_indexes)
            
            if count == 0:
                break
        
        print("Execution Time:", time.time() - start_time)
        return {
            "List-Time-Min": self.list_time_min,
            "List-Time-Max": self.list_time_max,
            "States": states,
            "State-Indexes": state_indexes,
            "Next-States": next_states,
            "Before-Removal-States": init_states,
            "After-Removal-States": final_states,
            "Execution-Time": time.time() - start_time
        }
    
    # Function to append a value to a key in the dictionary
    def append_to_dict(self, d, key, value):
        d.setdefault(key, []).append(value)
    
    def append_set_to_dict(self, d, key, value):
        d.setdefault(key, set()).add(value)
        
    def remove_indexes(self, input_list, indexes):
        return [value for index, value in enumerate(input_list) if index not in indexes]    
    