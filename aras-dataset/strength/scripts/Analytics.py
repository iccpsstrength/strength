import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import warnings
from z3 import *
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from ActualADM import *
import time

import copy


# Function to append a value to a key in the dictionary
def append_to_dict(d, key, value):
    d.setdefault(key, []).append(value)


class Analytics:

    def __init__(self, num_timeslots, num_zones, list_time_min, list_time_max, num_episodes, num_iterations, epsilon, learning_rate, discount_factor):        
        
        self.num_timeslots = num_timeslots
        self.num_zones = num_zones
        self.list_time_min = list_time_min
        self.list_time_max = list_time_max
        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = dict()
        self.next_states = dict()
        self.prev_states = dict()
        self.final_states = []
        
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
        print(len(states))
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
                flag = True
                
                arrival_zone_time_stay_durations = self.get_sorted_duration(arrival_time, arrival_zone)
                
                for stay_duration in range (len(arrival_zone_time_stay_durations) - 1):
                    if arrival_zone_time_stay_durations[stay_duration] == state_stay_duration:
                        next_state = str(arrival_time) + '-' + str(arrival_zone) + '-' + str(arrival_zone_time_stay_durations[stay_duration + 1])
                        flag = False
                        append_to_dict(next_states, i, state_indexes[next_state])
                                    
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
                                flag = False
                                append_to_dict(next_states, i, state_indexes[next_state])
                if flag == True:
                    append_to_dict(next_states, i, -1)
                    count += 1
            except:
                #print("Exception in", i, state, arrival_time + state_stay_duration)
                append_to_dict(next_states, i, len(states))
        print("count", count)
        return next_states

if __name__ == "__main__":
    
    num_episodes = 10
    num_iterations = 1000
    num_timeslots = 1440
    num_zones = 5
    eps = 30
    min_samples = 3
    epsilon = 0.7
    learning_rate = 0.8
    discount_factor = 0.9

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
            
    house_name = 'A'
    occupant_id = '1'   
    dataframe = pd.read_csv(str(parent_directory) + '\data\\cleaned\\' + 'Cleaned-Dataframe_House-' + str(house_name) + '_Occupant-' + str(occupant_id) + '.csv')                                                                                                       
    adm_algo = 'DBSCAN'
    knowledge = 'Full'
    
    actual_adm = ActualADM(adm_algo, dataframe, knowledge, house_name, occupant_id, num_timeslots, num_zones)
    list_time_min, list_time_max = actual_adm.noise_augmented_range_calculation()
    analytics = Analytics(num_timeslots, num_zones, list_time_min, list_time_max, num_episodes, num_iterations, epsilon, learning_rate, discount_factor)
    #total_costs, attack_schedules = analytics.model_training()
    
    states = analytics.get_all_possible_states()
    states_time = []
    states_zones = []
    states_stay = []
    
    state_indexes = {}
    
    for i in range(len(states)):
        state = states[i]
        arrival_time = int(state.split('-')[0])
        arrival_zone = int(state.split('-')[1])
        state_stay_duration = int(state.split('-')[2])
        
        states_time.append(arrival_time)
        states_zones.append(arrival_zone)
        states_stay.append(state_stay_duration)
        
        
        state_indexes[states[i]] = i
    
    next_states = {}
    
    next_states = analytics.generate_next_states(states, state_indexes)
    
    
    
    sz_wind   = 5
    rwrds     = [0, 1, 2, 4, 3]
    
    z3_states    = [Int( 'z3_states_' + str(i)) for i in range(sz_wind)]
    z3_sch_zn = [Int( 'z3_sch_zn_' + str(i)) for i in range(sz_wind)]
    z3_sch_at = [Int( 'z3_sch_at_' + str(i)) for i in range(sz_wind)]
    z3_sch_sd = [Int( 'z3_sch_sd_' + str(i)) for i in range(sz_wind)]
    
    cst_wnd = [Int( 'cst_wnd_' + str(i)) for i in range(sz_wind)]
    
    tot_cst = Int('tot_cst')

    start_time = time.time()
    s = Optimize()
    
    for i in range(sz_wind):
        s.add(z3_states[i] >= 0)
        s.add(z3_states[i] < len(states))
        
    for w in range(sz_wind):
        for t in range(len(states)):
            s.add(Implies(z3_states[w] == t, And(z3_sch_at[w] == states_time[t], z3_sch_zn[w] == states_zones[t], z3_sch_sd[w] == states_stay[t])))
     
    for w in range(sz_wind - 1):
        for t in range(len(states)):
            or_cnstrnts = []
            for cnstrnts in next_states[t]:
                or_cnstrnts.append(z3_states[w + 1] == cnstrnts)
            s.add(Implies(z3_states[w] == t, Or(or_cnstrnts)))
    
    for w in range(sz_wind):
        s.add(z3_states[w] != -1)
    
    for i in range(len(rwrds)):
        s.add(Implies(z3_sch_zn[0] == i, cst_wnd[0] == z3_sch_sd[0] * rwrds[i]))
        
    for i in range(1, sz_wind):
        for j in range(len(rwrds)):
            s.add(Implies(And(z3_sch_zn[i] == j, z3_sch_zn[i] == z3_sch_zn[i - 1]), cst_wnd[i] == (z3_sch_sd[i] - z3_sch_sd[i - 1]) * rwrds[j]))
            s.add(Implies(And(z3_sch_zn[i] == j, z3_sch_zn[i] != z3_sch_zn[i - 1]), cst_wnd[i] == z3_sch_sd[i] * rwrds[j]))
    
    s.add(tot_cst == Sum(cst_wnd))
    
    s.add(z3_sch_at[0] == 0)
    
    s.maximize(tot_cst)
    
    print(s.check())
    
    for i in range(sz_wind):
        print("states", i, s.model()[z3_states[i]], states[int(str(s.model()[z3_states[i]]))])
        print(i, s.model()[cst_wnd[i]])
    
    print("Total cost", s.model()[tot_cst])
    print("Execution time", time.time() - start_time)