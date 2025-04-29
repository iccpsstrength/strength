# Importing libraries
import pandas as pd
import numpy as np
from z3 import *
from itertools import combinations
import itertools
import json
from ActualADM import *


CO2_FRESH_AIR = 400             # CO2 concentration (ppm) of fresh air
TEMP_FRESH_AIR = 91.4           # Temperature (33 degree F) of the fresh air
CP_AIR = 1.026                  # Specific heat of fresh air
DEF_TEMP_SUPPLY_AIR =  55.4     # Default temperature (degree fahrenheit) of supply air (13 degree celsius)
MINUTES_IN_A_DAY = 1440         # Number of minutes in a day
OFF_PEAK_ENERGY_COST = 0.34     # OFF-PEAK energy cost (USD)
ON_PEAK_ENERGY_COST = 0.48      # ON-PEAK energy cost (USD)
ON_PEAK_START_SLOT = 960        # ON-PEAK start time (minute in a day)
ON_PEAK_END_SLOT = 1260         # ON-PEAK end time (minute in a day)
BATTER_STORAGE = 0.48           # Energy (kWh) produced by battery

zone_temp_setpoint = [0, 75.2, 75.2, 75.2, 75.2]     # list of temperature (fahrenheit) setpoint of the different zones
zone_co2_setpoint = [0, 1000, 1000, 1000, 1000]      # list of CO2 (ppm) setpoint of the corresponding zones
control_time = 1                                     # control time (in minute)

class AttackCost:
    
    def __init__(self, house_name, dataframe, cleaned_dataframe_occ_1, cleaned_dataframe_occ_2, processed_dataframe, list_time_min_occ_1, list_time_max_occ_1, list_time_min_occ_2, list_time_max_occ_2, attack_schedule_occ_1, attack_schedule_occ_2, dict_control_cost, num_timeslots, num_zones, activity_zone_map, zone_cost):
        
        self.house_name              = house_name
        self.dataframe               = dataframe
        self.cleaned_dataframe_occ_1 = cleaned_dataframe_occ_1
        self.cleaned_dataframe_occ_2 = cleaned_dataframe_occ_2
        self.processed_dataframe     = processed_dataframe
        self.attack_schedule_occ_1   = attack_schedule_occ_1
        self.attack_schedule_occ_2   = attack_schedule_occ_2
        self.list_time_min_occ_1     = list_time_min_occ_1
        self.list_time_max_occ_1     = list_time_max_occ_1
        self.list_time_min_occ_2     = list_time_min_occ_2
        self.list_time_max_occ_2     = list_time_max_occ_2
        self.activity_zone_map       = activity_zone_map
        
        self.num_timeslots           = num_timeslots
        self.num_zones               = num_zones
        self.dict_control_cost       = dict_control_cost
        
        self.zone_cost              = zone_cost
        
    def to_tuple(self, a):
        try:
            return tuple(self.to_tuple(i) for i in a)
        except TypeError:
            return a
        
    def get_attack_costs_without_appliance_triggering(self):
        attack_costs = []
        unit_energy_costs = []
        energy_consumptions = []
        prev_day = -1
        battery_usage = 0 # in kWh
        
        for j in range(0, len(self.dataframe), self.num_timeslots):

            for i in range(len(self.attack_schedule_occ_1)):
                current_minute = i     
                control_zone_occupants = self.dataframe.iloc[j + i][2:].values
                zone_occupants = [0, 0, 0, 0, 0]
                
                if control_zone_occupants[0] == 2:
                    zone_occupants[0] = 2
                elif control_zone_occupants[0] == 1:
                    zone_occupants[0] = 1
                    zone_occupants[int(self.attack_schedule_occ_1[i])] += 1
                elif control_zone_occupants[0] == 0:
                    zone_occupants[int(self.attack_schedule_occ_1[i])] += 1
                    zone_occupants[int(self.attack_schedule_occ_2[i])] += 1
                energy_consumption = self.dict_control_cost[self.to_tuple(zone_occupants)]
        
                if current_minute < ON_PEAK_START_SLOT or current_minute > ON_PEAK_END_SLOT:
                    unit_energy_cost = OFF_PEAK_ENERGY_COST
                else:
                    if battery_usage < BATTER_STORAGE:
                        battery_usage += energy_consumption
                        unit_energy_cost = OFF_PEAK_ENERGY_COST
                    else:
                        unit_energy_cost = ON_PEAK_ENERGY_COST
        
                current_cost = energy_consumption * unit_energy_cost
                attack_costs.append(current_cost)
                unit_energy_costs.append(unit_energy_cost)
                energy_consumptions.append(energy_consumption)
        
                #shatter_cost_house_A.append(dict_control_cost[to_tuple(zone_occupants)]) 
        day_wise_costs = []
        for i in range(0, 43200, 1440):
            day_wise_costs.append(sum(attack_costs[i:i+1400]))
        return day_wise_costs
    
        
    def get_attack_costs_with_appliance_triggering(self):
        
        with_appliance_costs_occ1 = []
        
        ## house A occ 1
        for i in range(0, 43200, 1440):
            cost = 0
            entrances = []
            prev_zone = self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][i]]
            attackable = np.zeros((1440))
            for j in range(i, i + 1440):
                zone_occupants = [0, 0, 0, 0, 0]
                zone_occupants[self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][j]]] += 1
                zone_occupants[self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][j]]] += 1
                # print(zone_occupants)
                if self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][j]] != prev_zone:
                    entrances.append([j%1440, self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][j]]])
                    prev_zone = self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][j]]
            
            for j in range(len(entrances) - 1):
                zone = entrances[j][1]
                time = entrances[j][0] 
                #print(entrances[j], zone, time)
                if len(self.list_time_min_occ_1[zone][time]) == 0:
                    attack_time = 0
                else:
                    attack_time = min(entrances[j + 1][0] - entrances[j][0], self.list_time_min_occ_1[zone][time][0])
                attackable[entrances[j][0] : entrances[j][0] + attack_time] = 1
                #print(entrances[j][0], attack_time)
            for j in range(len(attackable)):
                if attackable[j] == 1:
                    COST_PER_KWH = 0.395
                    cost += (self.zone_cost[self.attack_schedule_occ_1[j]] / 60000 ) * COST_PER_KWH
            with_appliance_costs_occ1.append(cost)
            
            
        with_appliance_costs_occ2 = []
        
        ## house A occ 1
        for i in range(0, 43200, 1440):
            cost = 0
            entrances = []
            prev_zone = self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][i]]
            attackable = np.zeros((1440))
            for j in range(i, i + 1440):
                zone_occupants = [0, 0, 0, 0, 0]
                zone_occupants[self.activity_zone_map[self.processed_dataframe['Occupant 1 Activity'][j]]] += 1
                zone_occupants[self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][j]]] += 1
                # print(zone_occupants)
                if self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][j]] != prev_zone:
                    entrances.append([j%1440, self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][j]]])
                    prev_zone = self.activity_zone_map[self.processed_dataframe['Occupant 2 Activity'][j]]
                    
            for j in range(len(entrances) - 1):
                zone = entrances[j][1]
                time = entrances[j][0] 
                #print(entrances[j], zone, time)
                #print(list_time_min_house_A_occ_1[zone][time])
                if len(self.list_time_min_occ_2[zone][time]) == 0:
                    attack_time = 0
                else:
                    attack_time = min(entrances[j + 1][0] - entrances[j][0], self.list_time_min_occ_2[zone][time][0])
                attackable[entrances[j][0] : entrances[j][0] + attack_time] = 1
                #print(entrances[j][0], attack_time)
            for j in range(len(attackable)):
                if attackable[j] == 1:
                    COST_PER_KWH = 0.395
                    cost += (self.zone_cost[self.attack_schedule_occ_2[j]] / 60000 ) * COST_PER_KWH
            with_appliance_costs_occ2.append(cost)
                
        return np.array(with_appliance_costs_occ1) + np.array(with_appliance_costs_occ2)
