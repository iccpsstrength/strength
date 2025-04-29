# Importing libraries
import pandas as pd
import numpy as np
from z3 import *
from itertools import combinations

# Fixed Parameter
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


# Variable Parameters
zone_temp_setpoint = [0, 75.2, 75.2, 75.2, 75.2]     # list of temperature (fahrenheit) setpoint of the different zones
zone_co2_setpoint = [0, 1000, 1000, 1000, 1000]      # list of CO2 (ppm) setpoint of the corresponding zones
control_time = 1                                     # control time (in minute)

class ControlCost:
    
    def __init__(self, house_name, dataframe, zones, zone_volume, activities, energy_appliances, benign_activity_appliances_map):
        
        self.house_name                     = house_name
        self.dataframe                      = dataframe
        self.zones                          = zones
        self.zone_volume                    = zone_volume
        self.activities                     = activities
        self.energy_appliances              = energy_appliances
        self.benign_activity_appliances_map = benign_activity_appliances_map
    
    def get_control_costs(self):
        
        self.possible_control_costs()
        
        all_control_costs = self.dataframe.copy()
        
        control_costs = []
        unit_energy_costs = []
        energy_consumptions = []
        
        prev_day = -1
        battery_usage = 0 # in kWh
        
        num_activities = len(self.activities)  
        
        for i in range(len(self.dataframe)):
            if i % 1440 == 0:
                print("House: " + str(self.house_name) + ", Day: " + str(int(i / 1400)))
            
            current_day = int(self.dataframe.iloc[i,0])
            current_minute = int(self.dataframe.iloc[i,1])
            
            if current_day!= prev_day:
                battery_usage = 0
                prev_day = current_day    
                
            activity_zone_occupants = [0] * (num_activities + 1)
            occ_1_activity_id = self.dataframe["Occupant 1 Activity"][i]
            occ_2_activity_id = self.dataframe["Occupant 2 Activity"][i]
            activity_zone_occupants[occ_1_activity_id] += 1
            activity_zone_occupants[occ_2_activity_id] += 1
            
            energy_consumption = self.dict_control_cost[self.to_tuple(activity_zone_occupants)]
            
            if current_minute < ON_PEAK_START_SLOT or current_minute > ON_PEAK_END_SLOT:
                unit_energy_cost = OFF_PEAK_ENERGY_COST
            else:
                if battery_usage < BATTER_STORAGE:
                    battery_usage += energy_consumption
                    unit_energy_cost = OFF_PEAK_ENERGY_COST
                else:
                    unit_energy_cost = ON_PEAK_ENERGY_COST
                    
            current_cost = energy_consumption * unit_energy_cost
            control_costs.append(current_cost)
            unit_energy_costs.append(unit_energy_cost)
            energy_consumptions.append(energy_consumption)
            
        all_control_costs['Unit Energy Cost ($)'] = unit_energy_costs
        all_control_costs['Energy Comsumption (kWh)'] = energy_consumptions
        all_control_costs['Control Cost ($)'] = control_costs 
        
        return all_control_costs
    
        
    def to_tuple(self, a):
        try:
            return tuple(self.to_tuple(i) for i in a)
        except TypeError:
            return a
    
    # Generating possible combination of occupants
    def possible_control_costs(self):
        
        
        num_activities = len(self.activities)  

        activity_zone_map = dict()
        for i in range(num_activities):
            activity_zone_map[int(self.activities["Activity ID"][i])] = int(self.activities["Zone ID"][i])

          
        activity_zone_temp_setpoint = [0]
        activity_zone_co2_setpoint = [0] 
        activity_pp_co2 = [0] + self.activities['CO2 Emission by Occupant (CFM)'].to_list()    # CO2 Emission by Occupant (cfm)
        activity_pp_heat = [0] + self.activities["Heat Radiation by Occupant (W)"].to_list()   # Heat Radiation by Occupant (W)
        activity_load = [0]                                                               # Heat radiated by Appliances (W)
        activity_zone_volume = [0]

        for i in range(1, num_activities + 1):
            activity_zone_temp_setpoint.append(zone_temp_setpoint[activity_zone_map[i]])
            activity_zone_co2_setpoint.append(zone_co2_setpoint[activity_zone_map[i]])
            activity_zone_volume.append(self.zone_volume[activity_zone_map[i]])
            activity_load.append(sum(self.benign_activity_appliances_map[i] * self.energy_appliances))
        
        
        activities = [i for i in range(1, num_activities + 1)]
        
        indexes = list(combinations(activities, 2))
        
        unique_samples = np.zeros((len(indexes) + num_activities, num_activities + 1))
        
        for i in range(len(indexes)):
            unique_samples[i][indexes[i][0]] = 1
            unique_samples[i][indexes[i][1]] = 1
        
        count = 1
        for i in range(len(indexes), len(indexes) + num_activities):
            unique_samples[i][count] = 2
            count += 1
        
        self.dict_control_cost = dict()
        
        for i in range(len(unique_samples)):
            sample = unique_samples[i]
            activity_zone_occupant = sample.tolist()
            self.dict_control_cost[self.to_tuple(sample)] = self.control_cost(activity_zone_occupant, activity_zone_temp_setpoint, activity_zone_volume, activity_pp_co2, activity_pp_heat, activity_load, activity_zone_co2_setpoint, control_time)
        
    
    # Returns energy consumption (kWh) based on the zone, activities, and appliances parameters at current timeslots
    def control_cost(self, activity_zone_occupant, activity_zone_temp_setpoint, activity_zone_volume, activity_pp_co2, activity_pp_heat, activity_load, activity_zone_co2_setpoint, control_time):
        '''
        PARAMETERS:
        activity_zone_occupant: list of occupants performing different activities
        activity_zone_temp_setpoint: list of temperature (fahrenheit) setpoint of the different zones
        activity_zone_volume: # Zones' volumes (cubic feet)
        activity_pp_co2: CO2 Emission by Occupant (cfm) performing corresponding activity
        activity_pp_heat: Heat Radiation by Occupant (W)
        activity_load: Heat radiated by Appliances (W)
        activity_zone_co2_setpoint: list of CO2 (ppm) setpoint of the corresponding zones
        control_time: time of control operation (in minute)
        '''
        
        num_activities = len(activity_zone_occupant)
        # initializing z3 variables
        v_vent_air = [Real( 'v_vent_air_' + str(i)) for i in range(num_activities)]   # Air required for ventillation (CFM)
        v_temp_air = [Real( 'v_temp_air_' + str(i)) for i in range(num_activities)]   # Air required for cooling (CFM)
        v_mixed_air = [Real( 'v_mixed_air_' + str(i)) for i in range(num_activities)]
        v_fresh_air = [Real( 'v_fresh_air_' + str(i)) for i in range(num_activities)]
        v_return_air = [Real( 'v_return_air_' + str(i)) for i in range(num_activities)]
        zone_cost = [Real( 'zone_cost' + str(i)) for i in range(num_activities)] 
        
        temp_supply_air = [ Real( 'temp_supply_air_' + str(i)) for i in range(num_activities)]
        temp_mixed_air = [ Real( 'temp_mixed_air_' + str(i)) for i in range(num_activities)]
        co2_mixed_air = [ Real( 'co2_mixed_air_' + str(i)) for i in range(num_activities)]
        total_zone_cost = Real('total_zone_cost')

        s = Solver()
        
        for i in range(1, num_activities):
            ############### v_vent_air ###############################
            
            s.add(activity_zone_occupant[i] * ((activity_pp_co2[i] * 1000000) / activity_zone_volume[i]) == 
                        (activity_zone_co2_setpoint[i] - (( 1 - (v_vent_air[i]) /activity_zone_volume[i]) * activity_zone_co2_setpoint[i] +  
                                                (v_vent_air[i] * CO2_FRESH_AIR) /  activity_zone_volume[i])))
            
            ############### v_temp_air ###############################
            if activity_zone_occupant[i] > 0:
                s.add(v_temp_air[i] *  (activity_zone_temp_setpoint[i] - DEF_TEMP_SUPPLY_AIR) * 0.3167 == activity_zone_occupant[i] * activity_pp_heat[i] + activity_load[i])
            else:
                s.add(v_temp_air[i] *  (activity_zone_temp_setpoint[i] - DEF_TEMP_SUPPLY_AIR) * 0.3167 == activity_zone_occupant[i] * activity_pp_heat[i]) 
        
            ############### v_mixed_air ###############################
            
            s.add(activity_zone_occupant[i] * ((activity_pp_co2[i] * 1000000) / activity_zone_volume[i]) == 
                  (activity_zone_co2_setpoint[i] - (( 1 - ( v_mixed_air[i] ) / activity_zone_volume[i]) * activity_zone_co2_setpoint[i] + 
                                         ( v_mixed_air[i] * co2_mixed_air[i]) / activity_zone_volume[i])))
        
            if activity_zone_occupant[i] > 0:
                s.add( v_mixed_air[i] * (activity_zone_temp_setpoint[i] - temp_supply_air[i]) * 0.3167 == activity_zone_occupant[i] * activity_pp_heat[i] + activity_load[i])
            else:
                s.add( v_mixed_air[i] * (activity_zone_temp_setpoint[i] - temp_supply_air[i]) * 0.3167 == activity_zone_occupant[i] * activity_pp_heat[i]) 

            s.add(v_mixed_air[i] == v_return_air[i] + v_fresh_air[i])
            s.add(co2_mixed_air[i] == activity_zone_co2_setpoint[i] * (v_return_air[i] / v_mixed_air[i]) + CO2_FRESH_AIR * (v_fresh_air[i] / v_mixed_air[i]))
            s.add(temp_mixed_air[i] == activity_zone_temp_setpoint[i] * (v_return_air[i] / v_mixed_air[i]) + TEMP_FRESH_AIR * (v_fresh_air[i] / v_mixed_air[i]))
        
            ############### temperature control algorithm ############
            s.add(Implies(v_vent_air[i] >= v_temp_air[i] , v_return_air[i] == 0))
            s.add(Implies(v_vent_air[i] < v_temp_air[i] ,  temp_supply_air[i] == 55.4))
        
            ############### other constraints ########################
            s.add(v_return_air[i] >= 0)
            s.add(temp_supply_air[i] >= 55.4)
            
            ############## cost constraint ###########################
            s.add(zone_cost[i] == v_mixed_air[i] * (temp_mixed_air[i] - DEF_TEMP_SUPPLY_AIR) * 0.3167 * (control_time / 60000) )
        s.add(total_zone_cost == Sum(zone_cost[1:]))
        s.check()
        
        for i in range(1, num_activities):
            v_vent_air[i] = float(Fraction(str(s.model()[v_vent_air[i]])))
            v_temp_air[i] = float(Fraction(str(s.model()[v_temp_air[i]])))
        
            v_mixed_air[i] = float(Fraction(str(s.model()[v_mixed_air[i]])))
            temp_mixed_air[i] = float(Fraction(str(s.model()[temp_mixed_air[i]])))
        
            temp_supply_air[i] = float(Fraction(str(s.model()[temp_supply_air[i]])))
        
            co2_mixed_air[i] = float(Fraction(str(s.model()[co2_mixed_air[i]])))
            v_return_air[i] = float(Fraction(str(s.model()[v_return_air[i]])))
            v_fresh_air[i] = float(Fraction(str(s.model()[v_fresh_air[i]])))
            
            zone_cost[i] = float(Fraction(str(s.model()[zone_cost[i]])))
        total_zone_cost = float(Fraction(str(s.model()[total_zone_cost])))
            

        return total_zone_cost
    
    
    
    