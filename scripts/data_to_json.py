# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:17:55 2024

@author: no23sane
"""
# %%
import json

# transformer size taken from Schneider electric: https://www.se.com/us/en/faqs/FA101694/
xmer_size_list = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 25, 37.5, 50, 75, 100, 167, 250, 333]

tech_inputs = {
    'pf': 0.9,  # M Al Shammari IEEE Paper allows min 0.85, typical values 0.95: https://ieeexplore.ieee.org/document/8956982
    'xfmer_size_factor': 1.2,
    'xfmer_SP_size_list': xmer_size_list
}


# Define the file path
json_file_path = "inputs_tech_xmer.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(tech_inputs, json_file)


# %%
econ_inputs = {
    'retail_high': 14.53/100,
    'retail_low': 14.53/100,
    'fit_high': 5.6/100,
    'fit_low': 5.6/100,
    'p2p_high': 10/100,
    'p2p_low': 10/100,
    'standing_costs': 20.39/100,
    'disc_rate': 0.03,
    'pv_prices': {
        "0": 1562,
        "1": 1562,
        "2": 1562,
        "3": 1562,
        "4": 1704,
        "5": 1704,
        "6": 1704,
        "7": 1704,
        "8": 1704,
        "9": 1704,
        "10": 1077,
        "11": 1077,
        "12": 1077,
        "13": 1077,
        "14": 1077,
        "15": 1077,
        "16": 1077,
        "17": 1077,
        "18": 1077,
        "19": 1077,
        "20": 1077,
        "30": 1077,
        "50": 1077,
        "75": 1077,
        "100": 1077
    },
    'o&m_costs': 0.01  # 1% of investment costs every year
}

# Define the file path
json_file_path = "inputs_econ_base_2020.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(econ_inputs, json_file)

# %%
econ_inputs_SEG_SA = {
    'retail_high': 14.53/100,
    'retail_low': 14.53/100,
    'fit_high': 1/100,
    'fit_low': 1/100,
    'p2p_high': 10/100,
    'p2p_low': 10/100,
    'standing_costs': 20.39/100,
    'disc_rate': 0.03,
    'pv_prices': {
        "0": 1562,
        "1": 1562,
        "2": 1562,
        "3": 1562,
        "4": 1704,
        "5": 1704,
        "6": 1704,
        "7": 1704,
        "8": 1704,
        "9": 1704,
        "10": 1077,
        "11": 1077,
        "12": 1077,
        "13": 1077,
        "14": 1077,
        "15": 1077,
        "16": 1077,
        "17": 1077,
        "18": 1077,
        "19": 1077,
        "20": 1077,
        "30": 1077,
        "50": 1077,
        "75": 1077,
        "100": 1077
    },
    'o&m_costs': 0.01  # 1% of investment costs every year
}

# Define the file path
json_file_path = "inputs_econ_base_2020_SEG_SA_2.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(econ_inputs_SEG_SA, json_file)

# %%
econ_inputs_EC_SA = {
    'retail_high': 14.53/100,
    'retail_low': 14.53/100,
    'fit_high': 5.6/100,
    'fit_low': 5.6/100,
    'p2p_high': 12/100,
    'p2p_low': 12/100,
    'standing_costs': 20.39/100,
    'disc_rate': 0.03,
    'pv_prices': {
        "0": 1562,
        "1": 1562,
        "2": 1562,
        "3": 1562,
        "4": 1704,
        "5": 1704,
        "6": 1704,
        "7": 1704,
        "8": 1704,
        "9": 1704,
        "10": 1077,
        "11": 1077,
        "12": 1077,
        "13": 1077,
        "14": 1077,
        "15": 1077,
        "16": 1077,
        "17": 1077,
        "18": 1077,
        "19": 1077,
        "20": 1077,
        "30": 1077,
        "50": 1077,
        "75": 1077,
        "100": 1077
    },
    'o&m_costs': 0.01  # 1% of investment costs every year
}

# Define the file path
json_file_path = "inputs_econ_base_2020_EC_SA_2.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(econ_inputs_EC_SA, json_file)

# %%


econ_inputs_future = {
    'retail_high': 15.77/100,
    'retail_low': 15.77/100,
    'fit_high': 2.74/100,
    'fit_low': 2.74/100,
    'p2p_high': 9.25/100,
    'p2p_low': 9.25/100,
    'standing_costs': 20.39/100,
    'disc_rate': 0.03,
    'pv_prices': {
        "0": 985,
        "1": 985,
        "2": 985,
        "3": 985,
        "4": 1075,
        "5": 1075,
        "6": 1075,
        "7": 1075,
        "8": 1075,
        "9": 1075,
        "10": 680,
        "11": 680,
        "12": 680,
        "13": 680,
        "14": 680,
        "15": 680,
        "16": 680,
        "17": 680,
        "18": 680,
        "19": 680,
        "20": 680,
        "30": 680,
        "50": 680,
        "75": 680,
        "100": 680
    },
    'o&m_costs': 0.01  # 1% of investment costs every year
}

# Define the file path
json_file_path = "inputs_econ_future_2030.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(econ_inputs_future, json_file)


# %%
inputs = {
    'Comm_Sizes': [5],  # ,20,50,100,200,500]
    'PDR': [0.5, 1, 2],  # ,2,0.25],#,1.0,2.0],
    'pProsumer': [0.2, 0.4, 0.6, 0.8, 1],  # ,0.3,0.1],#]5,0.5,0.75,1.0],#,1.0]
    'reps': 50
}

# Define the file path
json_file_path = "inputs_N5.json"

# Store the dictionary in a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(inputs, json_file)
