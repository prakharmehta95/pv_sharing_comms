# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:48:52 2020

@author: Prakhar Mehta

MAIN SCRIPT TO PERFORM THE COMMUNITY POWER BALANCE SIMULATIONS
"""
# %%
import p2p_npv_functions
import p2p_functions
import os
import time
import numpy_financial as npf
import copy
import datetime
import glob
import random
import json
import numpy as np
import pandas as pd

# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)

parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, 'data')
results_dir = os.path.join(parent_dir, 'results')

# %%
# import other required modules

# %%

# reading data inputs


def hourlyprices(start, end):
    '''
    generates hourly prices 
    '''
    from datetime import timedelta
    delta = timedelta(hours=0.5)
    hour_price = []
    while start <= end:
        if 5 < start.hour < 22 and start.weekday() != 6:
            hour_price.append('high')
        else:
            hour_price.append('low')
        start += delta
    return hour_price


start = datetime.datetime(2010, 1, 1, 0, 0)
end = datetime.datetime(2010, 12, 31, 23, 30)
hour_price = hourlyprices(start, end)


def get_pv_cap(prosumerlist, ind_prof_sum):
    '''
    calculate and return the pv capacity
    '''
    pv_sizes_260W_multiples = [x*0.260 for x in range(50000)]
    pv_cap = {}
    for _i in prosumerlist:
        pv = round(ind_prof_sum[_i].at[0, 'solar']/924, 2)  # average solar panel produces 924 kWh/Kwp in London Nigel B. Mason IET 2016
        pv_cap[_i] = min(pv_sizes_260W_multiples, key=lambda x: abs(x-pv))
        # pv_cap[_i] = next(x for x in pv_sizes_260W_multiples if x > pv) #always larger system is chosen, unit = kW
    return pv_cap


def load_econ_data(path, scenario):
    '''
    to load the econ data based on sensitivity analysis
    '''
    if scenario != "future":
        # Construct the file path based on the scenario number
        filename = f"inputs_econ_base_2020{scenario}.json"
        json_file_path = os.path.join(path, filename)
    else:
        # Construct the file path based on the scenario number
        filename = "inputs_econ_future_2030.json"
        json_file_path = os.path.join(path, filename)

    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


# reading economic data inputs
def load_simulation_data(path, N_sim):
    '''
    to load the econ data based on sensitivity analysis
    '''
    # Construct the file path based on the scenario number
    filename = f"inputs_N{N_sim}.json"
    json_file_path = os.path.join(path, filename)

    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


# %%
'''
Load input data
'''

# read demand data
demanddata_filename = 'london_cleaned.pickle'
demand = pd.read_pickle(os.path.join(data_dir, demanddata_filename))
demand = demand.drop('ID', axis=1)

# read PV data
pvdata_filename = 'PV_Gen_UK_20200815.pickle'
pv = pd.read_pickle(os.path.join(data_dir, pvdata_filename))


# reading xmer sizing inputs
techinputs_filename = 'inputs_tech_xmer.json'
tech_inputs_path = os.path.join(data_dir, techinputs_filename)
with open(tech_inputs_path, "r") as f:
    tech_inputs = json.load(f)


# choose 'sensitivity_analyses_name' based on if a sensitivity analysis is to be conducted. The economic input data is loaded accordingly
sensitivity_analyses_name = ""  # OPTIONS = "" , "_EC_SA_1" #, "_EC_SA_2" , "_SEG_SA_1" ,"_SEG_SA_2", "future"
econ_inputs = load_econ_data(data_dir, sensitivity_analyses_name)


# varying energy community configurations
N_sim = 5  # Community Size OPTIONS = 5,10,20,40,100
inputs = load_simulation_data(data_dir, N_sim)


# %%
'''
data types to hold intermediate and final results
'''

pv_categs = [int(x) for x in list(econ_inputs['pv_prices'].keys())]
ind_dem = {}
ind_pros_npv = {}
p2p_pros_npv = {}
ind_cons_bill = {}
p2p_cons_bill = {}

# each hh yearly profile
profiles_ind = {}
curt_profiles_ind = {}

# each hh yearly profile with sharing within the community (community represented by 'p2p')
profiles_p2p = {}
profiles_comm_ind_agg = {}
profiles_comm_p2p_agg = {}
prof_yr_sum_ind = {}
prof_yr_sum_p2p = {}
prof_yr_sum_comm_ind = {}
prof_yr_sum_comm_p2p = {}

dict_solar_ratios = {}

cons_p2p = {}
pros_p2p = {}
cf_ind_pros = {}
cf_p2p_pros = {}
comm_costs = {}
costs_no_pv = {}
costs_with_pv = {}
costs_with_pv_with_p2p = {}

comm_data = {}

ind_pros_npv_curt = {}
p2p_pros_npv_curt = {}
cf_ind_pros_curt = {}
cf_p2p_pros_curt = {}


# storing results - summary statistics
summ_stats = {
    'pros_ind_npv': dict.fromkeys(inputs['pProsumer']),
    'pros_p2p_npv': dict.fromkeys(inputs['pProsumer']),
    'pros_ind_npv_curt': dict.fromkeys(inputs['pProsumer']),
    'pros_p2p_npv_curt': dict.fromkeys(inputs['pProsumer']),
    'cons_ind_bill': dict.fromkeys(inputs['pProsumer']),
    'cons_p2p_bill': dict.fromkeys(inputs['pProsumer'])
}

# storing results - cost information
cost_stats = {
    'costs_no_pv': dict.fromkeys(inputs['pProsumer']),
    'costs_with_pv': dict.fromkeys(inputs['pProsumer']),
    'costs_with_pv_with_p2p': dict.fromkeys(inputs['pProsumer'])
}

# %%
'''
Energy community simulation code snippet here
'''

# SIMULATING ENERGY COMMUNITIES
for _i in inputs['Comm_Sizes']:  # for each community size
    N = _i
    for _j in inputs['PDR']:  # for each production-to-demand ratio
        for _k in inputs['pProsumer']:  # for each prosumer ratio
            #  more data types to hold intermediate results
            sims = []
            list_ind_pros_npv = []
            list_p2p_pros_npv = []
            list_ind_pros_npv_curt = []
            list_p2p_pros_npv_curt = []

            list_ind_cons_bill = []
            list_p2p_cons_bill = []
            list_costs_no_pv = []
            list_costs_with_pv = []
            list_costs_with_pv_with_p2p = []
            comm_data[_k] = {}

            seedlist = list(range(inputs['reps']))
            start1 = time.time()
            for _n in range(inputs['reps']):
                print('PDR = ', _j)
                print('pProsumer = ', _k)
                print('**rep = ', _n, '**')

                start_rep = time.time()

                sims.append(p2p_functions.Community(N, _j, _k, seedlist[_n],
                                                    hour_price, demand, pv))
                # half-hourly profiles - INDIVIDUAL
                prof_yr_sum_ind[_k], profiles_ind[_k], prosumerlist, consumerlist = sims[_n].individual()

                # installed PV capacity
                pv_cap = get_pv_cap(prosumerlist, prof_yr_sum_ind[_k])

                # P2P profiles
                profiles_p2p[_k], prof_yr_sum_p2p[_k], dict_solar_ratios[_k] = p2p_functions.Community.p2p(
                    profiles_ind[_k], tech_inputs)

                ind_profile_copy = copy.deepcopy(profiles_ind[_k])

                # aggregated profiles yearly
                prof_yr_sum_comm_ind[_k], prof_yr_sum_comm_p2p[_k], profiles_comm_ind_agg[_k], profiles_comm_p2p_agg[_k] = p2p_functions.Community.en_flows_community(
                    ind_profile_copy, profiles_p2p[_k], prof_yr_sum_ind[_k], prof_yr_sum_p2p[_k], tech_inputs)

                # curtailment for individual profiles
                profiles_ind[_k], prof_yr_sum_ind[_k], prof_yr_sum_comm_ind[_k] = p2p_functions.Community.curtailment(profiles_ind[_k],
                                                                                                                      profiles_comm_ind_agg[_k],
                                                                                                                      tech_inputs,
                                                                                                                      dict_solar_ratios[_k],
                                                                                                                      hour_price)

                # npv calculations for individual and community prosumers
                ind_pros_npv[_k], p2p_pros_npv[_k], cf_ind_pros[_k], cf_p2p_pros[_k], ind_pros_npv_curt[_k], p2p_pros_npv_curt[_k], cf_ind_pros_curt[_k], cf_p2p_pros_curt[_k] = p2p_more_functions.ind_pros_npv(
                    prof_yr_sum_ind[_k], prof_yr_sum_p2p[_k],
                    pv_cap, prosumerlist,
                    econ_inputs, pv_categs)

                # costs for consumers
                ind_cons_bill[_k], p2p_cons_bill[_k] = p2p_more_functions.ind_cons_bill(
                    prof_yr_sum_ind[_k], prof_yr_sum_p2p[_k], consumerlist, econ_inputs)

                # costs for prosumers
                costs_no_pv[_k], costs_with_pv[_k], costs_with_pv_with_p2p[_k] = p2p_more_functions.calc_costs(
                    prof_yr_sum_ind[_k], prof_yr_sum_p2p[_k], econ_inputs, pv_categs)

                # ---------------------
                # DATA COLLECTION

                # NPV and cost data
                list_ind_pros_npv.append(list(ind_pros_npv[_k].values()))
                list_p2p_pros_npv.append(list(p2p_pros_npv[_k].values()))
                list_ind_cons_bill.append(list(ind_cons_bill[_k].values()))
                list_p2p_cons_bill.append(list(p2p_cons_bill[_k].values()))

                # curtailed values of npvs
                list_ind_pros_npv_curt.append(list(ind_pros_npv_curt[_k].values()))
                list_p2p_pros_npv_curt.append(list(p2p_pros_npv_curt[_k].values()))

                list_costs_no_pv.append(list(costs_no_pv[_k].values()))
                list_costs_with_pv.append(list(costs_with_pv[_k].values()))
                list_costs_with_pv_with_p2p.append(list(costs_with_pv_with_p2p[_k].values()))

                # communuity level need data collection for every rep here
                # only first year out of the 30 year lifetime is considered
                # because the demand stays the same, and only solar deteriorates
                # by 0.6% every year
                temp_df = pd.DataFrame(data=None, index=['demand_kWh', 'demand_peak_kWp',
                                                         'solar_kWh', 'solar_peak_kWp', 'self_consumption_kWh',
                                                         'grid_import_max_kWp',
                                                         'grid_export_max_kWp', 'grid_export_total_kWh',
                                                         'grid_import_total_kWh', 'scr', 'ssr', 'npv_pros',
                                                         'npv_pros_curt'],
                                       columns=['no_trade', 'w_trade'])
                temp_df.at['demand_kWh', 'no_trade'] = prof_yr_sum_comm_ind[_k]['demand'][0]
                temp_df.at['solar_kWh', 'no_trade'] = prof_yr_sum_comm_ind[_k]['solar'][0]
                temp_df.at['demand_kWh', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['demand'][0]
                temp_df.at['solar_kWh', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['solar'][0]
                temp_df.at['demand_peak_kWp', 'no_trade'] = max(profiles_comm_ind_agg[_k][0]['demand'])
                temp_df.at['demand_peak_kWp', 'w_trade'] = max(profiles_comm_p2p_agg[_k][0]['demand'])
                temp_df.at['solar_peak_kWp', 'no_trade'] = max(profiles_comm_ind_agg[_k][0]['solar'])
                temp_df.at['solar_peak_kWp', 'w_trade'] = max(profiles_comm_p2p_agg[_k][0]['solar'])
                temp_df.at['self_consumption_kWh', 'no_trade'] = prof_yr_sum_comm_ind[_k]['sc'][0]
                temp_df.at['self_consumption_kWh', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['sc'][0] + \
                    prof_yr_sum_comm_p2p[_k]['sold_excessto_comm'][0]

                temp_df.at['curtailed_PV_kWh', 'no_trade'] = prof_yr_sum_comm_ind[_k]['excess_solar'][0] - \
                    prof_yr_sum_comm_ind[_k]['CURT_excess_solar'][0]
                temp_df.at['curtailed_PV_kWh', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['sold_excessto_GRID'][0] - \
                    prof_yr_sum_comm_p2p[_k]['CURT_sold_excessto_GRID'][0]

                temp_df.at['curtailed_PV_total_hours', 'no_trade'] = sum(profiles_comm_ind_agg[_k][0].CURT_needed)
                temp_df.at['curtailed_PV_total_hours', 'w_trade'] = sum(
                    profiles_comm_p2p_agg[_k][0].CURT_needed)/N  # because this column gets added again and again

                temp_df.at['curtailed_PV_list_hours', 'no_trade'] = np.where(
                    profiles_comm_ind_agg[_k][0].CURT_needed != 0, profiles_comm_ind_agg[_k][0].index, 0)
                temp_df.at['curtailed_PV_list_hours', 'w_trade'] = np.where(
                    profiles_comm_p2p_agg[_k][0].CURT_needed != 0, profiles_comm_p2p_agg[_k][0].index, 0)

                temp_df.at['sold_to_comm_kWh', 'no_trade'] = 0
                temp_df.at['sold_to_comm_kWh', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['sold_excessto_comm'][0]

                temp_df.at['pv_cap_kWh', 'no_trade'] = round(sum(pv_cap.values()))
                temp_df.at['pv_cap_kWh', 'w_trade'] = round(sum(pv_cap.values()))

                # for 'no trade' we must account for net_grid flow as there is still residual PV left in the community

                # maximum power imported/exported by the community over the year
                temp_df.at['grid_import_max_kWp', 'no_trade'] = max(profiles_comm_ind_agg[_k][0]['net_grid_flow'])
                temp_df.at['grid_export_max_kWp', 'no_trade'] = abs(
                    min(profiles_comm_ind_agg[_k][0]['net_grid_flow']))  # as it could be negative
                temp_df.at['grid_import_max_kWp', 'w_trade'] = max(profiles_comm_p2p_agg[_k][0]['net_grid_flow'])
                temp_df.at['grid_export_max_kWp', 'w_trade'] = abs(min(profiles_comm_p2p_agg[_k][0]['net_grid_flow']))

                # total energy imported/exported by community over the year
                temp_df.at['grid_import_total_kWh', 'no_trade'] = sum(profiles_comm_ind_agg[_k][0]['net_grid_flow'] >= 0)
                temp_df.at['grid_export_total_kWh', 'no_trade'] = abs(sum(profiles_comm_ind_agg[_k][0]['net_grid_flow'] < 0))
                temp_df.at['grid_import_total_kWh', 'w_trade'] = sum(profiles_comm_p2p_agg[_k][0]['net_grid_flow'] >= 0)
                temp_df.at['grid_export_total_kWh', 'w_trade'] = abs(sum(profiles_comm_p2p_agg[_k][0]['net_grid_flow'] < 0))

                # self-consumption 'scr' and self-sufficiency 'ssr' ratios
                temp_df.at['scr', 'no_trade'] = prof_yr_sum_comm_ind[_k]['SCR_t'][0]
                temp_df.at['ssr', 'no_trade'] = prof_yr_sum_comm_ind[_k]['SSR_t'][0]
                temp_df.at['scr', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['SCR_t'][0]
                temp_df.at['ssr', 'w_trade'] = prof_yr_sum_comm_p2p[_k]['SSR_t'][0]

                # NPV of the prosumers with/without community and with/without curtailment
                temp_df.at['npv_pros', 'no_trade'] = list_ind_pros_npv[_n]
                temp_df.at['npv_pros', 'w_trade'] = list_p2p_pros_npv[_n]
                temp_df.at['npv_pros_curt', 'no_trade'] = list_ind_pros_npv_curt[_n]
                temp_df.at['npv_pros_curt', 'w_trade'] = list_p2p_pros_npv_curt[_n]

                # Absolute numbers and the fractions of NPV positive prosumers with/without community
                temp_df.at['positive_npv_pros', 'no_trade'] = len([x for x in list_ind_pros_npv[_n] if x > 0])
                temp_df.at['positive_npv_pros', 'w_trade'] = len([x for x in list_p2p_pros_npv[_n] if x > 0])
                temp_df.at['frn_positive_npv_pros', 'no_trade'] = len(
                    [x for x in list_ind_pros_npv[_n] if x > 0])/len(list_ind_pros_npv[_n])
                temp_df.at['frn_positive_npv_pros', 'w_trade'] = len([x for x in list_p2p_pros_npv[_n] if x > 0])/len(list_p2p_pros_npv[_n])

                # Absolute numbers and the fractions of NPV positive prosumers with/without community with curtailment of PV
                temp_df.at['positive_npv_pros_curt', 'no_trade'] = len([x for x in list_ind_pros_npv_curt[_n] if x > 0])
                temp_df.at['positive_npv_pros_curt', 'w_trade'] = len([x for x in list_p2p_pros_npv_curt[_n] if x > 0])
                temp_df.at['frn_positive_npv_pros_curt', 'no_trade'] = len(
                    [x for x in list_ind_pros_npv_curt[_n] if x > 0])/len(list_ind_pros_npv_curt[_n])
                temp_df.at['frn_positive_npv_pros_curt', 'w_trade'] = len(
                    [x for x in list_p2p_pros_npv_curt[_n] if x > 0])/len(list_p2p_pros_npv_curt[_n])

                # storing results for each rep of each prosumer ratio.
                comm_data[_k][_n] = temp_df

                end_rep = time.time()
                print('time taken per run = ', end_rep - start_rep)

            # main results
            summ_stats['pros_ind_npv'][_k] = list_ind_pros_npv
            summ_stats['pros_p2p_npv'][_k] = list_p2p_pros_npv
            summ_stats['pros_ind_npv_curt'][_k] = list_ind_pros_npv_curt
            summ_stats['pros_p2p_npv_curt'][_k] = list_p2p_pros_npv_curt
            summ_stats['cons_ind_bill'][_k] = list_ind_cons_bill
            summ_stats['cons_p2p_bill'][_k] = list_p2p_cons_bill
            cost_stats['costs_no_pv'][_k] = list_costs_no_pv
            cost_stats['costs_with_pv'][_k] = list_costs_with_pv
            cost_stats['costs_with_pv_with_p2p'][_k] = list_costs_with_pv_with_p2p

            end = time.time()
            print('time taken ============ ', end - start1)

        # save results
        np.save(results_dir + '\2021_08_03_UK_cost_stats_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', cost_stats)
        np.save(results_dir + '\2021_08_03_UK_comm_data_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', comm_data)
        np.save(results_dir + '\2021_08_03_UK_summ_stats_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', summ_stats)

        np.save(results_dir + '\2021_08_03_UK_profiles_comm_yrly_no_trade_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', profiles_comm_ind_agg)
        np.save(results_dir + '\2021_08_03_UK_profiles_comm_yrly_w_trade_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', profiles_comm_p2p_agg)
        np.save(results_dir + '\2021_08_03_UK_profiles_comm_yr_SUM_no_trade_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', prof_yr_sum_comm_ind)
        np.save(results_dir + '\2021_08_03_UK_profiles_comm_yr_SUM_w_trade_N' + str(N_sim) + '_PDR' +
                str(_j) + '_pR1_3_5_7_10_reps50.npy', prof_yr_sum_comm_p2p)
