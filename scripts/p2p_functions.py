# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:51:53 2020

@author: Prakhar Mehta

SUPPORTIVE FUNCTIONS FOR COMMUNITY POWER BALANCES
"""

# %%
import numpy as np
import pandas as pd
import datetime
from datetime import timezone
import pytz
import time
import copy

SimPar = {}
SimPar['SamplesPerYear'] = 10


class Community(object):
    '''
    Energy Community class
    '''

    def __init__(self, N, pdr, pProsumer, r_seed, hour_price, dem, pv):

        self.size = N
        self.pdr = pdr
        self.pProsumer = pProsumer
        self.hour_price = hour_price
        self.dem = dem
        self.pv = pv
        np.random.seed(r_seed)

        # the first N * pProsumers are set as prosumers
        # note that load profiles are sampled randomly for each instance of this class
        self.isProsumer = np.zeros(N)
        Nprosumers = int(N * pProsumer)
        self.isProsumer[:Nprosumers] = int(1)

        self.flist_selected = np.random.choice(self.dem.columns, N)  # randomly selected households

        # initializing variables to hold info of appropriate lengths
        self.Pdemand_resid = np.zeros(SimPar['SamplesPerYear'])
        self.Pprod_resid = np.zeros(SimPar['SamplesPerYear'])
        self.Community_AnnualConsumption = 0
        self.Pdemand_Community = np.zeros(SimPar['SamplesPerYear'])
        self.Pprod_Community = np.zeros(SimPar['SamplesPerYear'])
        self.SelfConsumed_Community = 0.0  # in kWh

        # create field to store/access simulation results
        self.SimRes = {}
        self.SimRes['NPV'] = {}
        self.SimRes['SCR'] = {}
        self.SimRes['SSR'] = {}

    def organise_demand_data(data):
        '''
        returns one complete year of demand data profile
        01 Jan 2010 - 31 Dec 2010
        '''
        utc = pytz.UTC
        return data[data.datetime_UTC > utc.localize(datetime.datetime(2009, 12, 31, 23, 30))]  # ,pytz.UTC)]#.astimezone(tz = utc)]

    def read_data(dem, pv, nrInFlist):
        '''
        reads in the selected demand profiles and PV profiles
        '''

        dem_dat = dem[nrInFlist]
        bldg_id = dem_dat.name
        sol_dat = pv[nrInFlist]
        return dem_dat, sol_dat, bldg_id

    def pdr(Psolar, dem, pdr):
        '''
        scales the solar pv output to match the pdr requested
        '''
        return Psolar*pdr*(sum(dem)/sum(Psolar))

    def individual(self):
        '''
        assigns simulation parameters to each household
        Calls individual energy flows
        '''

        eflows = {}
        loadprofiles = {}
        prosumerlist = []
        consumerlist = []
        for _k in range(self.size):
            nrInFlist = self.flist_selected[_k]

            Pdemand, Psolar, bldg_id = Community.read_data(self.dem, self.pv, nrInFlist)

            dem = Pdemand  # .demand.tolist()
            tempname = 'B' + str(bldg_id)
            deg_rate, PV_lifetime = 0.005, 30  # PV degradation 0.5%, lifetime 30 years
            Psolar = Community.pdr(Psolar, dem, self.pdr)  # sets the pdr

            # set solar to zero if consumer
            Psolar = Psolar if self.isProsumer[_k] == 1 else pd.Series(data=list(np.zeros(8760*2)))
            if self.isProsumer[_k] == 1:
                prosumerlist.append(tempname)
            if self.isProsumer[_k] == 0:
                consumerlist.append(tempname)

            eflows[tempname], loadprofiles[tempname] = Community.en_flows(dem, Psolar, deg_rate, PV_lifetime, self.hour_price)

        return eflows, loadprofiles, prosumerlist, consumerlist  # ,loadprofiles_p2p

    def en_flows(dem, solar, deg_rate, PV_lifetime, hour_price):
        '''
        calculates individual energy flows in each building
        '''
        solar_outputs = [solar * ((1 - deg_rate) ** y) for y in range(PV_lifetime)]
        load_profile = pd.DataFrame(data=None, index=range(8760*2))
        load_profile["demand"] = dem  # do them once not 30 times!
        # Define price of electricity per hour of the day
        load_profile["hour_price"] = hour_price
        load_profile_yrly_dict = {}

        # temp array for sc
        d = np.array(dem)

        # Create a dictionary to contain the annual energy balances
        load_profile_year = {}
        load_profile_year["SCR_t"] = 0
        load_profile_year["SSR_t"] = 0
        load_profile_year["SCR_h"] = 0
        load_profile_year["SSR_h"] = 0
        load_profile_year["SCR_l"] = 0
        load_profile_year["SSR_l"] = 0

        for yr in range(PV_lifetime):

            # Define hourly solar system output for this building and hourly demand
            load_profile["solar"] = solar_outputs[yr]

            # Compute hourly net demand from grid and hourly excess solar
            load_profile["net_demand"] = (load_profile.demand -
                                          load_profile.solar)
            load_profile["excess_solar"] = (load_profile.solar -
                                            load_profile.demand)

            # Remove negative values by making them zero
            load_profile["net_demand"] = np.array(
                [x if x > 0 else 0 for x in load_profile["net_demand"]])
            load_profile["excess_solar"] = np.array(
                [x if x > 0 else 0 for x in load_profile["excess_solar"]])

            # Compute hourly self-consumed electricity
            # For the hours of the year with solar generation: self-consume all
            # solar generation if less than demand (s) or up to demand (d)
            s = np.array(solar_outputs[yr])
            load_profile["sc"] = [min(s[i], d[i]) if s[i] > 0 else 0 for i in
                                  range(8760*2)]

            # Compute annual energy balances regardless of hour prices
            for bal in ["solar", "demand", "net_demand", "excess_solar", "sc"]:
                load_profile_year[bal] = sum(load_profile[bal])

            # Compute annual energy balances for high and low price hours
            for bal in ["solar", "demand", "excess_solar", "net_demand", "sc"]:
                for pl in ["high", "low"]:
                    cond = (load_profile["hour_price"] == pl)
                    load_profile_year[bal+'_'+pl] = sum(
                        load_profile[bal].loc[cond])

            # Compute year TOTAL self-consumption rate (SCR)
            # and year TOTAL self-sufficiency rate (SSR)
            if load_profile_year["sc"] > 0:
                load_profile_year["SCR_t"] = (load_profile_year["sc"] /
                                              load_profile_year["solar"])
                load_profile_year["SSR_t"] = (load_profile_year["sc"] /
                                              load_profile_year["demand"])

            # Compute year SCR and SSR for high hours
            if load_profile_year["sc_high"] > 0:
                load_profile_year["SCR_h"] = (load_profile_year["sc_high"] /
                                              load_profile_year["solar_high"])
                load_profile_year["SSR_h"] = (load_profile_year["sc_high"] /
                                              load_profile_year["demand_high"])

            # Compute year SCR and SSR for low hours
            if load_profile_year["sc_low"] > 0:
                load_profile_year["SCR_l"] = (load_profile_year["sc_low"] /
                                              load_profile_year["solar_low"])
                load_profile_year["SSR_l"] = (load_profile_year["sc_low"] /
                                              load_profile_year["demand_low"])

            # Store results for each year in return dataframe
            if yr == 0:
                # If it is the first year, then create the dataframe
                lifetime_load_profile = pd.DataFrame(load_profile_year,
                                                     index=[0])
            else:
                # Append the dictionary containing the results for this year
                lifetime_load_profile = lifetime_load_profile.append(
                    load_profile_year, ignore_index=True)

            load_profile_yrly_dict[yr] = load_profile

        return lifetime_load_profile, load_profile_yrly_dict

# %%
    def p2p(ind_profiles, tech_inputs):
        '''
        calls p2p calculations
        '''
        ind_profile_copy = copy.deepcopy(ind_profiles)
        ind_profiles_p2p, dict_solar_ratios = Community.en_flows_p2p(ind_profile_copy, tech_inputs)
        p2p_profiles_sum = Community.en_flows_p2p_sum(ind_profiles_p2p)

        return ind_profiles_p2p, p2p_profiles_sum, dict_solar_ratios

    def en_flows_p2p_sum(ind_profiles_p2p):
        '''
        calculates sum of values over the year to make financial calculations easier
        '''
        load_profile_year = {}
        sum_dict = {}

        for _i in ind_profiles_p2p:
            df_pros = pd.DataFrame(data=None, index=[0])
            for _j in range(len(ind_profiles_p2p[_i])):
                for bal in ["solar", "demand", "net_demand", "excess_solar",
                            "sc", "sold_excessto_comm", "sold_excessto_GRID",
                            "net_demand_p2p", "net_grid_flow",
                            "CURT_sold_excessto_GRID", "CURT_net_grid_flow"]:
                    load_profile_year[bal] = sum(ind_profiles_p2p[_i][_j][bal])

                for bal in ["solar", "demand", "net_demand", "excess_solar",
                            "sc", "sold_excessto_comm", "sold_excessto_GRID",
                            "net_demand_p2p", "net_grid_flow",
                            "CURT_sold_excessto_GRID", "CURT_net_grid_flow"]:
                    for pl in ["high", "low"]:
                        cond = (ind_profiles_p2p[_i][_j]["hour_price"] == pl)
                        load_profile_year[bal+'_'+pl] = sum(ind_profiles_p2p[_i][_j][bal].loc[cond])

                df_pros = df_pros.append(load_profile_year, ignore_index=True)
                df_pros = df_pros.dropna()

            sum_dict[_i] = df_pros
        return sum_dict

    def en_flows_p2p(ind_profile, tech_inputs):
        '''
        calculates p2p energy flows
        '''
        keys = list(ind_profile.keys())
        dict_solar_ratios = {}
        for _k in range(len(ind_profile[keys[0]])):  # over the lifetime of the PV panel
            # every year these aggregated profiles are made
            agg_profiles = pd.DataFrame(data=None, index=range(8760*2))
            agg_profiles['solar_comm_t'] = 0
            agg_profiles['demand_comm_t'] = 0
            agg_profiles['excess_solar_b4_p2p'] = 0
            agg_profiles['net_demand_b4_p2p'] = 0
            commlist = []

            for _j in ind_profile:  # taking every building into account
                commlist.append(_j)
                if sum(ind_profile[_j][_k].solar) > 0:
                    # prosumer
                    # total solar and demand in the community
                    agg_profiles['solar_comm_t'] += ind_profile[_j][_k]['solar']
                    agg_profiles['demand_comm_t'] += ind_profile[_j][_k]['demand']
                    # total excess solar in the community
                    agg_profiles['excess_solar_b4_p2p'] += ind_profile[_j][_k]['excess_solar']
                    # excess demand due to prosumers
                    agg_profiles['net_demand_b4_p2p'] += ind_profile[_j][_k]['net_demand']
                else:
                    # consumer
                    # total demand
                    agg_profiles['demand_comm_t'] += ind_profile[_j][_k]['demand']
                    # excess demand due to consumers
                    agg_profiles['net_demand_b4_p2p'] += ind_profile[_j][_k]['net_demand']

            # proportion of excess solar of each prosumer at every hour compared to total aggregated P2P excess solar
            # and
            # proportion of net demand of each community member at every hour compared to total aggregated P2P net demand
            excess_solar_ratios = pd.DataFrame(data=None, index=range(8760*2), columns=commlist)
            excess_demand_ratios = pd.DataFrame(data=None, index=range(8760*2), columns=commlist)
            for i in commlist:
                excess_solar_ratios[i] = np.where(agg_profiles['excess_solar_b4_p2p'] != 0, ind_profile[i]
                                                  [_k]['excess_solar']/agg_profiles['excess_solar_b4_p2p'], 0)
                excess_demand_ratios[i] = np.where(agg_profiles['net_demand_b4_p2p'] != 0, ind_profile[i]
                                                   [_k]['net_demand']/agg_profiles['net_demand_b4_p2p'], 0)

            dict_solar_ratios[_k] = excess_solar_ratios

            agg_profiles['net_demand_p2p'] = agg_profiles['net_demand_b4_p2p']-agg_profiles['excess_solar_b4_p2p']
            agg_profiles["net_demand_p2p"] = np.array(
                [x if x > 0 else 0 for x in agg_profiles["net_demand_p2p"]])

            # excess solar sold from prosumers to COMMUNITY
            s = agg_profiles['excess_solar_b4_p2p']
            d = agg_profiles["net_demand_b4_p2p"]
            agg_profiles['solar_p2p_sold_to_comm'] = [min(s[i], d[i]) if s[i] > 0 else 0 for i in range(8760*2)]

            # excess solar left after p2p to community - so this is left for feed in to the grid
            agg_profiles['excess_solar_p2p'] = agg_profiles['excess_solar_b4_p2p'] - \
                agg_profiles['solar_p2p_sold_to_comm']  # agg_cons_profiles['demand_b4_p2p']
            agg_profiles["excess_solar_p2p"] = np.array(
                [x if x > 0 else 0 for x in agg_profiles["excess_solar_p2p"]])

            # calculate demand peak here and calculate curtailed values of excess solar to the grid.
            dem_peak = max(agg_profiles['demand_comm_t'])
            xfmer_size_temp = dem_peak*tech_inputs['xfmer_size_factor']/tech_inputs['pf']
            xfmer_size = next(x for x in tech_inputs['xfmer_SP_size_list'] if x >= xfmer_size_temp)
            agg_profiles['curt_exc_solar_to_GRID'] = np.where(
                agg_profiles['excess_solar_p2p'] > xfmer_size, xfmer_size, agg_profiles['excess_solar_p2p'])

            # parts of excess solar of a bldg sold to p2p cons/p2p pros/net demands etc...
            for _j in ind_profile:
                ind_profile[_j][_k]['sold_excessto_comm'] = excess_solar_ratios[_j]*agg_profiles['solar_p2p_sold_to_comm']
                ind_profile[_j][_k]['sold_excessto_GRID'] = excess_solar_ratios[_j]*agg_profiles['excess_solar_p2p']
                ind_profile[_j][_k]['net_demand_p2p'] = excess_demand_ratios[_j]*agg_profiles['net_demand_p2p']
                ind_profile[_j][_k]['net_grid_flow'] = ind_profile[_j][_k]['net_demand_p2p'] - ind_profile[_j][_k]['sold_excessto_GRID']
                ind_profile[_j][_k]['CURT_sold_excessto_GRID'] = excess_solar_ratios[_j]*agg_profiles['curt_exc_solar_to_GRID']
                ind_profile[_j][_k]['CURT_net_grid_flow'] = ind_profile[_j][_k]['net_demand_p2p'] - \
                    ind_profile[_j][_k]['CURT_sold_excessto_GRID']
                ind_profile[_j][_k]['CURT_needed'] = np.where(
                    ind_profile[_j][_k]['CURT_sold_excessto_GRID'] != ind_profile[_j][_k]['sold_excessto_GRID'], 1, 0)

        return ind_profile, dict_solar_ratios

    def en_flows_community(profiles_ind, profiles_p2p, ind_prof_sum, p2p_prof_sum, tech_inputs):
        '''
        Returns
        -------
        comm_no_p2p: DICT
            aggregate individual profiles without trade

        comm_p2p: DICT
            aggregate individual profiles with trade

        tempdict: DICT
            yearly profiles aggregated on the half-hourly level - without trade

        tempdict2: DICT
            yearly profiles aggregated on the half-hourly level - with trade

        '''
        # aggregate individual profiles and calculate scr, ssr
        comm_no_p2p = pd.DataFrame(data=0, index=ind_prof_sum[list((ind_prof_sum).keys())[0]].index,
                                   columns=ind_prof_sum[list((ind_prof_sum).keys())[0]].columns)
        for _i in ind_prof_sum.keys():
            comm_no_p2p += ind_prof_sum[_i]

        # aggregate p2p profiles and then calculate scr, ssr
        comm_p2p = pd.DataFrame(data=0, index=p2p_prof_sum[list((p2p_prof_sum).keys())[0]].index,
                                columns=p2p_prof_sum[list((p2p_prof_sum).keys())[0]].columns)
        for _i in p2p_prof_sum.keys():
            comm_p2p += p2p_prof_sum[_i]

        comm_no_p2p, comm_p2p = Community.comm_tech_kpis(comm_no_p2p, comm_p2p)

        # yearly profiles aggregated on the half-hourly level
        keys = list(profiles_ind.keys())
        tempdict = {}

        for _j in range(len(profiles_ind[keys[0]])):
            df = pd.DataFrame(data=0, index=profiles_ind[list(profiles_ind.keys())[0]][0].index,
                              columns=profiles_ind[list(profiles_ind.keys())[0]][0].columns)
            for _i in profiles_ind:
                hour_price = profiles_ind[_i][_j]['hour_price']
                profiles_ind[_i][_j] = profiles_ind[_i][_j].drop('hour_price', axis=1)
                df += profiles_ind[_i][_j]
            df['hour_price'] = hour_price
            df['net_grid_flow'] = df.net_demand - df.excess_solar
            dem_peak = max(df['demand'])
            xfmer_size_temp = dem_peak*tech_inputs['xfmer_size_factor']/tech_inputs['pf']
            xfmer_size = next(x for x in tech_inputs['xfmer_SP_size_list'] if x >= xfmer_size_temp)
            df['CURT_excess_solar'] = np.where(df['excess_solar'] > xfmer_size, xfmer_size, df['excess_solar'])
            df['CURT_net_grid_flow'] = df.net_demand - df.CURT_excess_solar
            df['CURT_needed'] = np.where(df['CURT_excess_solar'] != df['excess_solar'], 1, 0)
            tempdict[_j] = df

        keys = list(profiles_p2p.keys())
        tempdict2 = {}
        for _j in range(len(profiles_p2p[keys[0]])):
            df = pd.DataFrame(data=0, index=profiles_p2p[list(profiles_p2p.keys())[0]][0].index,
                              columns=profiles_p2p[list(profiles_p2p.keys())[0]][0].columns)
            for _i in profiles_p2p:
                hour_price = profiles_p2p[_i][_j]['hour_price']
                profiles_p2p[_i][_j] = profiles_p2p[_i][_j].drop('hour_price', axis=1)
                df += profiles_p2p[_i][_j]
            df['hour_price'] = hour_price
            df['net_grid_flow'] = df.net_demand_p2p - df.sold_excessto_GRID
            tempdict2[_j] = df

        return comm_no_p2p, comm_p2p, tempdict, tempdict2

    def comm_tech_kpis(comm_no_p2p, comm_p2p):
        '''
        Returns
        -------
        comm_no_p2p:
            calculated SCR, SSR
        comm_p2p: 
            calculated SCR, SSR

        '''
        comm_no_p2p['SCR_t'] = comm_no_p2p.sc/comm_no_p2p.solar
        comm_no_p2p['SCR_h'] = comm_no_p2p.sc_high/comm_no_p2p.solar_high
        comm_no_p2p['SCR_l'] = comm_no_p2p.sc_low/comm_no_p2p.solar_low
        comm_no_p2p['SSR_t'] = comm_no_p2p.sc/comm_no_p2p.demand
        comm_no_p2p['SSR_h'] = comm_no_p2p.sc_high/comm_no_p2p.demand_high
        comm_no_p2p['SSR_l'] = comm_no_p2p.sc_low/comm_no_p2p.demand_low

        comm_p2p['SCR_t'] = (comm_p2p.sc + comm_p2p.sold_excessto_comm)/comm_p2p.solar
        comm_p2p['SCR_h'] = (comm_p2p.sc_high + comm_p2p.sold_excessto_comm_high)/comm_p2p.solar_high
        comm_p2p['SCR_l'] = (comm_p2p.sc_low + comm_p2p.sold_excessto_comm_low)/comm_p2p.solar_low
        comm_p2p['SSR_t'] = (comm_p2p.sc + comm_p2p.sold_excessto_comm)/comm_p2p.demand
        comm_p2p['SSR_h'] = (comm_p2p.sc_high + comm_p2p.sold_excessto_comm_high)/comm_p2p.demand_high
        comm_p2p['SSR_l'] = (comm_p2p.sc_low + comm_p2p.sold_excessto_comm_low)/comm_p2p.demand_low

        return comm_no_p2p, comm_p2p

    def curtailment(ind_profile, profiles_comm_ind_agg, tech_inputs,
                    dict_solar_ratios, hour_price):
        '''
        Returns
        -------
        ind_profile: 
            DATAFRAME individual hourly profiles after curtailment
        ind_profile_sum:
            DATAFRAME yearly sums of individual hourly profiles after curtailment
        comm_no_p2p: 
            DATAFRAME community level SCR and SSR in the absence of sharing
        '''

        dem_peak = max(profiles_comm_ind_agg[0].demand)
        xfmer_size_temp = dem_peak*tech_inputs['xfmer_size_factor']/tech_inputs['pf']
        xfmer_size = next(x for x in tech_inputs['xfmer_SP_size_list'] if x >= xfmer_size_temp)
        for _k in ind_profile.keys():  # taking every building into account
            for _j in range(len(ind_profile[_k])):  # over the lifetime of the PV panel
                ind_profile[_k][_j]['CURT_excess_solar'] = profiles_comm_ind_agg[_j]['CURT_excess_solar']*dict_solar_ratios[_j][_k]

        ind_profile_sum = Community.curtailment_sum(ind_profile, hour_price)

        comm_no_p2p = pd.DataFrame(data=0, index=ind_profile_sum[list((ind_profile_sum).keys())[0]].index,
                                   columns=ind_profile_sum[list((ind_profile_sum).keys())[0]].columns)
        for _i in ind_profile_sum.keys():
            comm_no_p2p += ind_profile_sum[_i]

        comm_no_p2p['SCR_t'] = comm_no_p2p.sc/comm_no_p2p.solar
        comm_no_p2p['SCR_h'] = comm_no_p2p.sc_high/comm_no_p2p.solar_high
        comm_no_p2p['SCR_l'] = comm_no_p2p.sc_low/comm_no_p2p.solar_low
        comm_no_p2p['SSR_t'] = comm_no_p2p.sc/comm_no_p2p.demand
        comm_no_p2p['SSR_h'] = comm_no_p2p.sc_high/comm_no_p2p.demand_high
        comm_no_p2p['SSR_l'] = comm_no_p2p.sc_low/comm_no_p2p.demand_low

        return ind_profile, ind_profile_sum, comm_no_p2p

    def curtailment_sum(ind_profile, hour_price):
        '''
        Returns
        -------
        sum_dict:
            post curtailment yearly totals of "solar", "demand", "net_demand", "excess_solar",
                        "sc", "CURT_excess_solar"

        '''
        load_profile_year = {}
        sum_dict = {}

        for _i in ind_profile:
            df_pros = pd.DataFrame(data=None, index=[0])
            for _j in range(len(ind_profile[_i])):
                for bal in ["solar", "demand", "net_demand", "excess_solar",
                            "sc", "CURT_excess_solar"]:
                    load_profile_year[bal] = sum(ind_profile[_i][_j][bal])

                for bal in ["solar", "demand", "net_demand", "excess_solar",
                            "sc", "CURT_excess_solar"]:
                    for pl in ["high", "low"]:
                        cond = (ind_profile[_i][_j]["hour_price"] == pl)
                        load_profile_year[bal+'_'+pl] = sum(ind_profile[_i][_j][bal].loc[cond])
                df_pros = df_pros.append(load_profile_year, ignore_index=True)
                df_pros = df_pros.dropna()
            sum_dict[_i] = df_pros
        return sum_dict
