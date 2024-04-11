# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:05:54 2020

@author: Prakhar Mehta

SUPPORTIVE FUNCTIONS FOR CALCULATING THE NET PRESENT VALUES
"""

# %%
import pandas as pd
import numpy_financial as npf


def ind_cons_bill(ind_prof_sum, p2p_profiles_sum, consumerlist, econ_inputs):
    '''
    calculates electricity bill for individual consumers
    '''
    cons_bill_ind = {}
    cons_bill_p2p = {}
    for _i in consumerlist:
        cons_bill_ind[_i] = (sum(ind_prof_sum[_i].net_demand_high)*econ_inputs['retail_high'] +
                             sum(ind_prof_sum[_i].net_demand_low)*econ_inputs['retail_low'])
        cons_bill_p2p[_i] = ((sum(p2p_profiles_sum[_i].net_demand_high)-sum(p2p_profiles_sum[_i].net_demand_p2p_high))*econ_inputs['p2p_high'] +
                             (sum(p2p_profiles_sum[_i].net_demand_low)-sum(p2p_profiles_sum[_i].net_demand_p2p_low))*econ_inputs['p2p_low'] +
                             sum(p2p_profiles_sum[_i].net_demand_p2p_high)*econ_inputs['retail_high'] +
                             sum(p2p_profiles_sum[_i].net_demand_p2p_low)*econ_inputs['retail_low'])
    return cons_bill_ind, cons_bill_p2p


def ind_pros_npv(ind_prof_sum, p2p_profiles_sum, pv_cap, prosumerlist, econ_inputs, pv_categs):
    '''
    calculates the npv for individual prosumers and for prosumers in a p2p 
    community 
    '''
    # no curtailment
    npv_ind_dict = {}
    npv_p2p_dict = {}
    cf_ind_dict = {}
    cf_p2p_dict = {}

    # curtailed values
    npv_ind_dict_curt = {}
    npv_p2p_dict_curt = {}
    cf_ind_dict_curt = {}
    cf_p2p_dict_curt = {}

    for _i in prosumerlist:
        # gets the price category. takes the lower value.
        # example: if size = 9, then price per kW taken for 5 kWp PV panel size

        # manually setting PV system size to at least 0.5kWp
        if 0 <= pv_cap[_i] <= 0.5:
            pv_cap[_i] = 0.5

        pvp_ix = int(pv_cap[_i])
        if 20 < pvp_ix < 30:
            pvp_ix = 20
        elif 30 < pvp_ix < 50:
            pvp_ix = 30
        elif 50 < pvp_ix < 75:
            pvp_ix = 50
        elif 75 < pvp_ix < 100:
            pvp_ix = 75
        elif 100 < pvp_ix < 125:
            pvp_ix = 100
        elif 125 < pvp_ix < 150:
            pvp_ix = 125
        elif pvp_ix > 150:
            pvp_ix = 150

        inv = pv_cap[_i]*econ_inputs['pv_prices'][str(pvp_ix)]
        cf_ind = [-inv]
        cf_p2p = [-inv]
        cf_ind_curt = [-inv]
        cf_p2p_curt = [-inv]

        # cash flows in individual and p2p cases
        ind_cf, p2p_cf = calc_cash_flows(ind_prof_sum[_i], p2p_profiles_sum[_i], inv, econ_inputs)
        cf_ind.extend(ind_cf.net_cf.tolist())
        cf_p2p.extend(p2p_cf.net_cf.tolist())
        cf_ind_curt.extend(ind_cf.net_cf_curt.tolist())
        cf_p2p_curt.extend(p2p_cf.net_cf_curt.tolist())
        cf_ind_dict[_i] = cf_ind
        cf_p2p_dict[_i] = cf_p2p
        npv_ind_dict[_i] = npf.npv(econ_inputs['disc_rate'], cf_ind)
        npv_p2p_dict[_i] = npf.npv(econ_inputs['disc_rate'], cf_p2p)
        cf_ind_dict_curt[_i] = cf_ind_curt
        cf_p2p_dict_curt[_i] = cf_p2p_curt
        npv_ind_dict_curt[_i] = npf.npv(econ_inputs['disc_rate'], cf_ind_curt)
        npv_p2p_dict_curt[_i] = npf.npv(econ_inputs['disc_rate'], cf_p2p_curt)

    return npv_ind_dict, npv_p2p_dict, cf_ind_dict, cf_p2p_dict, npv_ind_dict_curt, npv_p2p_dict_curt, cf_ind_dict_curt, cf_p2p_dict_curt


def calc_cash_flows(df, df_p2p, inv, econ_inputs):
    '''
    calculates the cash flows every year
    '''
    cashflows_year_ind = {}
    cashflows_year_p2p = {}
    cashflows_year_ind_curt = {}
    cashflows_year_p2p_curt = {}

    for yr in range(len(df.index)):  # 25 years as that is the PV lifetime considered

        # Read year excess solar
        ex_solar_h = df["excess_solar_high"][yr]
        ex_solar_l = df["excess_solar_low"][yr]

        ex_solar_h_curt = df["CURT_excess_solar_high"][yr]
        ex_solar_l_curt = df["CURT_excess_solar_low"][yr]

        # Compute the revenues from feeding solar electricity to the grid
        cashflows_year_ind["FIT"] = ex_solar_h * econ_inputs["fit_high"] +\
            ex_solar_l * econ_inputs["fit_low"]

        cashflows_year_ind["FIT_curt"] = ex_solar_h_curt * econ_inputs["fit_high"] +\
            ex_solar_l_curt * econ_inputs["fit_low"]

        # Read avoided consumption from the grid (i.e. self-consumption)
        sc_h = df["sc_high"][yr]
        sc_l = df["sc_low"][yr]

        # Compute the savings from self-consuming solar electricity
        cashflows_year_ind["savings"] = sc_h * econ_inputs['retail_high'] +\
            sc_l * econ_inputs['retail_low']

        # Compute O&M costs
        cashflows_year_ind["O&M"] = inv * econ_inputs["o&m_costs"]

        # standing costs in the UK
        cashflows_year_ind["Standing_Costs"] = 365 * econ_inputs["standing_costs"]

        # Compute net cashflows to the agent
        cashflows_year_ind["net_cf"] = (cashflows_year_ind["FIT"] +
                                        cashflows_year_ind["savings"] -
                                        cashflows_year_ind["O&M"])

        cashflows_year_ind["net_cf_curt"] = (cashflows_year_ind["FIT_curt"] +
                                             cashflows_year_ind["savings"] -
                                             cashflows_year_ind["O&M"])

        # --------P2P---------------------
        # Read avoided consumption from the grid (i.e. self-consumption)
        sc_h_p2p = df_p2p["sc_high"][yr]
        sc_l_p2p = df_p2p["sc_low"][yr]
        # Compute the savings from self-consuming solar electricity
        cashflows_year_p2p["savings_sc"] = sc_h_p2p*econ_inputs['retail_high'] +\
            sc_l_p2p * econ_inputs['retail_low']

        # Read consumption from the community excess solar (only positive for prosumers, consumers  will be  0)
        ex_p2p_h = df_p2p["net_demand_high"][yr] - df_p2p["net_demand_p2p_high"][yr]
        ex_p2p_l = df_p2p["net_demand_low"][yr] - df_p2p["net_demand_p2p_low"][yr]
        # Compute the savings from self-consuming solar electricity
        cashflows_year_p2p["savings_buycomm"] = ex_p2p_h*(econ_inputs['retail_high']-econ_inputs["p2p_high"]) +\
            ex_p2p_l*(econ_inputs['retail_low']-econ_inputs["p2p_low"])

        # Read year solar sold to community
        sol_to_cons_h = df_p2p["sold_excessto_comm_high"][yr]
        sol_to_cons_l = df_p2p["sold_excessto_comm_low"][yr]
        # Compute the revenues from solar sold to consumers in P2P market
        cashflows_year_p2p["to_comm"] = sol_to_cons_h * econ_inputs["p2p_high"] + sol_to_cons_l * econ_inputs["p2p_low"]

        # Read year excess solar sold to GRID
        sol_to_grid_h = df_p2p["sold_excessto_GRID_high"][yr]
        sol_to_grid_l = df_p2p["sold_excessto_GRID_low"][yr]

        # Compute the revenues from solar sold to GRID
        cashflows_year_p2p["to_grid"] = sol_to_grid_h * econ_inputs["fit_high"] + sol_to_grid_l * econ_inputs["fit_low"]

        # Read year CURTAILED excess solar sold to GRID
        sol_to_grid_h_curt = df_p2p["CURT_sold_excessto_GRID_high"][yr]
        sol_to_grid_l_curt = df_p2p["CURT_sold_excessto_GRID_low"][yr]

        # Compute the revenues from CURTAILED solar sold to GRID
        cashflows_year_p2p["to_grid_curt"] = sol_to_grid_h_curt * econ_inputs["fit_high"] + sol_to_grid_l_curt * econ_inputs["fit_low"]

        # Compute O&M costs
        cashflows_year_p2p["O&M"] = inv * econ_inputs["o&m_costs"]

        # standing costs in the UK
        cashflows_year_p2p["Standing_Costs"] = 365 * econ_inputs["standing_costs"]

        # Compute net cashflows to the agent
        cashflows_year_p2p["net_cf"] = (cashflows_year_p2p["to_grid"] +
                                        cashflows_year_p2p["to_comm"] +
                                        cashflows_year_p2p["savings_sc"] +
                                        cashflows_year_p2p["savings_buycomm"] -
                                        cashflows_year_p2p["O&M"])

        cashflows_year_p2p["net_cf_curt"] = (cashflows_year_p2p["to_grid_curt"] +
                                             cashflows_year_p2p["to_comm"] +
                                             cashflows_year_p2p["savings_sc"] +
                                             cashflows_year_p2p["savings_buycomm"] -
                                             cashflows_year_p2p["O&M"])

    # Store results in return dataframe
        if yr == 0:
            # If it is the first year, then create the dataframe
            lifetime_cashflows_ind = pd.DataFrame(cashflows_year_ind, index=[0])
            lifetime_cashflows_p2p = pd.DataFrame(cashflows_year_p2p, index=[0])
        else:
            # Append the dictionary containing the results for this year
            lifetime_cashflows_ind = lifetime_cashflows_ind.append(cashflows_year_ind,
                                                                   ignore_index=True)
            lifetime_cashflows_p2p = lifetime_cashflows_p2p.append(cashflows_year_p2p,
                                                                   ignore_index=True)

    return lifetime_cashflows_ind, lifetime_cashflows_p2p


def calc_costs(df, df_p2p, econ_inputs, pv_categs):
    '''
    '''
    bill_ind_no_pv_dict = {}
    bill_ind_with_pv_dict = {}
    bill_ind_with_pv_with_p2p_dict = {}
    for _i in list(df.keys()):
        cost_ind_no_pv, cost_ind_with_pv, cost_ind_with_pv_with_p2p = costs(
            df[_i], df_p2p[_i], econ_inputs, pv_categs)

        bill_ind_no_pv_dict[_i] = sum(cost_ind_no_pv.bill.tolist())  # bill_ind_no_pv
        bill_ind_with_pv_dict[_i] = sum(cost_ind_with_pv.bill.tolist())
        bill_ind_with_pv_with_p2p_dict[_i] = sum(cost_ind_with_pv_with_p2p.bill.tolist())

    return bill_ind_no_pv_dict, bill_ind_with_pv_dict, bill_ind_with_pv_with_p2p_dict


def costs(df, df_p2p, econ_inputs, pv_categs):
    '''
    calculates the electricity costs of the households
    '''
    cost_ind_no_pv = {}
    cost_ind_with_pv = {}
    cost_ind_with_pv_with_p2p = {}

    # individual
    # (a) no prosumers, and (b) prosumers without p2p
    for yr in range(len(df.index)):  # 30 years as that is the PV lifetime considered
        dem_h = df['demand_high'][yr]
        dem_l = df['demand_low'][yr]
        # bill for HHs before they installed PV
        cost_ind_no_pv['bill'] = dem_h*econ_inputs['retail_high'] +\
            dem_l*econ_inputs['retail_low'] +\
            365*econ_inputs["standing_costs"]

        net_dem_h = df['net_demand_high'][yr]
        net_dem_l = df['net_demand_low'][yr]
        excess_solar_h = df['excess_solar_high'][yr]
        excess_solar_l = df['excess_solar_low'][yr]

        # bill for HHs before after installed PV but no P2P trading
        cost_ind_with_pv['bill'] = net_dem_h*econ_inputs['retail_high'] +\
            net_dem_l*econ_inputs['retail_low'] +\
            365*econ_inputs["standing_costs"]

        if yr == 0:
            # If it is the first year, then create the dataframe
            ebill_no_prosumer = pd.DataFrame(cost_ind_no_pv,
                                             index=[0])
            ebill_with_prosumer = pd.DataFrame(cost_ind_with_pv,
                                               index=[0])
        else:
            # Append the dictionary containing the results for this year
            ebill_no_prosumer = ebill_no_prosumer.append(
                cost_ind_no_pv, ignore_index=True)
            ebill_with_prosumer = ebill_with_prosumer.append(
                cost_ind_with_pv, ignore_index=True)

    # (c) prosumers with p2p
    for yr in range(len(df_p2p.index)):  # 25 years as that is the PV lifetime considered
        net_dem_h = df_p2p['net_demand_high'][yr]
        net_dem_l = df_p2p['net_demand_low'][yr]
        net_dem_p2p_h = df_p2p['net_demand_p2p_high'][yr]
        net_dem_p2p_l = df_p2p['net_demand_p2p_low'][yr]
        excess_solar_grid_h = df_p2p['sold_excessto_GRID_high'][yr]
        excess_solar_grid_l = df_p2p['sold_excessto_GRID_low'][yr]
        excess_solar_comm_h = df_p2p['sold_excessto_comm_high'][yr]
        excess_solar_comm_l = df_p2p['sold_excessto_comm_low'][yr]

        # bill for HHs before after installed PV and also P2P trading
        cost_ind_with_pv_with_p2p['bill'] = net_dem_p2p_h*econ_inputs['retail_high'] +\
            net_dem_p2p_l*econ_inputs['retail_low'] +\
            (net_dem_h-net_dem_p2p_h)*econ_inputs['p2p_high'] +\
            (net_dem_l-net_dem_p2p_l)*econ_inputs['p2p_low'] +\
            365*econ_inputs["standing_costs"]

        if yr == 0:
            # If it is the first year, then create the dataframe
            ebill_with_prosumer_with_p2p = pd.DataFrame(cost_ind_with_pv_with_p2p,
                                                        index=[0])
        else:
            # Append the dictionary containing the results for this year
            ebill_with_prosumer_with_p2p = ebill_with_prosumer_with_p2p.append(
                cost_ind_with_pv_with_p2p, ignore_index=True)

    return ebill_no_prosumer, ebill_with_prosumer, ebill_with_prosumer_with_p2p
