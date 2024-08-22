# pv_sharing_comms
 Source code for the Solar PV Sharing in Urban Communities research article

data
This folder contains all necessary data required to conduct the energy community power balance simulations 

scripts
This folder contains all necessary python scripts to conduct the energy community power balance simulations 
Run scripts in the following order: 

1. data_to_json.py - creates json files for simulation input parameters
2. pv_gen.py - generates PV production data (PV system electricity output) for a 1kWp system
3. p2p_loops.py - this is the main file which executes the simulations. It loads in data from the data folder and calls functions from the p2p_functions.py and p2p_npv_functions.py scripts. 