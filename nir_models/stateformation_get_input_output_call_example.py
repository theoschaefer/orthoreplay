#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:58:31 2025

@author: grossman
"""
# Import helper functions from stateformation_get_input_output.py:
from stateformation_get_input_output import get_input_output_stateformation
from stateformation_get_input_output import get_active_trials_input_comb
from stateformation_get_input_output import rewardfun

#%% Define manually:
csvpath = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/LOGs/' #  
# Call the fucntion the brings back input and outputs:
inputs, outputs = get_input_output_stateformation(sub='sub508',csvpath = csvpath, isRecurrent=True)   
 # INPUTS:
     # sub - either a real participant from the state formation experiment, e.g. 'sub508' see list of available particiapnts below 
     #       or, if sub='randomized_episode', returns a suffelled input of the 192 unique inputs of the task
     # csvpath - see above, path to where to place the State_Formation_behaviour.csv
     # isRecurrent - if True, inputs are returnet as a 2X16 array per input, first row holds the nodes of the context and 2nd row the nodes of the houses
     #             - if False, inputs are retured as 1X16 array per input - both client and house pair presented simultanously to the network
 # OUTPUTS:
     # inputs: [n_trialsX n time points X n input nodes] numpy array, to get input of the ith trial  inputs[i,:,:]
     # outputs: [n_trials X n outcomes] numpy array: for each tial, the reward for the left [i,0] and right [i,1] house

# all subjects names:
# ['sub501', 'sub502', 'sub503', 'sub504', 'sub505', 'sub506', 'sub507', 'sub508', 'sub509', 'sub510',
#  'sub511', 'sub512', 'sub513', 'sub514', 'sub515', 'sub516', 'sub517', 'sub518', 'sub519', 'sub520', 
#  'sub521', 'sub522', 'sub523', 'sub524', 'sub525', 'sub526', 'sub527', 'sub528', 'sub529', 'sub530', 
#  'sub531', 'sub532', 'sub533', 'sub534', 'sub535', 'sub536', 'sub537', 'sub538', 'sub539', 'sub540', 
#  'sub541', 'sub542', 'sub543', 'sub544', 'sub545', 'sub546', 'sub547', 'sub548', 'sub549', 'sub550', 
#  'sub551', 'sub552', 'sub553', 'sub554', 'sub555', 'sub556', 'sub557', 'sub558', 'sub559', 'sub560', 
#  'sub561', 'sub562', 'sub563', 'sub564'] 
