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
from stateformation_get_input_output import input2state

subjects =  ['sub501', 'sub502', 'sub503', 'sub504', 'sub505', 'sub506', 'sub507', 'sub508', 'sub509', 'sub510',
  'sub511', 'sub512', 'sub513', 'sub514', 'sub515', 'sub516', 'sub517', 'sub518', 'sub519', 'sub520', 
  'sub521', 'sub522', 'sub523', 'sub524', 'sub525', 'sub526', 'sub527', 'sub528', 'sub529', 'sub530', 
  'sub531', 'sub532', 'sub533', 'sub534', 'sub535', 'sub536', 'sub537', 'sub538', 'sub539','sub540', 
  'sub541', 'sub542','sub543',  'sub544', 'sub545', 'sub546', 'sub547', 'sub548', 'sub549', 'sub550', 
  'sub551', 'sub552', 'sub553' ,'sub554', 'sub555', 'sub556', 'sub557', 'sub558', 'sub559', 'sub560', 
  'sub561', 'sub562', 'sub563', 'sub564'] 

#%% Define manually:
csvpath = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/LOGs/' #  


# Call the function the brings back input and outputs:
inputs, outputs, isPassive, states, block, passive_type, responses, correct_responses = get_input_output_stateformation(sub='sub501',csvpath = csvpath, isRecurrent=True, withPassive = True)   


 # INPUTS:
     # sub: either a real participant from the state formation experiment, e.g. 'sub508' see list of available particiapnts below 
     #      or, if sub='randomized_episode', returns a suffelled input of the 192 unique inputs of the task
     # csvpath: see above, path to where to place the State_Formation_behaviour.csv
     # isRecurrent: - if True, inputs are returnet as a 2X16 array per input, first row holds the nodes of the context and 2nd row the nodes of the houses
     #              - if False, inputs are retured as 1X16 array per input - both client and house pair presented simultanously to the network
     # withPassive: If True, the outputs include also passive trials inserted pre and post active trials segments.
     #              Passive trials inputs include a single house (not a pair), encoded in the left house position (inputs[i,1,4:10]), 
     #              the right house input is all zeros (inputs[i,1,10:]).
     #              For passive trials, the output arrays will have only the value of the left house (outputs[i,0]) and nan in the right house (outputs[i,1])
                     
 # OUTPUTS:
     # inputs: [n_trialsX n time points X n input nodes] numpy array, to get input of the ith trial  inputs[i,:,:]
     # outputs: [n_trials X n outcomes] numpy array: for each tial, the reward for the left [i,0] and right [i,1] house
     # isPassive: [n_trials] array with 0 for active trials and 1 for passive trials 
     # states: [n_trials] arrays of strings stating the state on each trial. Active trials are encoded as e.g. a1a2, meaning context a with house type 1 on the left and house type 2 on the right.
     #         passive trials are encoded as e.g., b2_l, meaning context b with house type 2 on left position (position is not relevant for a serial architecture). All passive inputs are encoded as positioned on the left by default.
     # block: [n_trial] araay with the learning blokc number. 11 blocks alltogether, each contains 42 trials (aside from sub501, sub502 for whom it has 40 trials each)
     # passive_type: [n_trial] which indicates for passive trials, to which segment they belong in the block (pre_early /  post_early / pre_late /  post_late):
     #         A single learning block has this structure: 
     #         4 passive trials (pre_early) --> 4 active trials --> 4 passive trials repeat (post_early) --> 16 active trials --> 5 passive trials (pre_late) --> 4 active trials --> 5 passive trials (post_late)
     #         pre_late and post_late contain 5 passive trials, since one of the 5 states  was also present in the active trials coming inbetween (we added these as of sub503 to get a some ceiling measure of neural change)
     # responses: [n_trials] arrays of strings containing repsonse per trial: for active trials:  'right'/ 'left'; for passive trials: 'more', 'less'; if there was no response in time: 'none'
     # correct_responses: [n_trials] arrays of strings containing the correct repsonse per trial [active:'right'/'left', passive: 'more'/'less']

# all subjects names:
# ['sub501', 'sub502', 'sub503', 'sub504', 'sub505', 'sub506', 'sub507', 'sub508', 'sub509', 'sub510',
#   'sub511', 'sub512', 'sub513', 'sub514', 'sub515', 'sub516', 'sub517', 'sub518', 'sub519', 'sub520', 
#   'sub521', 'sub522', 'sub523', 'sub524', 'sub525', 'sub526', 'sub527', 'sub528', 'sub529', 'sub530', 
#   'sub531', 'sub532', 'sub533', 'sub534', 'sub535', 'sub536', 'sub537', 'sub538', 'sub539','sub540', 
#   'sub541', 'sub542','sub543',  'sub544', 'sub545', 'sub546', 'sub547', 'sub548', 'sub549', 'sub550', 
#   'sub551', 'sub552', 'sub553' ,'sub554', 'sub555', 'sub556', 'sub557', 'sub558', 'sub559', 'sub560', 
#   'sub561', 'sub562', 'sub563', 'sub564'] 

# Subjects to take care of partial blocks when considering behaviour! 
# 'sub542', 'sub544', 'sub553' ,'sub554'
