#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 07:38:01 2025

@author: grossman
"""
import numpy as np
import pandas as pd
#%%
def get_input_output_stateformation(sub,csvpath,isRecurrent):
    # Some hardcode params of the task:
    n_feat_itemA=2 # 2 binary client features (male/female, glasses/no glasses)
    n_feat_itemB=3 # 3 binary house features (forest/mountain, swing/pool, )
    n_actions=2 # 2 choice alternative  - left house / right house
    n_input_nodes = 16 # 16 inputs node - each 2 nodes code a single binary featue, 8 features altogether (client + 2 houses)
    reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0,0]) #weights of each feature - client f1, client f2, house f1, house f2, house f3   
    if sub=='randomized_episode':
        inputs = get_active_trials_input_comb(n_input_nodes,n_actions,n_feat_itemA) #inputs: 192X16 array (need to break into 2 time points when passed forward)
        maxtrials = inputs.shape[0]
        assert(maxtrials==192)
        
    else:
        # Load CSV summerizing behaviour of all participants
        log_df = pd.read_csv(csvpath + 'State_Formation_behaviour.csv')
        
        # Filter active trials in day 1 of selected subjects 
        csub_log = log_df[(log_df['sub']==sub) & (log_df['is_passive']==0) & (log_df['block_type_num']<12)]
        
        # Convert inputs from strings to numpy array
        tmp = np.array(csub_log.input)
        inputs = np.array([np.fromstring(tmp[i][1:-1], dtype=int, sep=' ') for i in range(len(tmp))])
    
    # Break input into 2 time point (2X16 instead of 1X16), and compute reward (i.e. target output):
    n_trials = inputs.shape[0]
    outputs = np.zeros((n_trials,2)) #trials X outputnodes
    recurrent_inputs = np.zeros((n_trials,2,16)) # trials X time points X input nodes 
    outputs[:],recurrent_inputs[:] = np.nan, np.nan
    for itrial in range(n_trials):
        recurrent_inputs[itrial,0,:]=np.hstack((inputs[itrial][:4],np.zeros(12)))
        recurrent_inputs[itrial,1,:]=np.hstack((np.zeros(4),inputs[itrial][4:]))
        cinputs = recurrent_inputs[itrial,:,:]
        # Compute reward (i.e, Netwokks tragt output for left and right output nodes)
        outputs[itrial,:] = [rewardfun(cinputs, x, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) for x in range(n_actions)]

    if isRecurrent:
        inputs = recurrent_inputs.copy()
 
    return inputs, outputs

#%% Define a function that generates a shuffled episode of 192 trials (all possible inputs)
def get_active_trials_input_comb(n_input_nodes,n_actions,n_feat_itemA):       

    # make inputs by combining all choice options
    itemA_options = np.array([[0, 1, 0, 1],
                           [0, 1, 1, 0],
                           [1, 0, 0, 1],
                           [1, 0, 1, 0]])
    itemB_options = np.array([[0,1,0,1,0,1],[0,1,0,1,1,0],
                             [0,1,1,0,0,1],[0,1,1,0,1,0],
                             [1,0,0,1,0,1],[1,0,0,1,1,0],
                             [1,0,1,0,0,1],[1,0,1,0,1,0]])
                             #all possible options for a single itemB
    inputs = np.zeros((itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1), n_input_nodes)) # set array with all possible inputs (each row is an input for a single decision stage (i.e. itemA/itemB); entries 0-3 represent compA, entries 4-8 represent compB )
    inputs_itemA = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    inputs_itemBacomp = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    inputs_itemBbcomp = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    trial_idx = -1
    
    for itemA in np.arange(0,itemA_options.shape[0]):
        for itemBacomp in np.arange(0, itemB_options.shape[0]):  
            for idx, itemBbcomp in enumerate(np.setdiff1d(np.arange(0, itemB_options.shape[0]), itemBacomp)):
                if ~(itemB_options[itemBacomp,0:4]==itemB_options[itemBbcomp,0:4]).all():
                    trial_idx = trial_idx+1
                    inputs[trial_idx, :] = np.concatenate([itemA_options[itemA,:], itemB_options[itemBacomp,:], itemB_options[itemBbcomp]])
                    inputs_itemA[trial_idx]=itemA
                    inputs_itemBacomp[trial_idx]=itemBacomp
                    inputs_itemBbcomp[trial_idx]=itemBbcomp
    
    inputs=inputs[:trial_idx+1,:]
    inputs_itemA=inputs_itemA[:trial_idx+1]
    inputs_itemBacomp=inputs_itemBacomp[:trial_idx+1]
    inputs_itemBbcomp=inputs_itemBbcomp[:trial_idx+1]
 
    return inputs 

#%% Define a function that gets an input and returns reward for selected action (0--> left house, 1--> right house)
def rewardfun(cinputs, selected_action, reward_weights, n_feat_itemA, n_feat_itemB,n_actions, n_input_nodes): 
    # make choicemap: mask inputs based on each of 2 possible choices
    choicemap = np.empty((n_actions, n_input_nodes))
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))# itemB:1
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))# itemB:2
    
    cidx = np.flatnonzero(choicemap[selected_action, :])
    reward_weights_tmp = reward_weights.copy()
    R = np.dot(cinputs[0,cidx[np.arange(0, n_feat_itemA*2)]], reward_weights_tmp[0:n_feat_itemA*2]) \
        * np.dot(cinputs[1,cidx[np.arange(n_feat_itemA*2,n_feat_itemA*2+n_feat_itemB*2)]], reward_weights_tmp[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2] ) 
    R=(R*6.25+50)/100 #make sure to add this to all other!
    return(R)


   
            