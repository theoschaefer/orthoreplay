#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 07:38:01 2025

@author: grossman
"""
import numpy as np
import pandas as pd
#%%
def get_input_output_stateformation(sub,csvpath,isRecurrent,withPassive):
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
        
        
        if withPassive:
            # Filter active and assive trials in day 1 of selected subject 
            csub_log = log_df[(log_df['sub']==sub) & (log_df['block_type_num']<12)]
            
        else:
            # Filter active trials in day 1 of selected subject
            csub_log = log_df[(log_df['sub']==sub) & (log_df['is_passive']==0) & (log_df['block_type_num']<12)]
            
        # Convert inputs from strings to numpy array
        tmp = np.array(csub_log.input)
        inputs = np.array([np.fromstring(tmp[i][1:-1], dtype=int, sep=' ') for i in range(len(tmp))])
    
    # Break input into 2 time point (2X16 instead of 1X16), and compute reward (i.e. target output):
    n_trials = inputs.shape[0]
    outputs = np.zeros((n_trials,2)) #trials X outputnodes
    states = np.empty(n_trials, dtype=object)
    recurrent_inputs = np.zeros((n_trials,2,16)) # trials X time points X input nodes 
    outputs[:],recurrent_inputs[:], states[:] = np.nan, np.nan, np.nan
    isPassive = np.zeros(n_trials)
    for itrial in range(n_trials):
        recurrent_inputs[itrial,0,:]=np.hstack((inputs[itrial][:4],np.zeros(12)))
        recurrent_inputs[itrial,1,:]=np.hstack((np.zeros(4),inputs[itrial][4:]))
        cinputs = recurrent_inputs[itrial,:,:]
        
        
        # Compute reward (i.e, Netwokks tragt output for left and right output nodes)
        if np.all(cinputs[1,10:]==0): # passive trial - osingle house is presented (coded in the left house position at cinputs[1,4:10])
            outputs[itrial,0] = rewardfun(cinputs, 0, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) 
            outputs[itrial,1],isPassive[itrial]  = np.nan, 1 #insert nan in output for right house

        else: # active trial - 2 possible outputs
            outputs[itrial,:] = [rewardfun(cinputs, x, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) for x in range(n_actions)]
            
        # Log state code
        states[itrial] = input2state(inputs[itrial][np.array([0,1,4,5,6,7,10,11,12,13])])
        

    if isRecurrent:
        inputs = recurrent_inputs.copy()
 
    return inputs, outputs, isPassive, states

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
    # geberate choicemap: mask inputs based on each of 2 possible choices
    choicemap = np.empty((n_actions, n_input_nodes))
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))# itemB:1
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))# itemB:2
    
    cidx = np.flatnonzero(choicemap[selected_action, :])
    reward_weights_tmp = reward_weights.copy()
    R = np.dot(cinputs[0,cidx[np.arange(0, n_feat_itemA*2)]], reward_weights_tmp[0:n_feat_itemA*2]) \
        * np.dot(cinputs[1,cidx[np.arange(n_feat_itemA*2,n_feat_itemA*2+n_feat_itemB*2)]], reward_weights_tmp[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2] ) 
    R=(R*6.25+50)/100 
    return(R)


#%% Define a function that recieves input array/state codes and returns state code/input array, respectively   
def input2state(x):
    # a1/b1 = 0,1,0,1
    # a2/b2 = 0,1,1,0
    # a3/b3 = 1,0,0,1
    # a4/b4 = 1,0,1,0
     
    state_2_input_dict = {'a1a2': np.array([1,0,0,1,0,1,0,1,1,0]),
                              'a2a1': np.array([1,0,0,1,1,0,0,1,0,1]),
                              'a1a3': np.array([1,0,0,1,0,1,1,0,0,1]),
                              'a3a1': np.array([1,0,1,0,0,1,0,1,0,1]),
                              'a1a4': np.array([1,0,0,1,0,1,1,0,1,0]),
                              'a4a1': np.array([1,0,1,0,1,0,0,1,0,1]),
                              'a2a3': np.array([1,0,0,1,1,0,1,0,0,1]),
                              'a3a2': np.array([1,0,1,0,0,1,0,1,1,0]),
                              'a2a4': np.array([1,0,0,1,1,0,1,0,1,0]),
                              'a4a2': np.array([1,0,1,0,1,0,0,1,1,0]),
                              'a3a4': np.array([1,0,1,0,0,1,1,0,1,0]),
                              'a4a3': np.array([1,0,1,0,1,0,1,0,0,1]),
                              'b1b2': np.array([0,1,0,1,0,1,0,1,1,0]),
                              'b2b1': np.array([0,1,0,1,1,0,0,1,0,1]),
                              'b1b3': np.array([0,1,0,1,0,1,1,0,0,1]),
                              'b3b1': np.array([0,1,1,0,0,1,0,1,0,1]),
                              'b1b4': np.array([0,1,0,1,0,1,1,0,1,0]),
                              'b4b1': np.array([0,1,1,0,1,0,0,1,0,1]),
                              'b2b3': np.array([0,1,0,1,1,0,1,0,0,1]),
                              'b3b2': np.array([0,1,1,0,0,1,0,1,1,0]),
                              'b2b4': np.array([0,1,0,1,1,0,1,0,1,0]),
                              'b4b2': np.array([0,1,1,0,1,0,0,1,1,0]),
                              'b3b4': np.array([0,1,1,0,0,1,1,0,1,0]),
                              'b4b3': np.array([0,1,1,0,1,0,1,0,0,1]),
                              'a1_l': np.array([1,0,0,1,0,1,0,0,0,0]),
                              'a2_l': np.array([1,0,0,1,1,0,0,0,0,0]),
                              'a3_l': np.array([1,0,1,0,0,1,0,0,0,0]),
                              'a4_l': np.array([1,0,1,0,1,0,0,0,0,0]),
                              'b1_l': np.array([0,1,0,1,0,1,0,0,0,0]),
                              'b2_l': np.array([0,1,0,1,1,0,0,0,0,0]),
                              'b3_l': np.array([0,1,1,0,0,1,0,0,0,0]),
                              'b4_l': np.array([0,1,1,0,1,0,0,0,0,0]),
                              'a1_r': np.array([1,0,0,0,0,0,0,1,0,1]),
                              'a2_r': np.array([1,0,0,0,0,0,0,1,1,0]),
                              'a3_r': np.array([1,0,0,0,0,0,1,0,0,1]),
                              'a4_r': np.array([1,0,0,0,0,0,1,0,1,0]),
                              'b1_r': np.array([0,1,0,0,0,0,0,1,0,1]),
                              'b2_r': np.array([0,1,0,0,0,0,0,1,1,0]),
                              'b3_r': np.array([0,1,0,0,0,0,1,0,0,1]),
                              'b4_r': np.array([0,1,0,0,0,0,1,0,1,0])} #for the sake of simplicity we take only those for the left hand side
    
    
    # Reverse dictionary function
    def reverse_dict(d):
        return {tuple(v): k for k, v in d.items()}
    
    # Create reversed dictionaries
    input_2_state_dict = reverse_dict(state_2_input_dict)
    #example ussage: input_2_state_dict[tuple(input_array)]
    
    if isinstance(x, str): #convert state to array
        y = state_2_input_dict[x]
    elif isinstance(x, np.ndarray): #convert array to state
        y = input_2_state_dict[tuple(x)]
    
    return(y)