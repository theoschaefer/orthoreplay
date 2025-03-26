#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:19:17 2025

@author: alexandernitsch

Lab retreat: neural network group

Easy non-recurrent network for Shany's task

"""


#%% IMPORTS

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


#%% INPUTS AND OUTPUTS FROM SHANY'S TASK
 
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


csvpath = '/Users/alexandernitsch/Nextcloud/Retreat/nn/' #  

# Call the fucntion the brings back input and outputs:
inputs, outputs = get_input_output_stateformation(sub='sub508',csvpath = csvpath, isRecurrent=False)   
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


#%% DEFINE AND TRAIN NETWORK

inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(inputs, outputs)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

# Hyperparameters
input_size = 16  # 4 for clients + 6 for left stimuli + 6 for right stimuli
hidden_size1 = 80
hidden_size2 = 40
output_size = 2  # Two outputs: reward for left and right stimuli
learning_rate = 0.01

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)


# Weigths and biases
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias) # make explicit

# Apply custom initialization
model.apply(initialize_weights)

# Loss and optimizer
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training 
losses = []
accuracies = []

for inp, tar in train_loader:
    
    # Forward pass
    train_output = model(inp)
    loss = criterion(train_output, tar)
    losses.append(loss.detach().numpy())
    
    # accuracy
    accuracies.append(np.argmax(train_output.detach().numpy())==np.argmax(tar.detach().numpy()))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
# Plot training loss
plt.plot(range(len(inputs)), losses, label='Training loss')
plt.xlabel('Trial')
plt.ylabel('Loss')
plt.title('Training loss over trials')
plt.legend()
plt.show()


# Plot accuracy with moving average

window_size = 10
moving_average = pd.Series(accuracies).rolling(window=window_size).mean()

plt.plot(accuracies, label='Accuracies', marker='o', linestyle='None')
plt.plot(moving_average, label=f'Moving average', color='orange')
plt.axhline(y=0.5, color='red', linestyle='--', label='Chance level (50%)')
plt.title('Accuracy per trial with moving average')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
# plt.legend()
plt.grid(True)
plt.show()


# Accessing weights and biases of the model
def print_weights_and_biases(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")
            # print(f"{name}")

# After training, print the weights and biases
# print_weights_and_biases(model)
