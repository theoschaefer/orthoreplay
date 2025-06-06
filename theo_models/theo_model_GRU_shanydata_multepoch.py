#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:58:24 2025

@author: schaefer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:22:57 2025

@author: schaefer
"""

#=============================================================================
# SETUP
#=============================================================================

import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


dir_script = os.getcwd()
dir_project = os.path.dirname(dir_script)



#%% Data functions
#=============================================================================
# LOAD AND TRANSFORM DATA
#=============================================================================


filename_data = f'{dir_project}/State_Formation_behaviour.csv'
df_rawdata = pd.read_csv(filename_data)
df_data = df_rawdata.copy()


def get_input_output_stateformation(sub, csvpath, isRecurrent):
    # Some hardcoded task parameters:
    n_feat_itemA = 2   # 2 binary client features (male/female, glasses/no glasses)
    n_feat_itemB = 3   # 3 binary house features (e.g., forest/mountain, swing/pool)
    n_actions = 2      # 2 choice alternatives: left house / right house
    n_input_nodes = 16 # 16 input nodes (each 2 nodes code a single binary feature)
    reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0,0])
    
    if sub == 'randomized_episode':
        inputs = get_active_trials_input_comb(n_input_nodes, n_actions, n_feat_itemA)
        maxtrials = inputs.shape[0]
        assert(maxtrials == 192)
    else:
        log_df = pd.read_csv(csvpath)
        # log_df = pd.read_csv(csvpath + 'State_Formation_behaviour.csv')
        csub_log = log_df[(log_df['sub'] == sub) & (log_df['is_passive'] == 0) & (log_df['block_type_num'] < 12)]
        tmp = np.array(csub_log.input)
        inputs = np.array([np.fromstring(tmp[i][1:-1], dtype=int, sep=' ') for i in range(len(tmp))])
    
    n_trials = inputs.shape[0]
    outputs = np.zeros((n_trials, 2))  # two output nodes per trial
    recurrent_inputs = np.zeros((n_trials, 2, 16))  # trials x time points x input nodes 
    
    # Initialize with NaN for debugging if needed
    outputs[:] = np.nan
    recurrent_inputs[:] = np.nan
    
    for itrial in range(n_trials):
        # Timestep 1: cue (first 4 features padded with zeros for remaining 12 dims)
        recurrent_inputs[itrial, 0, :] = np.hstack((inputs[itrial][:4], np.zeros(12)))
        # Timestep 2: target information (pad with zeros for first 4 dims, then features 4:16)
        recurrent_inputs[itrial, 1, :] = np.hstack((np.zeros(4), inputs[itrial][4:]))
        cinputs = recurrent_inputs[itrial, :, :]
        # Compute reward for each action (0: left, 1: right) using the reward function
        outputs[itrial, :] = [rewardfun(cinputs, x, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) 
                              for x in range(n_actions)]
    
    if isRecurrent:
        inputs = recurrent_inputs.copy()
    return inputs, outputs


def get_active_trials_input_comb(n_input_nodes, n_actions, n_feat_itemA):
    # make inputs by combining all choice options
    itemA_options = np.array([[0, 1, 0, 1],
                              [0, 1, 1, 0],
                              [1, 0, 0, 1],
                              [1, 0, 1, 0]])
    itemB_options = np.array([[0,1,0,1,0,1],
                              [0,1,0,1,1,0],
                              [0,1,1,0,0,1],
                              [0,1,1,0,1,0],
                              [1,0,0,1,0,1],
                              [1,0,0,1,1,0],
                              [1,0,1,0,0,1],
                              [1,0,1,0,1,0]])
    # all possible options for a single itemB
    # There will be a total of 192 trials (rows)
    inputs = np.zeros((itemA_options.shape[0] * itemB_options.shape[0] * (itemB_options.shape[0] - 1), n_input_nodes))
    trial_idx = -1
    
    for itemA in np.arange(itemA_options.shape[0]):
        for itemBacomp in np.arange(itemB_options.shape[0]):  
            for itemBbcomp in np.setdiff1d(np.arange(itemB_options.shape[0]), itemBacomp):
                # Here we check that the first four features are not identical (as in your provided code)
                if not (itemB_options[itemBacomp,0:4] == itemB_options[itemBbcomp,0:4]).all():
                    trial_idx += 1
                    # Concatenate: itemA_options (4 features), then itemBacomp (6 features) and itemBbcomp (6 features)
                    inputs[trial_idx, :] = np.concatenate([itemA_options[itemA, :],
                                                             itemB_options[itemBacomp, :],
                                                             itemB_options[itemBbcomp, :]])
    inputs = inputs[:trial_idx+1, :]
    return inputs


# Reward function
def rewardfun(cinputs, selected_action, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes): 
    # Define a choicemap that masks inputs based on each of 2 possible choices
    choicemap = np.empty((n_actions, n_input_nodes))
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))  # left option
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))  # right option
    
    cidx = np.flatnonzero(choicemap[selected_action, :])
    reward_weights_tmp = reward_weights.copy()
    R = np.dot(cinputs[0, cidx[np.arange(0, n_feat_itemA*2)]], reward_weights_tmp[0:n_feat_itemA*2]) * \
        np.dot(cinputs[1, cidx[np.arange(n_feat_itemA*2, n_feat_itemA*2+n_feat_itemB*2)]], 
               reward_weights_tmp[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2])
    # Scale reward as in your provided code
    R = (R * 6.25 + 50) / 100
    return R


# Convert to PyTorch dataset
class StateFormationDataset(Dataset):
    def __init__(self, sub, csvpath, isRecurrent=True):
        # Use the provided function to get inputs and outputs.
        inputs, outputs = get_input_output_stateformation(sub, csvpath, isRecurrent)
        # Convert arrays to appropriate torch tensors.
        # inputs: shape (n_trials, 2, 16); outputs: shape (n_trials, 2)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
        # Compute the optimal actions (ground truth) as the index of the maximum reward value
        self.optimal_actions = torch.argmax(self.outputs, dim=1)
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.optimal_actions[idx]


#%% GRU Network
#=============================================================================
# Network Model
#=============================================================================

class GRUNet(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=2, num_layers=1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU RNN layer
        # self.gru = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh', bias=True, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        # Linear Hidden layer
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hidden_activation = nn.ReLU()
        # Fully connected layer mapping hidden state to output reward estimates for two actions
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        rnn_out, _ = self.gru(x)
        # rnn_out, _ = self.gru(x, h0)
        # Take the output at the last time step
        rnn_out = rnn_out[:, -1, :]
        fc_hidden_out = self.hidden_activation(self.fc_hidden(rnn_out))
        output = self.fc(fc_hidden_out)
        return output

# # Two RNN layers
# class GRUNet(nn.Module):
#     def __init__(self, input_size=16, hidden_size=64, output_size=2, num_layers=2):
#         super(GRUNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # GRU layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         # Fully connected layer mapping hidden state to output reward estimates for two actions
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         # Initialize hidden state
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
#         out, _ = self.gru(x, h0)
#         # Take the output at the last time step
#         out = out[:, -1, :]
#         output = self.fc(out)
#         return output



#%% Run model

sub = "sub508"  # Alternative: "randomized_episode"

batch_size = 1
dataset = StateFormationDataset(sub, filename_data, isRecurrent=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model = GRUNet(input_size=16, hidden_size=50, output_size=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 10
epoch_accuracies = []
epoch_losses = []

for epoch in tqdm(range(num_epochs)):
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GRUNet(input_size=16, hidden_size=50, output_size=2)
    #% Training Loop and Saving Network Choices
    model.train()
    l_choices = []  
    l_losses = []
    l_accuracies = [] 
    
    # Loop through trials
    for inputs_batch, targets_batch, optimal_actions_batch in dataloader:
        inputs_batch = inputs_batch  # shape: (batch, 2, 16)
        targets_batch = targets_batch # shape: (batch, 2)
        optimal_actions_batch = optimal_actions_batch # shape: (batch,)
    
        optimizer.zero_grad()
        outputs_pred = model(inputs_batch)  # shape: (batch, 2)
        loss = criterion(outputs_pred, targets_batch)
        loss.backward()
        optimizer.step()
        
        # Compute predicted choices: index of maximum predicted reward
        _, predicted_actions = torch.max(outputs_pred, dim=1)
        l_choices.extend(predicted_actions.cpu().numpy().tolist())
    
        correct = (predicted_actions == optimal_actions_batch).sum().item()
    
        l_losses.append(loss.item())
        l_accuracies.append(correct)
    
    epoch_accuracies.append(l_accuracies)
    epoch_losses.append(l_losses)
    


#================== Plot last Epoch Results ============================================#


# window = 20
# moving_acc = np.convolve(l_accuracies, np.ones(window)/window, mode='valid') * 100
# moving_loss = np.convolve(l_losses, np.ones(window)/window, mode='valid')


# plt.figure(figsize=(10, 5))

# #% Plotting the Training Accuracy
# plt.subplot(1, 2, 1)
# # plt.plot(np.arange(1, len(l_accuracies)+1), l_accuracies, marker='o')
# plt.plot(np.arange(1, len(moving_acc)+1), moving_acc, marker='o', color='green')
# plt.xlabel("Trial")
# plt.ylabel("Training Accuracy (%)")
# plt.title("Training Accuracy over Trials")
# plt.axhline(y=50, color='red', linestyle='dotted')
# plt.grid(True)
# plt.ylim([0,100])
# # plt.show()

# #% Plotting the Training Loss
# plt.subplot(1, 2, 2)
# # plt.plot(np.arange(1, len(l_losses)+1), l_losses, marker='o')
# plt.plot(np.arange(1, len(moving_loss)+1), moving_loss, marker='o', color='darkred')
# plt.xlabel("Trial")
# plt.ylabel("Training Loss")
# plt.title("Loss over Trials")
# plt.axhline(y=0, color='green', linestyle='dotted')
# plt.grid(True)
# plt.show()


#%%
#================== Plot last Epoch Results ============================================#


window = 20
moving_acc = [np.convolve(l_acc, np.ones(window)/window, mode='valid') * 100 
              for l_acc in epoch_accuracies]
moving_loss = [np.convolve(l_loss, np.ones(window)/window, mode='valid') 
               for l_loss in epoch_losses]


plt.figure(figsize=(10, 5))

#% Plotting the Training Accuracy
plt.subplot(1, 2, 1)
# plt.plot(np.arange(1, len(l_accuracies)+1), l_accuracies, marker='o')
# plt.plot(np.arange(1, len(moving_acc[0])+1), moving_acc[0], marker='o', alpha=.1)
plt.plot(moving_acc, marker='o', alpha=.1)
plt.xlabel("Trial")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy over Trials")
plt.axhline(y=50, color='red', linestyle='dotted')
plt.grid(True)
plt.ylim([0,100])
# plt.show()

#% Plotting the Training Loss
plt.subplot(1, 2, 2)
# plt.plot(np.arange(1, len(l_losses)+1), l_losses, marker='o')
plt.plot(moving_loss, marker='o', color='darkred', alpha=.1)
# plt.plot(np.arange(1, len(moving_loss[0])+1), moving_loss[0], marker='o', color='darkred', alpha=.1)
plt.xlabel("Trial")
plt.ylabel("Training Loss")
plt.title("Loss over Trials")
plt.axhline(y=0, color='green', linestyle='dotted')
plt.grid(True)
plt.show()




