
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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_activation = nn.ReLU()
        # GRU RNN layer
        # self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh', bias=True, batch_first=True)
        self.rnn = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        # Fully connected Linear Hidden layer
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        # Fully connected output layer mapping hidden state to output reward estimates for two actions
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, return_hidden=False):
        # Initialize hidden state
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        rnn_out, hidden = self.rnn(x)
        # rnn_out, _ = self.gru(x, h0)
        # Take the output at the last time step
        rnn_out_last = rnn_out[:, -1, :]
        fc_hidden_out = self.hidden_activation(self.fc_hidden(rnn_out_last))
        output = self.fc_out(fc_hidden_out)
        
        if return_hidden:
            return output, hidden, fc_hidden_out
        else:
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

criterion = nn.MSELoss()

num_independent_runs = 50  # Number of independent training runs ("epochs")
all_runs_accuracies = []
all_runs_losses = []

for run in tqdm(range(num_independent_runs)):
    # Reinitialize the model for each independent run
    model = GRUNet(input_size=16, hidden_size=64, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    run_accuracies = []  # Track accuracy for this run (number correct per trial)
    run_losses = []      # Track loss for this run
    
    # Train over all trials in the dataset
    for inputs, targets, optimal_actions in dataloader:
        optimizer.zero_grad()
        outputs_pred = model(inputs)  # Shape: (batch, 2)
        loss = criterion(outputs_pred, targets)
        loss.backward()
        optimizer.step()
        
        # Calculate predicted choices: index of maximum predicted reward
        _, predicted_actions = torch.max(outputs_pred, dim=1)
        correct = (predicted_actions == optimal_actions).sum().item()
        
        run_losses.append(loss.item())
        run_accuracies.append(correct)
    
    all_runs_accuracies.append(run_accuracies)
    all_runs_losses.append(run_losses)



#================== Plot last Epoch Results ============================================#


# Define a moving average window
window = 25

# Compute moving averages for each run
moving_acc_runs = [np.convolve(run_acc, np.ones(window)/window, mode='valid') * 100 
                   for run_acc in all_runs_accuracies]
moving_loss_runs = [np.convolve(run_loss, np.ones(window)/window, mode='valid') 
                    for run_loss in all_runs_losses]

# Calculate the mean moving average across runs (assuming all runs have the same length)
mean_moving_acc = np.mean(moving_acc_runs, axis=0)
mean_moving_loss = np.mean(moving_loss_runs, axis=0)

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
for run_acc in moving_acc_runs:
    plt.plot(run_acc, marker='.', alpha=0.1, color='green')
plt.plot(mean_moving_acc, marker='.', color='black', linewidth=2, label='Mean Accuracy')
plt.xlabel("Trial")
plt.ylabel("Training Accuracy (%)")
plt.title("Independent Training Runs: Accuracy")
plt.axhline(y=50, color='red', linestyle='dotted', label='Chance (50%)')
plt.grid(True)
plt.ylim([0, 100])
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
for run_loss in moving_loss_runs:
    plt.plot(run_loss, marker='.', alpha=0.1, color='darkred')
plt.plot(mean_moving_loss, marker='.', color='black', linewidth=2, label='Mean Loss')
plt.xlabel("Trial")
plt.ylabel("Training Loss")
plt.title("Independent Training Runs: Loss")
plt.axhline(y=0, color='green', linestyle='dotted', label='Zero Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


#%% Condition files 




#%% Extract and Store Hidden States

# Switch to evaluation mode
model.eval()
all_hidden_states = []

with torch.no_grad():
    for inputs, targets, optimal_actions in dataloader:
        # Pass the data through the network and request hidden states
        output, hidden, fc_hidden_out = model(inputs, return_hidden=True)
        # Extract the hidden state from the last layer for the current batch.
        # hidden has shape (num_layers, batch, hidden_size), so hidden[-1] gives shape (batch, hidden_size)
        final_hidden = hidden[-1].cpu().numpy()
        all_hidden_states.append(final_hidden)

# Concatenate all hidden states along the batch dimension
all_hidden_states = np.concatenate(all_hidden_states, axis=0)



#%% Apply PCA for Dimensionality Reduction

from sklearn.decomposition import PCA

# Perform PCA to reduce hidden state dimensions (e.g., to 2 components)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_hidden_states)

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of GRU Hidden States")
plt.grid(True)
plt.show()


#%% t-SNE

from sklearn.manifold import TSNE


# Apply t-SNE to reduce hidden state dimensionality to 2 components.
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(all_hidden_states)

# Plot t-SNE results.
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE of GRU Hidden States")
plt.grid(True)
plt.show()


#%% UMAP

import umap


# Apply UMAP to reduce dimensionality to 2 components.
umap_embedder = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_embedder.fit_transform(all_hidden_states)

# Plot UMAP results.
plt.figure(figsize=(8, 6))
plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.7)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("UMAP of GRU Hidden States")
plt.grid(True)
plt.show()