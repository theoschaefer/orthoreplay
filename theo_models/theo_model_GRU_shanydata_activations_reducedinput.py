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
import matplotlib.patches as mpatches
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

def hot_to_binary(hot_vector):
    """
    Converts a 16-element hot (one-hot) vector into an 8-element binary vector.
    Assumes that the hot vector is organized in pairs:
      For each pair, if the sum is zero, default to 0; otherwise, the index of the maximum element (0 or 1)
    """
    binary = []
    for i in range(0, len(hot_vector), 2):
        pair = hot_vector[i:i+2]
        # If both entries are zero, default to 0; otherwise, take the argmax.
        if np.sum(pair) == 0:
            bit = 0
        else:
            bit = int(np.argmax(pair))
        binary.append(bit)
    return np.array(binary)

def get_active_trials_input_comb(n_input_nodes, n_actions, n_feat_itemA):
    """
    Create all trial inputs using binary encoding.
    
    For client features: 2 bits representing 2 binary features.
      Possible combinations: [0,0], [0,1], [1,0], [1,1] (4 options)
    
    For house features: 3 bits representing 3 binary features.
      Possible combinations: 8 options (from [0,0,0] to [1,1,1]).
    
    Each trial is the concatenation:
      [client (2 bits), left house (3 bits), right house (3 bits)]
    Total length = 2 + 3 + 3 = 8.
    
    We impose the constraint that the two houses must be different.
    """
    # Client options: all 4 combinations of 2 binary features.
    itemA_options = np.array([[0,0],
                              [0,1],
                              [1,0],
                              [1,1]])
    # House options: all 8 combinations of 3 binary features.
    itemB_options = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
    total_trials = itemA_options.shape[0] * (itemB_options.shape[0] * (itemB_options.shape[0] - 1))
    inputs = np.zeros((total_trials, n_input_nodes), dtype=int)
    trial_idx = -1
    for itemA in range(itemA_options.shape[0]):
        for itemBacomp in range(itemB_options.shape[0]):
            for itemBbcomp in range(itemB_options.shape[0]):
                if itemBacomp != itemBbcomp:  # Ensure houses differ.
                    trial_idx += 1
                    inputs[trial_idx, :] = np.concatenate([itemA_options[itemA],
                                                             itemB_options[itemBacomp],
                                                             itemB_options[itemBbcomp]])
    inputs = inputs[:trial_idx+1, :]
    return inputs

def rewardfun(cinputs, selected_action, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes):
    """
    Compute the reward given the two-timestep input cinputs (shape: [2, n_input_nodes]).
    
    For left action (selected_action==0): use client features (indices 0 to n_feat_itemA-1)
      and left house features (indices n_feat_itemA to n_feat_itemA+n_feat_itemB-1).
      
    For right action (selected_action==1): use client features and right house features
      (indices n_feat_itemA+n_feat_itemB to n_input_nodes-1).
    
    Reward is computed as the product of a dot product over client features and a dot product
    over house features, then scaled.
    """
    if selected_action == 0:
        client_vals = cinputs[0, 0:n_feat_itemA]
        house_vals = cinputs[1, n_feat_itemA:n_feat_itemA+n_feat_itemB]
        weights_client = reward_weights[0:n_feat_itemA]
        weights_house = reward_weights[n_feat_itemA:n_feat_itemA+n_feat_itemB]
    elif selected_action == 1:
        client_vals = cinputs[0, 0:n_feat_itemA]
        house_vals = cinputs[1, n_feat_itemA+n_feat_itemB:n_input_nodes]
        weights_client = reward_weights[0:n_feat_itemA]
        weights_house = reward_weights[n_feat_itemA:n_feat_itemA+n_feat_itemB]
    R = np.dot(client_vals, weights_client) * np.dot(house_vals, weights_house)
    R = (R * 6.25 + 50) / 100
    return R

def get_input_output_stateformation(sub, csvpath, isRecurrent):
    # Updated task parameters for binary coding:
    n_feat_itemA = 2   # 2 client binary features (1 bit each)
    n_feat_itemB = 3   # 3 house binary features (1 bit each)
    n_actions = 2      # 2 choice alternatives: left house / right house
    n_input_nodes = n_feat_itemA + 2 * n_feat_itemB  # 2 + 2*3 = 8
    # Define reward weights for client (2 weights) and house (3 weights).
    reward_weights = np.array([-1, 2, -1, 1, -3])  # total 5 weights
    
    if sub == 'randomized_episode':
        inputs = get_active_trials_input_comb(n_input_nodes, n_actions, n_feat_itemA)
    else:
        log_df = pd.read_csv(csvpath)
        csub_log = log_df[(log_df['sub'] == sub) & (log_df['is_passive'] == 0) & (log_df['block_type_num'] < 12)]
        tmp = np.array(csub_log.input)
        # Convert the string representations into integer arrays and then convert from hot to binary.
        inputs = np.array([hot_to_binary(np.fromstring(tmp[i][1:-1], dtype=int, sep=' ')) for i in range(len(tmp))])
    
    n_trials = inputs.shape[0]
    outputs = np.zeros((n_trials, 2))  # two output nodes per trial
    recurrent_inputs = np.zeros((n_trials, 2, n_input_nodes))  # trials x time points x input nodes 
    
    outputs[:] = np.nan
    recurrent_inputs[:] = np.nan
    
    for itrial in range(n_trials):
        # Timestep 1: cue is the first n_feat_itemA bits, padded with zeros for remaining features.
        recurrent_inputs[itrial, 0, :] = np.hstack((inputs[itrial][:n_feat_itemA], np.zeros(n_input_nodes - n_feat_itemA)))
        # Timestep 2: target info: pad with zeros for client, then the house features.
        recurrent_inputs[itrial, 1, :] = np.hstack((np.zeros(n_feat_itemA), inputs[itrial][n_feat_itemA:]))
        cinputs = recurrent_inputs[itrial, :, :]
        outputs[itrial, :] = [rewardfun(cinputs, x, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes)
                              for x in range(n_actions)]
    
    if isRecurrent:
        inputs = recurrent_inputs.copy()
    return inputs, outputs

# Convert to PyTorch dataset
class StateFormationDataset(Dataset):
    def __init__(self, sub, csvpath, isRecurrent=True):
        inputs, outputs = get_input_output_stateformation(sub, csvpath, isRecurrent)
        # inputs: shape (n_trials, 2, 8); outputs: shape (n_trials, 2)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
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
    def __init__(self, input_size=8, hidden_size=64, output_size=2, num_layers=1):
        super(GRUNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_activation = nn.ReLU()
        # Using a simple RNN (you can switch to GRU if preferred)
        self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh', bias=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, return_hidden=False):
        rnn_out, hidden = self.rnn(x)
        rnn_out_last = rnn_out[:, -1, :]
        fc_hidden_out = self.hidden_activation(self.fc_hidden(rnn_out_last))
        output = self.fc_out(fc_hidden_out)
        if return_hidden:
            return output, hidden, fc_hidden_out
        else:
            return output

#%% Run model

sub = "sub501"  # or "randomized_episode"

df_data_sub = df_data.query("sub == @sub & is_passive==0 & block_type_num < 12")

batch_size = 1
dataset = StateFormationDataset(sub, filename_data, isRecurrent=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()

num_independent_runs = 50  # Number of independent training runs ("epochs")
all_runs_accuracies = []
all_runs_losses = []
all_runs_hidden = []

for run in tqdm(range(num_independent_runs)):
    model = GRUNet(input_size=8, hidden_size=64, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    run_accuracies = []
    run_losses = []
    
    for inputs, targets, optimal_actions in dataloader:
        optimizer.zero_grad()
        outputs_pred = model(inputs)
        loss = criterion(outputs_pred, targets)
        loss.backward()
        optimizer.step()
        
        _, predicted_actions = torch.max(outputs_pred, dim=1)
        correct = (predicted_actions == optimal_actions).sum().item()
        
        run_losses.append(loss.item())
        run_accuracies.append(correct)
    
    all_runs_accuracies.append(run_accuracies)
    all_runs_losses.append(run_losses)
    
    # Extract and store hidden states
    model.eval()
    all_hidden_states = []
    with torch.no_grad():
        for inputs, targets, optimal_actions in dataloader:
            output, hidden, fc_hidden_out = model(inputs, return_hidden=True)
            final_hidden = hidden[-1].cpu().numpy()
            all_hidden_states.append(final_hidden)
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    all_runs_hidden.append(all_hidden_states)

all_hidden_states = np.mean(all_runs_hidden, axis=0)

#================== Plot last Epoch Results ============================================#

window = 25
moving_acc_runs = [np.convolve(run_acc, np.ones(window)/window, mode='valid') * 100 
                   for run_acc in all_runs_accuracies]
moving_loss_runs = [np.convolve(run_loss, np.ones(window)/window, mode='valid') 
                    for run_loss in all_runs_losses]

mean_moving_acc = np.mean(moving_acc_runs, axis=0)
mean_moving_loss = np.mean(moving_loss_runs, axis=0)

plt.figure(figsize=(12, 5))
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
# Now update the mapping functions for the new binary coding.

def get_cue_label(input_vector):
    """
    input_vector: 1D array (length 8). The first 2 elements represent client features.
    Mapping:
      (0,0) -> "male_noglass"
      (0,1) -> "male_glass"
      (1,0) -> "female_noglass"
      (1,1) -> "female_glass"
    """
    client_bits = tuple(input_vector[:2])
    cue_mapping = {
        (0,0): "male_noglass",
        (0,1): "male_glass",
        (1,0): "female_noglass",
        (1,1): "female_glass"
    }
    return cue_mapping.get(client_bits, "unknown")

def house_label(bits):
    """
    bits: 1D array (length 3) representing house features.
    Mapping:
      Feature 1 (material): 0 -> "wood", 1 -> "stone"
      Feature 2 (scene): 0 -> "forest", 1 -> "mountain"
      Feature 3 (accessory): 0 -> "swing", 1 -> "pool"
    """
    mapping_feature1 = {0: "wood", 1: "stone"}
    mapping_feature2 = {0: "forest", 1: "mountain"}
    mapping_feature3 = {0: "swing", 1: "pool"}
    return f"{mapping_feature1[bits[0]]}_{mapping_feature2[bits[1]]}_{mapping_feature3[bits[2]]}"

def get_stim_label(input_vector):
    """
    input_vector: 1D array (length 8).
      Left house: indices 2:5
      Right house: indices 5:8
    """
    left_bits = input_vector[2:5]
    right_bits = input_vector[5:8]
    left_label = house_label(left_bits)
    right_label = house_label(right_bits)
    return f"{left_label}_left__{right_label}_right"

# Example:
example_vector = np.array([1,0, 0,1,0, 1,1,0])
print("Cue ID:", get_cue_label(example_vector))
print("Stim ID:", get_stim_label(example_vector))

def parse_input_str(input_str):
    """
    Parses a string representation of a vector, e.g. "[1 0 0 1 0 1 1 0]".
    """
    return np.fromstring(input_str.strip()[1:-1], sep=' ', dtype=int)

df_data_sub['parsed_input'] = df_data_sub['input'].apply(parse_input_str)
df_data_sub['cue_id'] = df_data_sub['parsed_input'].apply(get_cue_label)
df_data_sub['stim_id'] = df_data_sub['parsed_input'].apply(get_stim_label)
print(df_data_sub[['input', 'cue_id', 'stim_id']])



#%% Dimensionality reduction plotting (separate plot for cue_id)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def dimensionality_reduction_plot(states, labels, algorithm='PCA', n_components=2, perplexity=30, random_state=42, ax=None):
    """
    Reduces the dimensionality of states using the specified algorithm and plots on the given axis.
    """
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}
    dot_colors = [color_dict[label] for label in labels]
    
    algorithm = algorithm.upper()
    if algorithm == 'PCA':
        from sklearn.decomposition import PCA
        dr_model = PCA(n_components=n_components)
    elif algorithm == 'TSNE':
        from sklearn.manifold import TSNE
        dr_model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    elif algorithm == 'UMAP':
        import umap
        dr_model = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Algorithm must be one of 'PCA', 'TSNE', or 'UMAP'.")
    
    embedding = dr_model.fit_transform(states)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    
    ax.scatter(embedding[:, 0], embedding[:, 1], c=dot_colors, alpha=0.7)
    ax.set_xlabel(f"{algorithm} Component 1")
    ax.set_ylabel(f"{algorithm} Component 2")
    ax.grid(True)
    patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
    ax.legend(handles=patches, fontsize='small')
    
    return embedding

# Plot cue_id separately:
fig, ax = plt.subplots(figsize=(8,6))
embedding = dimensionality_reduction_plot(
    states=all_hidden_states,
    labels=df_data_sub['cue_id'].values,
    algorithm='PCA',
    ax=ax
)
ax.set_title("PCA of Hidden States - Cue ID")
plt.show()
