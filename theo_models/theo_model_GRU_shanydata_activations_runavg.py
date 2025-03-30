
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


def get_input_output_stateformation(sub, csvpath, isRecurrent):
    # Some hardcoded task parameters:
    n_feat_itemA = 2   # 2 binary client features (male/female, glasses/no glasses)
    n_feat_itemB = 3   # 3 binary house features (e.g., forest/mountain, swing/pool)
    n_actions = 2      # 2 choice alternatives: left house / right house
    n_input_nodes = 16 # 16 input nodes (each 2 nodes code a single binary feature)
    # reward_weights = np.array([-1,2,0,0,-1,1,-0,0,-3,10])
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
        self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh', 
                          bias=True, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, 1, bias=True, batch_first=True)
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

df_data_sub = df_data.query("sub == @sub & is_passive==0 & block_type_num < 12")
# df_data_sub = log_df[(log_df['sub'] == sub) & (log_df['is_passive'] == 0) & (log_df['block_type_num'] < 12)]


batch_size = 1
dataset = StateFormationDataset(sub, filename_data, isRecurrent=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()

num_independent_runs = 50  # Number of independent training runs ("epochs")
all_runs_accuracies = []
all_runs_losses = []
all_runs_hidden = []

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

    #% Extract and Store Hidden States

    # Switch to evaluation mode
    model.eval()
    all_hidden_states = []

    with torch.no_grad():
        for inputs, targets, optimal_actions in dataloader:
            # Pass the data through the network and request hidden states
            output, hidden, fc_hidden_out = model(inputs, return_hidden=True)
            # Extract the hidden state from the last layer for the current batch.
            # hidden has shape (num_layers, batch, hidden_size), so hidden[-1] gives shape (batch, hidden_size)
            # final_hidden = fc_hidden_out.cpu().numpy()
            final_hidden = hidden[-1].cpu().numpy()
            all_hidden_states.append(final_hidden)
            
    # Concatenate all hidden states along the batch dimension
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    all_runs_hidden.append(all_hidden_states)


# Save mean result
all_hidden_states = np.mean(all_runs_hidden, axis=0)



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


# Example mapping for the cue:
def get_cue_label(input_vector):
    """
    input_vector: 1D array or list with 16 binary elements.
    The first 4 elements encode two binary features (2 bits each).
    """
    # Convert first 4 elements to a tuple for dictionary lookup.
    cue_bits = tuple(input_vector[:4])
    cue_mapping = {
        (1, 0, 1, 0): "male_glass",
        (1, 0, 0, 1): "male_noglass",
        (0, 1, 1, 0): "female_glass",
        (0, 1, 0, 1): "female_noglass"
    }
    return cue_mapping.get(cue_bits, "unknown")

# Mapping function for one house (6-bit vector)
def house_label(bits):
    """
    bits: 6-element array or list, assumed to encode 3 binary features as 3 pairs.
    For each feature, we map:
        Feature 1: (0,1) -> "wood", (1,0) -> "stone"
        Feature 2: (0,1) -> "forest", (1,0) -> "mountain"
        Feature 3: (0,1) -> "swing", (1,0) -> "pool"
    """
    features = np.array(bits).reshape(3, 2)
    label_parts = []
    mapping_feature1 = {(1, 0): "wood", (0, 1): "stone"}
    mapping_feature2 = {(1, 0): "forest", (0, 1): "mountain"}
    mapping_feature3 = {(1, 0): "swing", (0, 1): "pool"}
    mappings = [mapping_feature1, mapping_feature2, mapping_feature3]
    
    for pair, mapping in zip(features, mappings):
        pair_tuple = tuple(pair)
        label_parts.append(mapping.get(pair_tuple, "unknown"))
    return "_".join(label_parts)

def get_stim_label(input_vector):
    """
    input_vector: 1D array or list with 16 elements.
    We assume:
      - Elements 4 to 9 (indices 4:10) encode the left stimulus (6 bits).
      - Elements 10 to 15 (indices 10:16) encode the right stimulus (6 bits).
    This function computes a label for each house and combines them.
    """
    left_bits = input_vector[4:10]
    right_bits = input_vector[10:16]
    left_label = house_label(left_bits)
    right_label = house_label(right_bits)
    # Combine labels; adjust the separator as desired.
    return f"{left_label}_left__{right_label}_right"

# Example: Process a single input vector.
example_vector = np.array([0, 1, 0, 1,   0, 1, 0, 1, 0, 1,   1, 0, 0, 1, 1, 0])
print("Cue ID:", get_cue_label(example_vector))
print("Stim ID:", get_stim_label(example_vector))

# Now, assume you have a DataFrame (df_data) with a column 'input'
# that contains the string representation of a 16-element vector.
# For example: "[0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0]"

def parse_input_str(input_str):
    """
    Parses a string representation of a vector, e.g. "[0 1 0 1 ...]".
    Adjust this function if your format differs.
    """
    # Remove brackets and split by space, then convert to integers.
    return np.fromstring(input_str.strip()[1:-1], sep=' ', dtype=int)



# Parse the input strings into numeric arrays.
df_data_sub['parsed_input'] = df_data_sub['input'].apply(parse_input_str)

# Create the cue_id and stim_id columns.
df_data_sub['cue_id'] = df_data_sub['parsed_input'].apply(get_cue_label)
df_data_sub['stim_id'] = df_data_sub['parsed_input'].apply(get_stim_label)

# Optionally, drop the parsed_input column if no longer needed.
# df_data.drop(columns=['parsed_input'], inplace=True)

print(df_data_sub[['input', 'cue_id', 'stim_id']])




#%% Dimensionality reduction

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def dimensionality_reduction_plot(states, labels, algorithm='PCA', n_components=2, perplexity=30, random_state=42):
    """
    Reduces the dimensionality of the states using the specified algorithm and plots the results.
    
    Parameters:
      states (np.array): Array of hidden states with shape (n_samples, state_dim).
      labels (array-like): 1D array or list of labels for each state.
      algorithm (str): Dimensionality reduction algorithm; one of 'PCA', 'TSNE', or 'UMAP'.
      n_components (int): Number of components to reduce to (default 2).
      perplexity (int): Perplexity parameter for t-SNE (default 30).
      random_state (int): Random state for reproducibility (default 42).
    
    Returns:
      embedding (np.array): The reduced states (embedding) with shape (n_samples, n_components).
    """
    # Build color mapping from labels
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}
    dot_colors = [color_dict[label] for label in labels]
    
    # Dimensionality reduction based on selected algorithm
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
    
    # Compute the embedding
    embedding = dr_model.fit_transform(states)
    
    # Set plot title if not provided explicitly
    title = f"{algorithm} of Hidden States"
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=dot_colors, alpha=0.7)
    plt.xlabel(f"{algorithm} Component 1")
    plt.ylabel(f"{algorithm} Component 2")
    plt.title(title)
    plt.grid(True)
    
    # Create legend mapping colors to labels
    patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
    plt.legend(handles=patches)
    
    plt.show()
    
    return embedding


def get_label_dotcolors(stimtype='stim_id', side=None, feature=None):
    '''
    stimtype : cue_id or stim_id
    side : 0 for left, 1 for right
    feature: 0 material, 1 scene, 2 irrelevant
    '''

    if stimtype=='cue_id':
        labels = df_data_sub['cue_id'].values
    elif stimtype=='stim_id':
        labels = df_data_sub['stim_id'].values
        
        if side is not None:
            labels = [label.split('__')[side] for label in df_data_sub['stim_id'].values]
            
            if feature is not None:
                labels = [label.split('_')[feature] for label in labels]
    
    unique_labels = np.unique(labels)

    # Create a color mapping: assign a unique color to each label.
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Map each trial's label to a color.
    dot_colors = [color_dict[label] for label in labels]
    
    return dot_colors


#% label loop


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Modified version of your dimensionality_reduction_plot that accepts an axis.
def dimensionality_reduction_plot(states, labels, algorithm='PCA', n_components=2, perplexity=30, random_state=42, ax=None):
    """
    Reduces the dimensionality of the states using the specified algorithm and plots on the provided axis.
    
    Parameters:
      states (np.array): Array of hidden states with shape (n_samples, state_dim).
      labels (array-like): 1D array or list of labels for each state.
      algorithm (str): Dimensionality reduction algorithm ('PCA', 'TSNE', or 'UMAP').
      n_components (int): Number of components to reduce to.
      perplexity (int): Perplexity for t-SNE.
      random_state (int): Random state for reproducibility.
      ax (matplotlib.axes.Axes): Axis on which to plot; if None, a new figure is created.
      
    Returns:
      embedding (np.array): The reduced states with shape (n_samples, n_components).
    """
    # Build color mapping from labels.
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}
    dot_colors = [color_dict[label] for label in labels]
    
    # Select dimensionality reduction algorithm.
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
    
    # Compute the embedding.
    embedding = dr_model.fit_transform(states)
    
    # If no axis is provided, create a new one.
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(embedding[:, 0], embedding[:, 1], c=dot_colors, alpha=0.7)
    ax.set_xlabel(f"{algorithm} Component 1")
    ax.set_ylabel(f"{algorithm} Component 2")
    ax.grid(True)
    
    patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
    ax.legend(handles=patches, fontsize='small')
    
    return embedding

#-------------------------------------------------------------------
# Multi-subplot loop: for each side and each feature.
#-------------------------------------------------------------------

# Define which sides and features to plot.
# For stim_id, we assume the string format is e.g.:
# "stone_mountain_pool_left__wood_forest_swing_right"
# In the splitting below, side=0 extracts the left part and side=1 extracts the right part.
# Then, feature indices (0,1,2) pick out (e.g.) material, scene, and the third feature respectively.
sides = [0, 1]       # 0 = left, 1 = right
features = [0, 1, 2] # e.g., 0 = material, 1 = scene, 2 = irrelevant
algorithm = 'TSNE'  # PCA, TSNE, UMAP

# Create a subplot grid.
fig, axes = plt.subplots(nrows=len(sides), ncols=len(features), figsize=(18, 10))

# Loop over sides and features.
# Here, we assume that df_data_sub is a DataFrame with a column 'stim_id'.
for i, side in enumerate(sides):
    for j, feat in enumerate(features):
        # Extract labels from 'stim_id'
        # First, split the stim_id by '__' to get left/right strings.
        # Then, split by '_' and extract the desired feature.
        labels = [label.split('__')[side] for label in df_data_sub['stim_id'].values]
        labels = [sub_label.split('_')[feat] for sub_label in labels]
        
        # Optional: set a title indicating which side and feature.
        title = f"{algorithm}: Side {side} - Feature {feat}"
        
        # Use dimensionality_reduction_plot to plot on the given axis.
        dimensionality_reduction_plot(all_hidden_states, labels, 
                                      algorithm=algorithm, ax=axes[i, j])
        axes[i, j].set_title(title)

plt.tight_layout()
plt.show()
 


# Create a new figure and axis.
fig, ax = plt.subplots(figsize=(8, 6))

# Use the dimensionality_reduction_plot function to plot the hidden states,
# using the cue_id labels to color the dots.
embedding = dimensionality_reduction_plot(
    states=all_hidden_states, 
    labels=df_data_sub['cue_id'].values, 
    algorithm=algorithm,  # PCA, TSNE, UMAP
    ax=ax
)

# Set a title for clarity.
ax.set_title(f"{algorithm} of Hidden States - Cue ID")
plt.show()
 



# #%% Apply PCA for Dimensionality Reduction


# # # labels = df_data_sub['cue_id'].values
# # # labels = df_data_sub['stim_id'].values
# labels = [label.split('__')[0].split('_')[3] for label in df_data_sub['stim_id'].values]
# unique_labels = np.unique(labels)
# unique_labels = ['left','right']

# # Create a color mapping: assign a unique color to each label.
# cmap = plt.cm.get_cmap('viridis', len(unique_labels))
# color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

# # Map each trial's label to a color.
# dot_colors = [color_dict[label] for label in labels]


# # Perform PCA to reduce hidden state dimensions (e.g., to 2 components)
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(all_hidden_states)

# # Plot the PCA results
# plt.figure(figsize=(8, 6))
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dot_colors, alpha=0.7)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("PCA of GRU Hidden States")
# plt.grid(True)

# # Create a legend for the colors.
# patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
# plt.legend(handles=patches)

# plt.show()


# #%% t-SNE



# # Apply t-SNE to reduce hidden state dimensionality to 2 components.
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# tsne_result = tsne.fit_transform(all_hidden_states)

# # Plot t-SNE results.
# plt.figure(figsize=(8, 6))
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=dot_colors, alpha=0.7)
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.title("t-SNE of GRU Hidden States")
# plt.grid(True)

# # Create a legend for the colors.
# patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
# # plt.legend(handles=patches)


# plt.show()


# #%% UMAP



# # Apply UMAP to reduce dimensionality to 2 components.
# umap_embedder = umap.UMAP(n_components=2, random_state=42)
# umap_result = umap_embedder.fit_transform(all_hidden_states)

# # Plot UMAP results.
# plt.figure(figsize=(8, 6))
# plt.scatter(umap_result[:, 0], umap_result[:, 1], c=dot_colors, alpha=0.7)
# plt.xlabel("UMAP Component 1")
# plt.ylabel("UMAP Component 2")
# plt.title("UMAP of GRU Hidden States")
# plt.grid(True)

# # Create a legend for the colors.
# patches = [mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels]
# plt.legend(handles=patches)

# plt.show()

