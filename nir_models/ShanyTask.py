# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:43:53 2025

@author: moneta
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
# Add your script directory to the Python path
sys.path.append('C:/Users/moneta/work/retreat/orthoreplay/nir_models/')
from stateformation_get_input_output import get_input_output_stateformation
from stateformation_get_input_output import get_active_trials_input_comb
from stateformation_get_input_output import rewardfun
csvpath = 'C:/Users/moneta/work/retreat/orthoreplay/nir_models/' #  
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

class DecisionNet(nn.Module):
    def __init__(self, 
                 layers=['lstm', 'linear', 'linear'], 
                 inputs=[None, None, None], 
                 output_size=2):
        super(DecisionNet, self).__init__()
        self.layers = nn.ModuleList()
        for l in range(len(layers)):
            size_in = inputs[l]
            size_out = inputs[l + 1] if l < (len(inputs) - 1) else output_size
            if layers[l] == 'lstm':
                self.layers.append(nn.LSTM(size_in, size_out, batch_first=True))  # LSTM for sequential input
            elif layers[l] == 'gru':
                self.layers.append(nn.GRU(size_in, size_out, batch_first=True))  # GRU for sequential input
            elif layers[l] == 'linear':
                self.layers.append(nn.Linear(size_in, size_out))
            elif layers[l] == 'rnn':
                self.layers.append(nn.RNN(size_in, size_out, batch_first=True))  # RNN for sequential input
           

    def forward(self, xs, hidden=None, separate_sequences=True):
            """
            Forward pass through the network.
            xs is a list of inputs: either one or two.
            separate_sequences is a flag to reset hidden states between inputs.
            """
            def process_input(x, hidden):
                activations = []
                for l, layer in enumerate(self.layers):
                    if isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
                        x, hidden = layer(x, hidden)  # RNN/LSTM/GRU output with optional hidden state
                        x = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # Use the last hidden state as the output
                    elif isinstance(layer, nn.Linear):
                        x = F.relu(layer(x))  # Apply ReLU activation for linear layers
                    if l != (len(self.layers) - 1):  # no need as its only 2 and we softmax it anyway
                        activations.append(x.detach().numpy())
                return x, activations
            
            if len(xs) == 1:
                # Process single input
                output, activations = process_input(xs[0], hidden)
            elif len(xs) == 2:
                # Process both inputs through the network
                output1, activations1 = process_input(xs[0], hidden)
                if separate_sequences:
                    hidden = None  # Reset hidden state for the second input
                output2, activations2 = process_input(xs[1], hidden)

                # Concatenate the outputs
                output = torch.cat((output1, output2), dim=-1)
                activations = activations1 + activations2
            
            # Apply softmax to the final output
            final_output = F.softmax(output, dim=-1)
            
            return final_output, activations

'''
Input is in three steps:
    (1) cue, full everything and not freeze anything
    (2) stim 1 with sequentiality so hidden=None
    (3) stim 2 with sequentiality so hidden=None
    (4) back propogate the reward
'''

    
    
    


# Task setup
num_stimuli = 4  # Total possible stimuli
num_cues = 2      # Two possible context cues
input_size=num_stimuli+num_cues
hidden_size = 32  # Hidden layer size
output_size = 2   # Two possible choices
learning_rate=0.01
reward_noise=0.01
epsilon=None # if None then by softmax probabilities
# Create the model, loss function, and optimizer

model = DecisionNet(layers=['lstm', 'linear'], 
                    inputs=[input_size,hidden_size], 
                    output_size=output_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Reward functions for each cue
reward_functions = [
    {0: 1.0, 1: 0.5, 2: 0.3, 3: 0.2},  # Cue 0 reward mapping for stimuli 0-3
    {0: 0.2, 1: 0.3, 2: 0.5, 3: 1.0}   # Cue 1 reward mapping for stimuli 0-3
]


def generate_trial():
    """
    Generates a single trial with a random cue and a random pair of stimuli.
    - The cue represents the task context.
    - The two stimuli are potential choices the network must select from.
    """
    cue = np.random.randint(0, num_cues)  # Randomly select a cue (0 or 1)
    stim_indices = np.random.choice(num_stimuli, 2, replace=False)  # Pick 2 unique stimuli
    return cue, stim_indices

def run_episode():
    """
    Runs one decision-making trial where the input is processed in two sequential steps:
    1. The network first receives a context cue.
    2. The network then receives two possible stimuli and must decide which one to select.
    """
    
    # Step 1: Process cue input
    cue, stim_indices = generate_trial()  # Get random cue and stimuli
    
    cue_one_hot = np.zeros(input_size)  
    cue_one_hot[cue] = 1 
    cue_tensor = torch.tensor(cue_one_hot, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Convert to tensor and add batch and sequence dimensions
    
    # Process the cue through the model
    _, (hn, cn), _ = model(cue_tensor)  # Get the hidden state after processing the cue
    
    # Step 2: Process stimulus input
    stim_one_hot = np.zeros(num_cues + num_stimuli) 
    stim_one_hot[stim_indices + 2] = 1 
    stim_tensor = torch.tensor(stim_one_hot, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Convert to tensor
    
    # Process the stimuli through the model using the hidden state from the cue
    action_probs, (hn, cn), activities = model(stim_tensor, (hn, cn))  # Use the hidden state from the cue
    
    # Action selection using softmax probabilities
    # Epsilon-greedy action selection
    if epsilon is not None:
        if np.random.rand() < epsilon:
            action = torch.tensor(np.random.choice([0, 1]))  # Explore: choose a random action
        else:
            action = torch.argmax(action_probs)  # Exploit: choose the action with the highest probability
        probout=action_probs.log().squeeze()[action]
    else:
        action_dist = torch.distributions.Categorical(action_probs)  # Create a categorical distribution
        action = action_dist.sample()  # Sample an action (random choice based on probabilities)
        probout = action_dist.log_prob(action)
    # Determine reward based on chosen stimulus
    chosen_stim = stim_indices[action.item()]  # Get selected stimulus index
    reward = reward_functions[cue][chosen_stim]  # Retrieve reward based on cue-dependent function and selected stimulus
    reward = np.random.normal(reward,reward_noise)
    # Save outputs for each trial
    max_reward = max(reward_functions[cue][stim_indices[0]],
                     reward_functions[cue][stim_indices[1]])  # Track the maximum reward for the current cue
    
    return probout, reward, max_reward, activities, action_probs.detach().numpy(), stim_indices  # Return log probability and received reward


def train_model(num_episodes=2000):
    """
    Trains the model using policy gradient reinforcement learning.
    - Runs multiple episodes (trials).
    - Updates the model to reinforce actions that lead to higher rewards.
    """
    # Lists to store output for later analysis
    losses = []
    rewards = []
    max_rewards = []
    hidden_activations = []
    output_activations = []
    accuracy = []
    stims=[]
    
    for episode in range(num_episodes):
        optimizer.zero_grad()  # Reset gradients before each episode
        
        log_prob, reward, max_reward, activities, action_probs,stim_indices = run_episode()  # Run a single decision-making trial
        
        # Save data for analysis
        losses.append(-log_prob.item())  # Record the loss for each episode
        rewards.append(reward)  # Record the reward for each episode
        max_rewards.append(max_reward)  # Record the maximum reward for each episode
        accuracy.append(reward==max_reward)
        hidden_activations.append(activities)  # Store the hidden state activations
        output_activations.append(action_probs)  # Store the output layer activations (softmax probs)
        stims.append(stim_indices)
        
        loss = -log_prob * reward  # Compute policy gradient loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item()} Reward: {reward}")  # Print progress every 100 episodes
    
    # Optionally, save the recorded values to a file or return them
    return losses, rewards, max_rewards, hidden_activations, output_activations,stims,accuracy

# Example of training the model and saving the outputs
losses, rewards, max_rewards, hidden_activations, output_activations,stims,accuracy = train_model()


# plot

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Assuming you have already run the training and have the following data:
# losses, rewards, max_rewards, hidden_activations, output_activations, stims

# Calculate the difference between max rewards and rewards
reward_diffs = np.array(max_rewards) - np.array(rewards)

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(14, 3))

# Subplot 1: Losses over time
axs[0].plot(losses, label='Loss')
axs[0].set_title('Losses Over Time')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Subplot 2: Smoothed max-reward minus rewards over time
smoothed_reward_diffs = gaussian_filter1d(reward_diffs, sigma=2)
axs[1].plot(smoothed_reward_diffs, label='Smoothed Reward Difference', color='orange')
axs[1].set_title('Smoothed Max-Reward Minus Obtained Rewards Over Time')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Normalized Reward Difference')
axs[1].legend()

# Subplot 3: Smoothed max-reward minus rewards over time
smoothed_accuracy = gaussian_filter1d(accuracy, sigma=2)
axs[2].plot(smoothed_accuracy, label='Smoothed accuracy', color='orange')
axs[2].set_title('Accuracy')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('acc')
axs[2].legend()

# Subplot 4: Output activation (mean over episodes) as bar plot with SD
mean_output_activations = np.mean(output_activations, axis=0).squeeze()
std_output_activations = np.std(output_activations, axis=0).squeeze()
axs[3].bar([0,1], mean_output_activations, yerr=std_output_activations, color='green', alpha=0.7, label='Mean Output Activation',capsize=5,error_kw={'elinewidth': 2, 'capthick': 2} )
axs[3].set_title('Output Activation Over Time')
axs[3].set_xlabel('Action')
axs[3].set_ylabel('Proability')
axs[3].legend()  

plt.tight_layout()
plt.show()

