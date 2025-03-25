#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:10:07 2025

@author: theo schaefer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Task Setup
# -------------------
num_cues = 2
num_stimuli = 4
# Total input dimension: cue (2) + left stimulus (4) + right stimulus (4) = 10.
input_size = num_cues + 2 * num_stimuli  
gru_hidden_size = 32

# Reward functions: For each cue, each stimulus has a reward.
reward_functions = [
    {0: 1.0, 1: 0.5, 2: 0.3, 3: 0.2},  # Cue 0
    {0: 0.2, 1: 0.3, 2: 0.5, 3: 1.0}   # Cue 1
]

def generate_trial():
    """
    Generates a trial with a cue and two candidate stimuli (left and right).
    """
    cue = np.random.randint(0, num_cues)
    stim_indices = np.random.choice(num_stimuli, 2, replace=False)
    return cue, stim_indices

def one_hot(index, num_classes):
    vec = torch.zeros(num_classes)
    vec[index] = 1.0
    return vec

def build_sequence(cue, left_stim, right_stim):
    """
    Build a 2-step sequence:
    
    Time step 1: Cue is provided.
      Input: [cue one-hot (length 2), zeros (length 4 for left), zeros (length 4 for right)]
    
    Time step 2: Both stimuli are provided simultaneously.
      Input: [zeros (2), left stimulus one-hot (4), right stimulus one-hot (4)]
      
    Returns:
        Tensor of shape (1, 2, 10) where 1 is batch size.
    """
    # Time step 1: cue only.
    cue_vec = one_hot(cue, num_cues)          # Shape: (2,)
    zeros_stim = torch.zeros(num_stimuli)       # Shape: (4,)
    ts1 = torch.cat([cue_vec, zeros_stim, zeros_stim], dim=0)  # (10,)
    
    # Time step 2: both stimuli.
    zeros_cue = torch.zeros(num_cues)           # (2,)
    left_vec = one_hot(left_stim, num_stimuli)    # (4,)
    right_vec = one_hot(right_stim, num_stimuli)  # (4,)
    ts2 = torch.cat([zeros_cue, left_vec, right_vec], dim=0)   # (10,)
    
    sequence = torch.stack([ts1, ts2], dim=0)  # Shape: (2, 10)
    sequence = sequence.unsqueeze(0)           # Add batch dimension -> (1, 2, 10)
    return sequence

# -------------------
# Recurrent Model with an Additional Layer
# -------------------
class RecurrentModel(nn.Module):
    def __init__(self, input_size, gru_hidden_size):
        super(RecurrentModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_hidden_size, batch_first=True)
        # Additional fully connected layer to help learn complex interactions.
        self.fc1 = nn.Linear(gru_hidden_size, gru_hidden_size)
        # Final output layer producing 2 scores (left and right).
        self.fc2 = nn.Linear(gru_hidden_size, 2)
    
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_size) with batch_size=1 in our case.
        out, hidden = self.gru(x, hidden)
        final_output = out[:, -1, :]   # Take the final time step's output.
        x = F.relu(self.fc1(final_output))
        scores = self.fc2(x)           # Map to 2 scores.
        return scores, hidden

# -------------------
# Training Loop
# -------------------
def train_model(num_trials=5000, lr=0.001):
    model = RecurrentModel(input_size, gru_hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    choices_history = []
    performance_history = []
    
    for trial in range(num_trials):
        cue, stim_indices = generate_trial()
        left_stim, right_stim = stim_indices  # left and right candidates
        
        # Determine rewards for left and right based on the cue.
        rewards = [
            reward_functions[cue][int(left_stim)],
            reward_functions[cue][int(right_stim)]
        ]
        best_choice = np.argmax(rewards)
        
        # Build the 2-step sequence input.
        seq = build_sequence(cue, left_stim, right_stim)
        scores, _ = model(seq)         # scores: (1, 2)
        scores = scores.squeeze(0)     # Shape: (2,)
        probs = F.softmax(scores, dim=0)
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()  # 0 = left, 1 = right
        chosen_reward = rewards[action.item()]
        
        choices_history.append({
            'trial': trial,
            'cue': cue,
            'stim_indices': stim_indices.tolist(),
            'action': action.item(),
            'reward': chosen_reward,
            'best_choice': best_choice,
            'correct': int(action.item() == best_choice),
            'probs': probs.detach().numpy().tolist()
        })
        
        loss = -m.log_prob(action) * chosen_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        performance_history.append(int(action.item() == best_choice))
    
    return model, choices_history, performance_history

def analyze_choices(choices_history):
    total_trials = len(choices_history)
    example_indices = [0, total_trials // 2, total_trials - 1]
    for i in example_indices:
        ch = choices_history[i]
        print(f"Trial {ch['trial']}:")
        print(f"  Cue: {ch['cue']}")
        print(f"  Left Stimulus: {ch['stim_indices'][0]}, Right Stimulus: {ch['stim_indices'][1]}")
        print(f"  Chosen Action: {ch['action']} (Probabilities: {ch['probs']})")
        print(f"  Reward Received: {ch['reward']}")
        print(f"  Best Choice: {ch['best_choice']}")
        print(f"  Correct? {ch['correct']}\n")

if __name__ == "__main__":
    model, choices_history, performance_history = train_model(num_trials=5000, lr=0.001)
    analyze_choices(choices_history)
    
    # Plot moving average performance.
    window = 100
    moving_avg = [np.mean(performance_history[max(0, i-window):i+1]) for i in range(len(performance_history))]
    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg)
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Average)")
    plt.title("Model Performance with Extra Layer")
    plt.show()
