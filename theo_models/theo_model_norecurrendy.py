#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:52:32 2025

@author: schaefer
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
num_cues = 2     # Two possible cues
num_stimuli = 4  # Four possible stimuli
hidden_size = 32 # Hidden layer size

# Reward functions: for each cue, a mapping from stimulus -> reward
reward_functions = [
    {0: 1.0, 1: 0.5, 2: 0.3, 3: 0.2},  # Cue=0
    {0: 0.2, 1: 0.3, 2: 0.5, 3: 1.0}   # Cue=1
]

def generate_trial():
    """
    Returns:
      cue: int in [0, 1]
      stim_indices: array of 2 distinct stimuli in [0, 3]
    """
    cue = np.random.randint(0, num_cues)
    stim_indices = np.random.choice(num_stimuli, 2, replace=False)
    return cue, stim_indices

def one_hot(indices, num_classes):
    """Convert a 1D tensor of indices to one-hot vectors."""
    return F.one_hot(indices, num_classes=num_classes).float()

# -------------------
# Binary Network
# -------------------
class CueStimulusNetwork(nn.Module):
    """
    Concatenates one-hot(cue) and one-hot(stimulus) to form a 6D input
    (2 dims for cue + 4 dims for stimulus = 6).
    Then uses a feedforward network to produce a scalar score.
    """
    def __init__(self, num_cues, num_stimuli, hidden_size):
        super(CueStimulusNetwork, self).__init__()
        input_size = num_cues + num_stimuli  # 2 + 4 = 6
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, cue_onehot, stim_onehot):
        """
        Args:
          cue_onehot: shape (batch_size, num_cues)
          stim_onehot: shape (batch_size, num_stimuli)
        Returns:
          A tensor of shape (batch_size, 1) giving a scalar score.
        """
        x = torch.cat([cue_onehot, stim_onehot], dim=1)
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score

# -------------------
# Training
# -------------------
def train_model(num_trials=5000, lr=0.01):
    """
    Trains the binary network with REINFORCE on the simple task.
    Returns:
      model: the trained network
      choices_history: a list of dicts with info for each trial
      performance_history: 1 if correct, 0 otherwise (per trial)
    """
    model = CueStimulusNetwork(num_cues, num_stimuli, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    choices_history = []
    performance_history = []

    for trial in range(num_trials):
        cue, stim_indices = generate_trial()
        # Reward for each of the 2 candidate stimuli
        rewards = [reward_functions[cue][int(s)] for s in stim_indices]
        best_choice = np.argmax(rewards)

        # Prepare model input
        cue_tensor = torch.tensor([cue, cue], dtype=torch.long)      # shape (2,)
        stim_tensor = torch.tensor(stim_indices, dtype=torch.long)   # shape (2,)
        cue_onehot = one_hot(cue_tensor, num_cues)                   # shape (2, 2)
        stim_onehot = one_hot(stim_tensor, num_stimuli)              # shape (2, 4)

        # Forward pass: get scores for the two candidates
        scores = model(cue_onehot, stim_onehot)  # shape (2, 1)
        probs = F.softmax(scores.view(-1), dim=0)  # shape (2,)

        # Sample an action
        m = torch.distributions.Categorical(probs)
        action = m.sample()  # 0 or 1
        chosen_reward = rewards[action.item()]

        # Record trial info
        choices_history.append({
            'trial': trial,
            'cue': cue,
            'stim_indices': stim_indices,
            'action': action.item(),
            'reward': chosen_reward,
            'best_choice': best_choice,
            'correct': int(action.item() == best_choice),
            'probs': probs.detach().numpy().tolist()
        })

        # REINFORCE loss
        loss = -m.log_prob(action) * chosen_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track performance
        performance_history.append(int(action.item() == best_choice))

    return model, choices_history, performance_history

# -------------------
# Helper: Print Example Trials
# -------------------
def analyze_choices(choices_history):
    """
    Prints some example trials: beginning, middle, and last.
    """
    total_trials = len(choices_history)
    example_indices = [0, total_trials // 2, total_trials - 1]
    
    for i in example_indices:
        ch = choices_history[i]
        print(f"Trial {ch['trial']}")
        print(f"  Cue: {ch['cue']}")
        print(f"  Candidate Stimuli: {ch['stim_indices']}")
        print(f"  Action chosen: {ch['action']} (Prob: {ch['probs']})")
        print(f"  Reward received: {ch['reward']}")
        print(f"  Best choice would be: {ch['best_choice']}")
        print(f"  Correct? {ch['correct']}\n")

# -------------------
# Main Script
# -------------------
if __name__ == "__main__":
    # Train the binary model
    model, choices_history, performance_history = train_model(num_trials=5000, lr=0.01)
    
    # Print sample trials
    analyze_choices(choices_history)
    
    # Plot moving average performance
    window = 100
    moving_avg = [
        np.mean(performance_history[max(0, i-window):i+1]) 
        for i in range(len(performance_history))
    ]
    plt.plot(moving_avg)
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Average)")
    plt.title("Binary Model Performance")
    plt.show()
