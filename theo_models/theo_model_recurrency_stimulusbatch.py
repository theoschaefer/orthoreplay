#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:21:22 2025

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
num_cues = 2     # Two possible cues (one-hot: length 2)
num_stimuli = 4  # Four possible stimuli (one-hot: length 4)
input_size = num_cues + num_stimuli  # 6-dimensional input
gru_hidden_size = 32  # Hidden size for GRU

# Reward functions: for each cue, mapping from stimulus -> reward
reward_functions = [
    {0: 1.0, 1: 0.5, 2: 0.3, 3: 0.2},  # For cue 0
    {0: 0.2, 1: 0.3, 2: 0.5, 3: 1.0}   # For cue 1
]

def generate_trial():
    """
    Generates a trial with a random cue and two candidate stimuli.
    Returns:
        cue: int in [0, 1]
        stim_indices: numpy array of 2 unique stimuli in [0, 3]
    """
    cue = np.random.randint(0, num_cues)
    stim_indices = np.random.choice(num_stimuli, 2, replace=False)
    return cue, stim_indices

def one_hot(indices, num_classes):
    """
    Convert tensor of indices to one-hot vectors.
    """
    return F.one_hot(indices, num_classes=num_classes).float()

def build_sequence(cue, stimulus):
    """
    Build a 2-step input sequence (shape: [seq_len, input_size]) for one candidate.
    
    Step 1: The cue is presented (cue one-hot; stimulus part is zeros).
    Step 2: The candidate stimulus is presented (stimulus one-hot; cue part is zeros).
    """
    # Step 1: cue presentation
    cue_tensor = one_hot(torch.tensor([cue]), num_cues)  # shape: (1, 2)
    zeros_stim = torch.zeros(1, num_stimuli)               # shape: (1, 4)
    step1 = torch.cat([cue_tensor, zeros_stim], dim=1)      # shape: (1, 6)
    
    # Step 2: candidate stimulus presentation
    zeros_cue = torch.zeros(1, num_cues)                   # shape: (1, 2)
    stim_tensor = one_hot(torch.tensor([stimulus]), num_stimuli)  # shape: (1, 4)
    step2 = torch.cat([zeros_cue, stim_tensor], dim=1)      # shape: (1, 6)
    
    # Sequence is two time steps (seq_len=2)
    sequence = torch.cat([step1, step2], dim=0)            # shape: (2, 6)
    return sequence

# -------------------
# GRU-based Recurrent Model
# -------------------
class RecurrentModel(nn.Module):
    """
    Processes a 2-step sequence (cue then candidate stimulus) using a GRU.
    The output from the final time step is mapped to a scalar score.
    """
    def __init__(self, input_size, gru_hidden_size):
        super(RecurrentModel, self).__init__()
        # Optional: a linear layer to transform input (here we use input_size directly)
        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_hidden_size, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, 1)  # Map final hidden state to a score
    
    def forward(self, x, hidden=None):
        """
        Args:
          x: Tensor of shape (batch_size, seq_len, input_size)
          hidden: optional initial hidden state
        Returns:
          scores: Tensor of shape (batch_size, 1) from the final time step.
          hidden: Final hidden state.
        """
        out, hidden = self.gru(x, hidden)
        # Use the GRU output from the final time step (i.e., index -1)
        final_output = out[:, -1, :]  # shape: (batch_size, gru_hidden_size)
        score = self.fc(final_output) # shape: (batch_size, 1)
        return score, hidden

# -------------------
# Training Loop for the Recurrent Model
# -------------------
def train_recurrent_model(num_trials=5000, lr=0.01):
    """
    Trains the GRU-based model using REINFORCE.
    For each trial, we build a two-step sequence for each candidate stimulus.
    Returns:
        model: Trained recurrent model.
        choices_history: List of trial details.
        performance_history: List (per trial) of 1 (correct) or 0 (incorrect).
    """
    model = RecurrentModel(input_size=input_size, gru_hidden_size=gru_hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    choices_history = []
    performance_history = []
    
    for trial in range(num_trials):
        cue, stim_indices = generate_trial()
        # Determine rewards for the two candidate stimuli
        rewards = [reward_functions[cue][int(s)] for s in stim_indices]
        best_choice = np.argmax(rewards)
        
        # For each candidate, build a 2-step sequence: [cue then candidate stimulus]
        seqs = []
        for s in stim_indices:
            seq = build_sequence(cue, s)  # shape: (2, 6)
            seqs.append(seq)
        # Stack sequences to form a batch: shape (batch_size=2, seq_len=2, input_size=6)
        input_batch = torch.stack(seqs, dim=0)
        
        # Forward pass: get scores for each candidate
        scores, _ = model(input_batch)  # scores: (2, 1)
        scores = scores.view(-1)        # shape: (2,)
        probs = F.softmax(scores, dim=0) # softmax over the two candidate scores
        
        # Sample an action
        m = torch.distributions.Categorical(probs)
        action = m.sample()  # returns 0 or 1
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
        
        performance_history.append(int(action.item() == best_choice))
    
    return model, choices_history, performance_history

# -------------------
# Helper: Analyze and Print Example Trials
# -------------------
def analyze_choices(choices_history):
    """
    Prints example trial details from the beginning, middle, and end.
    """
    total_trials = len(choices_history)
    example_indices = [0, total_trials // 2, total_trials - 1]
    
    for i in example_indices:
        ch = choices_history[i]
        print(f"Trial {ch['trial']}:")
        print(f"  Cue: {ch['cue']}")
        print(f"  Candidate Stimuli: {ch['stim_indices']}")
        print(f"  Chosen Action: {ch['action']} (Probabilities: {ch['probs']})")
        print(f"  Reward Received: {ch['reward']}")
        print(f"  Best Choice: {ch['best_choice']}")
        print(f"  Correct? {ch['correct']}\n")

# -------------------
# Main Script
# -------------------
if __name__ == "__main__":
    # Train the GRU-based recurrent model
    model, choices_history, performance_history = train_recurrent_model(num_trials=5000, lr=0.01)
    
    # Print exemplary trial details (beginning, middle, end)
    analyze_choices(choices_history)
    
    # Plot moving average performance over trials
    window = 200
    moving_avg = [np.mean(performance_history[max(0, i-window):i+1]) for i in range(len(performance_history))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg)
    plt.xlabel("Trial")
    plt.ylabel("Accuracy (Moving Average)")
    plt.title("GRU-based Model Performance Over Trials")
    plt.show()
