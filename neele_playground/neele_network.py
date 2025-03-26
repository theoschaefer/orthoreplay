# %% ===== Import(ant) stuff =====
import matplotlib.pyplot as plt
import torch
import torch.nn as nn #parent object for models
import torch.optim as optim
#import torchvision.transform as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Import helper functions from stateformation_get_input_output.py:
from stateformation_get_input_output import get_input_output_stateformation
from stateformation_get_input_output import get_active_trials_input_comb
from stateformation_get_input_output import rewardfun
csvpath = '/Users/neele/DocsVincent/github/orthoreplay/'  

# %% ===== Import Real Estate Data for Pytorch =====

inputs, outputs = get_input_output_stateformation(sub='sub508', csvpath = csvpath, isRecurrent=False) 

# %% ===== Define the neural network class =====

class nn_retreat(nn.Module):
    # INSTANTIATE
    def __init__(self, inodes: int, hnodes: int, onodes: int) -> None:
        super(nn_retreat, self).__init__()
        self.fc1 = nn.Linear(inodes, hnodes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hnodes, onodes)

    # DESCRIBE
    def __str__(self):
        desc = f"This neural network has {self.inodes} input, {self.hnodes} hidden, and {self.onodes} output nodes. "
        desc = desc + f"The learning rate is: {self.lr}. "
        desc = desc + f"The activation function used is sigmoid with random normally distributed weights."
        return desc
    
    # FORWARD
    def forward(self, x): # x: example/batch of examples
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # TRAIN 
    def train(model, dataloader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
