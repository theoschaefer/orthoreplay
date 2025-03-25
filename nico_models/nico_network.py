# nico's main runner file for the network

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
# Import helper functions from stateformation_get_input_output.py:
from stateformation_get_input_output import get_input_output_stateformation
from stateformation_get_input_output import get_active_trials_input_comb
from stateformation_get_input_output import rewardfun

# Load the data
csvpath = '~/code/nn_orthoreplay/' #  
# Call the fucntion the brings back input and outputs:
inputs, targets = get_input_output_stateformation(sub='sub508',csvpath = csvpath, isRecurrent=True)   
inputs = np.transpose(inputs, (1, 2, 0))
inputs = inputs[np.newaxis, :, :, :]
targets = np.transpose(targets, (1, 0))
targets = targets[np.newaxis, :, :]

inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)


# Define hyperparameters
output_dim = targets.shape[1] # 2 output
batch_size = inputs.shape[0] # trialwise updating 
n_steps = inputs.shape[1] # 2 time steps
input_dim = inputs.shape[2] # 16 features
n_trials = inputs.shape[3] # 256 trials

hidden_rnn_dim = 50
hidden_fc_dim = 50
hidden_rnn_activation = 'tanh'
hidden_fc_activation = 'relu'
output_activation = 'softmax' 
learning_rate = 0.01

# Define network
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_rnn_dim, hidden_fc_dim, output_dim):
        super(SimpleRNN, self).__init__()
        # Recurrent hidden layer
        self.rnn = nn.RNN(input_dim, hidden_rnn_dim, nonlinearity=hidden_rnn_activation, bias=True, batch_first=True)
        # Fully connected hidden layer
        self.fc_hidden = nn.Linear(hidden_rnn_dim, hidden_fc_dim, bias=True)
        # Output layer
        self.output = nn.Linear(hidden_fc_dim, output_dim, bias=True)
       
        # Dynamically set the activation function for the fully connected hidden layer
        if hidden_fc_activation == 'relu':
            self.hidden_activation = nn.ReLU()
        elif hidden_fc_activation == 'tanh':
            self.hidden_activation = nn.Tanh()
        elif hidden_fc_activation == 'sigmoid':
            self.hidden_activation = nn.Sigmoid()
        elif hidden_fc_activation == 'linear':
            self.hidden_activation = nn.Identity()  # No activation (linear output)
        else:
            raise ValueError(f"Unsupported activation function: {hidden_fc_activation}")
        
        # Dynamically set the activation function for the output layer
        if output_activation == 'relu':
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)  # Softmax for classification
        elif output_activation == 'linear':
            self.output_activation = nn.Identity()  # No activation (linear output)
        else:
            raise ValueError(f"Unsupported activation function: {output_activation}")

    def forward(self, x):
        # Pass through RNN layer
        rnn_out, _ = self.rnn(x)  # x should have shape (batch_size, n_steps, input_dim)
        # Take the output of the last time step
        rnn_last_out = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_rnn_dim)
        # Pass through fully connected hidden layer
        fc_hidden_out = self.hidden_activation(self.fc_hidden(rnn_last_out))
        # Pass through output layer
        output = self.output_activation(self.output(fc_hidden_out))
        return output


model = SimpleRNN(input_dim, hidden_rnn_dim, hidden_fc_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
losses = []  # Store the loss values
accuracies = []
model.train()
for crep in range(2): 
    for ctrial in range(n_trials):
        trial_input = inputs[:, :, :,ctrial]
        trial_target = targets[:, :, ctrial]     
        optimizer.zero_grad()
        output = model(trial_input)
        loss = criterion(output, trial_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # Store the loss value
        # Determine which output node has the maximum value
        predicted_node = torch.argmax(output, dim=1)  # Index of the maximum output node
        target_node = torch.argmax(trial_target, dim=1)  # Index of the maximum target node
        
        # Compute accuracy for this trial
        accuracy = (predicted_node == target_node).float().mean().item()
        accuracies.append(accuracy)
        
        print('Epoch:', ctrial, 'Loss:', loss.item())

# Compute running averages
window_size = 15

# Function to compute running average
def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Compute running averages for losses and accuracies
running_losses = running_average(losses, window_size)
running_accuracies = running_average(accuracies, window_size)

# Plot the running averages
plt.figure(figsize=(12, 5))

# Plot running average of losses
plt.subplot(1, 2, 1)
plt.plot(range(len(running_losses)), running_losses, label='Running Avg Loss', color='blue')
plt.xlabel('Trial')
plt.ylabel('Loss')
plt.title('Running Average of Losses')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(running_accuracies)), running_accuracies, label='Running Avg Accuracy', color='orange')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Running Average of Accuracies')
plt.legend()
plt.show()


plt.plot(range(n_trials*2), losses, label='Training Loss')
plt.show()

plt.plot(range(n_trials*2), accuracies, label='Training Loss')
plt.show()

activations['rnn'] = np.array(activations['rnn'])
activations['fc_hidden'] = np.array(activations['fc_hidden'])

