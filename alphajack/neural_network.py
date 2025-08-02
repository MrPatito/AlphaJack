"""
Component 2: Neural Network Architecture

This module defines the neural network for the AlphaJack AI, using PyTorch.
The network takes game state as input and outputs policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """Neural Network for AlphaJack AI"""
    
    def __init__(self, input_size: int, num_actions: int, dropout_rate: float = 0.2):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        
        # Shared hidden layers with batch normalization and dropout
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Policy head
        self.policy_head = nn.Linear(128, num_actions)
        
        # Value head
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Forward pass through the network"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure x has proper shape for batch normalization
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Shared layers with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Policy head
        policy_logits = self.policy_head(x)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Value head
        value = torch.tanh(self.value_head(x))
        
        return policy_probs, value

