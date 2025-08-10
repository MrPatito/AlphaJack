"""
Component 3: Monte Carlo Tree Search (MCTS)

This module implements the MCTS algorithm that uses the neural network
to guide its search for the best move in Blackjack.
"""

import math
import numpy as np
import torch
from typing import List, Optional, Dict
from copy import deepcopy

from .game_environment import BlackjackEnvironment, Action
from .neural_network import NeuralNetwork


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None, prior_prob: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior_probability = prior_prob
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Action, MCTSNode] = {}
        self.is_expanded = False
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return not self.is_expanded
    
    def get_value(self) -> float:
        """Get the average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_uct_score(self, c_param: float = 1.4) -> float:
        """Calculate the UCT (Upper Confidence Bound for Trees) score"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.get_value()
        exploration = c_param * self.prior_probability * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """Select the child with the highest UCT score"""
        return max(self.children.values(), key=lambda child: child.get_uct_score(c_param))
    
    def expand(self, env: BlackjackEnvironment, neural_net: NeuralNetwork) -> float:
        """Expand this node by adding children for all legal actions"""
        if self.is_expanded:
            return self.get_value()
        
        legal_actions = env.get_legal_actions()
        
        if not legal_actions:  # Terminal state
            self.is_expanded = True
            return 0.0
        
        # Get policy and value from neural network
        with torch.no_grad():
            # Set network to eval mode for inference to avoid BatchNorm issues
            was_training = neural_net.training
            neural_net.eval()
            
            state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            policy_probs, value = neural_net(state_tensor)
            
            # Restore original training mode
            if was_training:
                neural_net.train()
        
        # Create children for legal actions
        for action in legal_actions:
            action_prob = policy_probs[0][action.value].item()
            self.children[action] = MCTSNode(
                state=self.state,  # Will be updated when action is taken
                parent=self,
                action=action,
                prior_prob=action_prob
            )
        
        self.is_expanded = True
        return value.item()
    
    def backup(self, value: float):
        """Backup the value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(-value)  # Flip value for opponent's perspective
    
    def get_action_probabilities(self, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities based on visit counts"""
        action_probs = np.zeros(4)  # 4 possible actions
        
        if not self.children:
            return action_probs
        
        # Get visit counts for each action
        visit_counts = np.array([
            self.children.get(Action(i), MCTSNode(self.state)).visit_count 
            for i in range(4)
        ])
        
        if temperature == 0:
            # Deterministic selection
            best_action = np.argmax(visit_counts)
            action_probs[best_action] = 1.0
        else:
            # Temperature-based selection
            if visit_counts.sum() > 0:
                visit_counts = visit_counts ** (1.0 / temperature)
                action_probs = visit_counts / visit_counts.sum()
        
        return action_probs


class MCTS:
    """Monte Carlo Tree Search algorithm"""
    
    def __init__(self, neural_net: NeuralNetwork, num_simulations: int = 100, c_param: float = 1.4):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_param = c_param
    
    def search(self, env: BlackjackEnvironment) -> (Action, np.ndarray):
        """
        Run MCTS search and return the best action and action probabilities
        
        Args:
            env: The game environment
            
        Returns:
            best_action: The action with the highest visit count
            action_probs: Probability distribution over actions based on visit counts
        """
        root_state = env.get_state()
        root = MCTSNode(root_state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Create a copy of the environment for simulation
            env_copy = deepcopy(env)
            self._simulate(root, env_copy)
        
        # Get action probabilities from visit counts
        action_probs = root.get_action_probabilities(temperature=1.0)
        
        # Select best action (most visited)
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None, action_probs
        
        best_action_idx = np.argmax([
            root.children.get(action, MCTSNode(root_state)).visit_count 
            for action in legal_actions
        ])
        best_action = legal_actions[best_action_idx]
        
        return best_action, action_probs
    
    def _simulate(self, node: MCTSNode, env: BlackjackEnvironment):
        """Run a single MCTS simulation"""
        path = []
        current = node
        
        # Selection: traverse tree until we reach a leaf
        while not current.is_leaf():
            action = current.select_child(self.c_param).action
            path.append((current, action))
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Update current node
            if action in current.children:
                current = current.children[action]
                current.state = next_state
            
            if done:
                # Terminal state reached
                for node_action_pair in reversed(path):
                    node_action_pair[0].backup(reward)
                return
        
        # Expansion and evaluation
        value = current.expand(env, self.neural_net)
        
        # If terminal state
        if not current.children:
            # Use actual game reward if available
            legal_actions = env.get_legal_actions()
            if not legal_actions:  # Game is over
                _, final_reward, _ = env.step(Action.STAND)  # This should return the final reward
                value = final_reward
        
        # Backup
        current.backup(value)
    
    def get_action_probabilities(self, env: BlackjackEnvironment, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities for the current state"""
        _, action_probs = self.search(env)
        
        if temperature == 0:
            # Deterministic selection
            best_action_idx = np.argmax(action_probs)
            deterministic_probs = np.zeros_like(action_probs)
            deterministic_probs[best_action_idx] = 1.0
            return deterministic_probs
        else:
            # Apply temperature
            if action_probs.sum() > 0:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs = action_probs / action_probs.sum()
            return action_probs
