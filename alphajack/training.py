"""
Component 4: Training Loop

This module orchestrates the self-play and network training process for AlphaJack.
It manages the training data collection, neural network updates, and overall learning process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import random
from collections import deque
import pickle
import os

from .game_environment import BlackjackEnvironment, Action
from .neural_network import NeuralNetwork
from .mcts import MCTS


class TrainingData:
    """Container for training data"""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.action_probs: List[np.ndarray] = []
        self.outcomes: List[float] = []
    
    def add_data(self, state: np.ndarray, action_probs: np.ndarray, outcome: float):
        """Add a training example"""
        self.states.append(state.copy())
        self.action_probs.append(action_probs.copy())
        self.outcomes.append(outcome)
    
    def clear(self):
        """Clear all training data"""
        self.states.clear()
        self.action_probs.clear()
        self.outcomes.clear()
    
    def __len__(self):
        return len(self.states)


class AlphaJackTrainer:
    """Main trainer for the AlphaJack AI"""
    
    def __init__(self, 
                 state_size: int = 10,  # Updated for enhanced state representation
                 num_actions: int = 4,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 num_simulations: int = 100):
        
        self.state_size = state_size
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        
        # Initialize neural network
        self.neural_net = NeuralNetwork(state_size, num_actions)
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        
        # Initialize MCTS
        self.mcts = MCTS(self.neural_net, num_simulations)
        
        # Training data buffer
        self.training_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_stats = {
            'games_played': 0,
            'total_rewards': 0.0,
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'policy_entropy': [],
            'win_rates': [],
            'loss_rates': [],
            'push_rates': [],
            'average_rewards': [],
            'iteration_stats': [],
            'learning_rates': [],
            'temperatures': []
        }
    
    def self_play_game(self, env: BlackjackEnvironment, temperature: float = 1.0) -> Tuple[List, float]:
        """
        Play a single game using MCTS and collect training data
        
        Args:
            env: The game environment
            temperature: Temperature for action selection (higher = more exploration)
            
        Returns:
            game_data: List of (state, action_probs) tuples from the game
            final_reward: The final outcome of the game
        """
        env.reset()
        game_data = []
        
        while not env.game_over:
            # Get current state
            current_state = env.get_state()
            
            # Get action probabilities from MCTS
            action_probs = self.mcts.get_action_probabilities(env, temperature)
            
            # Store the state and action probabilities
            game_data.append((current_state.copy(), action_probs.copy()))
            
            # Select action based on probabilities
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            # Filter action probabilities for legal actions only
            legal_action_probs = np.zeros(self.num_actions)
            for action in legal_actions:
                legal_action_probs[action.value] = action_probs[action.value]
            
            # Normalize probabilities
            if legal_action_probs.sum() > 0:
                legal_action_probs = legal_action_probs / legal_action_probs.sum()
            else:
                # Uniform distribution over legal actions
                for action in legal_actions:
                    legal_action_probs[action.value] = 1.0 / len(legal_actions)
            
            # Sample action
            action_idx = np.random.choice(self.num_actions, p=legal_action_probs)
            selected_action = Action(action_idx)
            
            # Take action
            _, _, done = env.step(selected_action)
            
            if done:
                break
        
        # Get final reward
        final_reward = env._calculate_final_reward()
        
        return game_data, final_reward
    
    def collect_self_play_data(self, num_games: int, temperature: float = 1.0) -> TrainingData:
        """
        Collect training data from multiple self-play games
        
        Args:
            num_games: Number of games to play
            temperature: Temperature for action selection
            
        Returns:
            training_data: Collected training data
        """
        # Set network to eval mode for self-play to avoid BatchNorm issues
        self.neural_net.eval()
        
        training_data = TrainingData()
        env = BlackjackEnvironment()
        
        total_reward = 0.0
        
        for game_idx in range(num_games):
            # Play a game
            game_data, final_reward = self.self_play_game(env, temperature)
            total_reward += final_reward
            
            # Add game data to training set
            for state, action_probs in game_data:
                training_data.add_data(state, action_probs, final_reward)
            
            # Update statistics
            self.training_stats['games_played'] += 1
            
            if (game_idx + 1) % 100 == 0:
                avg_reward = total_reward / (game_idx + 1)
                print(f"Completed {game_idx + 1}/{num_games} games. Avg reward: {avg_reward:.3f}")
        
        # Update total rewards
        self.training_stats['total_rewards'] += total_reward
        
        return training_data
    
    def train_network(self, training_data: TrainingData, epochs: int = 10):
        """
        Train the neural network on collected data
        
        Args:
            training_data: Training data from self-play
            epochs: Number of training epochs
        """
        if len(training_data) == 0:
            return
        
        # Set network to training mode
        self.neural_net.train()
        
        # Convert data to tensors
        states = torch.tensor(np.array(training_data.states), dtype=torch.float32)
        action_probs = torch.tensor(np.array(training_data.action_probs), dtype=torch.float32)
        outcomes = torch.tensor(training_data.outcomes, dtype=torch.float32).unsqueeze(1)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, action_probs, outcomes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        epoch_entropies = []
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            epoch_entropy = 0.0
            num_batches = 0
            
            for batch_states, batch_action_probs, batch_outcomes in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted_probs, predicted_values = self.neural_net(batch_states)
                
                # Calculate losses
                policy_loss = self._calculate_policy_loss(predicted_probs, batch_action_probs)
                value_loss = self._calculate_value_loss(predicted_values, batch_outcomes)
                total_loss = policy_loss + value_loss
                
                # Calculate policy entropy
                batch_entropy = self._calculate_policy_entropy(predicted_probs)
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_entropy += batch_entropy
                num_batches += 1
            
            # Average losses for the epoch
            if num_batches > 0:
                epoch_policy_loss /= num_batches
                epoch_value_loss /= num_batches
                epoch_total_loss /= num_batches
                epoch_entropy /= num_batches
                
                self.training_stats['policy_losses'].append(epoch_policy_loss)
                self.training_stats['value_losses'].append(epoch_value_loss)
                self.training_stats['total_losses'].append(epoch_total_loss)
                epoch_entropies.append(epoch_entropy)
        
        # Store average entropy for this training iteration
        if epoch_entropies:
            avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
            self.training_stats['policy_entropy'].append(avg_entropy)
        
        print(f"Training completed. Final losses - Policy: {epoch_policy_loss:.4f}, "
              f"Value: {epoch_value_loss:.4f}, Total: {epoch_total_loss:.4f}")
    
    def _calculate_policy_loss(self, predicted_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """Calculate policy loss (cross-entropy)"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        predicted_probs = torch.clamp(predicted_probs, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        policy_loss = -torch.sum(target_probs * torch.log(predicted_probs), dim=1)
        return torch.mean(policy_loss)
    
    def _calculate_policy_entropy(self, predicted_probs: torch.Tensor) -> float:
        """Calculate average policy entropy"""
        epsilon = 1e-8
        predicted_probs = torch.clamp(predicted_probs, epsilon, 1 - epsilon)
        entropy = -torch.sum(predicted_probs * torch.log(predicted_probs), dim=1)
        return torch.mean(entropy).item()
    
    def _calculate_value_loss(self, predicted_values: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Calculate value loss (mean squared error)"""
        return nn.MSELoss()(predicted_values, target_values)
    
    def train(self, num_iterations: int, games_per_iteration: int = 100, epochs_per_iteration: int = 10):
        """
        Main training loop
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            epochs_per_iteration: Number of neural network training epochs per iteration
        """
        print(f"Starting AlphaJack training: {num_iterations} iterations")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Epochs per iteration: {epochs_per_iteration}")
        print(f"MCTS simulations: {self.num_simulations}")
        print("-" * 50)
        
        for iteration in range(num_iterations):
            print(f"\\nIteration {iteration + 1}/{num_iterations}")
            
            # Collect self-play data
            print("Collecting self-play data...")
            temperature = max(0.1, 1.0 - iteration * 0.1)  # Decay temperature over time
            self.training_stats['temperatures'].append(temperature)
            
            # Collect stats before training
            iter_start_games = self.training_stats['games_played']
            iter_start_rewards = self.training_stats['total_rewards']
            
            training_data = self.collect_self_play_data(games_per_iteration, temperature)
            
            # Calculate iteration statistics
            iter_games = self.training_stats['games_played'] - iter_start_games
            iter_rewards = self.training_stats['total_rewards'] - iter_start_rewards
            iter_avg_reward = iter_rewards / max(1, iter_games)
            
            # Train neural network
            print(f"Training neural network on {len(training_data)} samples...")
            self.train_network(training_data, epochs_per_iteration)
            
            # Update MCTS with new network
            self.mcts.neural_net = self.neural_net
            
            # Evaluate current performance
            eval_results = self.evaluate(num_games=100, verbose=False)
            
            # Store iteration statistics
            self.training_stats['win_rates'].append(eval_results['win_rate'])
            self.training_stats['loss_rates'].append(eval_results['loss_rate'])
            self.training_stats['push_rates'].append(eval_results['push_rate'])
            self.training_stats['average_rewards'].append(eval_results['avg_reward'])
            self.training_stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            iteration_stat = {
                'iteration': iteration + 1,
                'games_played': iter_games,
                'avg_reward': iter_avg_reward,
                'eval_win_rate': eval_results['win_rate'],
                'temperature': temperature
            }
            self.training_stats['iteration_stats'].append(iteration_stat)
            
            # Print statistics
            avg_reward = (self.training_stats['total_rewards'] / 
                         max(1, self.training_stats['games_played']))
            print(f"Average reward so far: {avg_reward:.3f}")
            print(f"Evaluation - Win: {eval_results['win_rate']:.1%}, "
                  f"Loss: {eval_results['loss_rate']:.1%}, "
                  f"Push: {eval_results['push_rate']:.1%}")
            
            # Save checkpoint
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration + 1}.pth")
        
        print("\\nTraining completed!")
        print(f"Total games played: {self.training_stats['games_played']}")
        print(f"Final average reward: {avg_reward:.3f}")
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'neural_net_state_dict': self.neural_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        if os.path.exists(filepath):
            # PyTorch 2.6 defaults to weights_only=True which can break loading
            # older checkpoints containing optimizer state or numpy scalars.
            # We explicitly set weights_only=False and map to current device.
            try:
                # Load on CPU to be device-agnostic; model/tensors can be moved later if needed
                checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
            except TypeError:
                # For older torch versions without weights_only arg
                checkpoint = torch.load(filepath, map_location="cpu")
            self.neural_net.load_state_dict(checkpoint['neural_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            
            # Update MCTS with loaded network
            self.mcts.neural_net = self.neural_net
            
            print(f"Checkpoint loaded: {filepath}")
        else:
            print(f"Checkpoint file not found: {filepath}")
    
    def evaluate(self, num_games: int = 100, verbose: bool = True) -> dict:
        """
        Evaluate the current policy
        
        Args:
            num_games: Number of games to play for evaluation
            
        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        # Set network to eval mode for evaluation
        self.neural_net.eval()
        
        env = BlackjackEnvironment()
        total_reward = 0.0
        wins = 0
        losses = 0
        pushes = 0
        
        if verbose:
            print(f"Evaluating policy on {num_games} games...")
        
        for game_idx in range(num_games):
            env.reset()
            
            while not env.game_over:
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    break
                
                # Use deterministic policy (temperature = 0)
                action_probs = self.mcts.get_action_probabilities(env, temperature=0.0)
                
                # Select best action
                best_action_idx = np.argmax([
                    action_probs[action.value] for action in legal_actions
                ])
                selected_action = legal_actions[best_action_idx]
                
                # Take action
                _, _, done = env.step(selected_action)
                if done:
                    break
            
            # Get final reward
            reward = env._calculate_final_reward()
            total_reward += reward
            
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                pushes += 1
        
        avg_reward = total_reward / num_games
        win_rate = wins / num_games
        loss_rate = losses / num_games
        push_rate = pushes / num_games
        
        results = {
            'avg_reward': avg_reward,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'push_rate': push_rate,
            'total_games': num_games
        }
        
        if verbose:
            print(f"Evaluation Results:")
            print(f"Average Reward: {avg_reward:.3f}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Loss Rate: {loss_rate:.1%}")
            print(f"Push Rate: {push_rate:.1%}")
        
        return results
