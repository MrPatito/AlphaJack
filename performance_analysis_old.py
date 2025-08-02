"""
performance_analysis.py

This script extracts and visualizes key performance metrics from training logs,
such as training loss, winning rate, value loss, policy loss, policy entropy, and more.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
from typing import Dict, List


def load_training_stats(checkpoint_path: str):
    """Load training statistics from a checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        print("Make sure you have trained a model first using:")
        print("  python3 main.py --mode train")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        # Load with weights_only=False to handle numpy arrays in the checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    if 'training_stats' not in checkpoint:
        print("Error: No training statistics found in checkpoint!")
        print("This checkpoint may be from an incomplete training session.")
        return None
    
    return checkpoint['training_stats']


def print_training_summary(stats: Dict) -> None:
    """Print a comprehensive summary of training statistics"""
    print("\n" + "="*80)
    print("ALPHAJACK TRAINING PERFORMANCE SUMMARY")
    print("="*80)
    
    # Basic training info
    total_games = stats.get('games_played', 0)
    total_iterations = len(stats.get('policy_losses', []))
    print(f"Total Training Iterations: {total_iterations}")
    print(f"Total Games Played: {total_games:,}")
    
    if total_iterations == 0:
        print("No training data found!")
        return
    
    # Calculate overall average reward if available
    total_rewards = stats.get('total_rewards', 0)
    if total_games > 0:
        overall_avg_reward = total_rewards / total_games
        print(f"Overall Average Reward: {overall_avg_reward:.4f}")
    
    # Latest performance metrics (if available)
    win_rates = stats.get('win_rates', [])
    loss_rates = stats.get('loss_rates', [])
    push_rates = stats.get('push_rates', [])
    average_rewards = stats.get('average_rewards', [])
    
    if win_rates or loss_rates or push_rates or average_rewards:
        print("\n" + "-"*50)
        print("LATEST PERFORMANCE METRICS")
        print("-"*50)
        
        if win_rates:
            print(f"Win Rate: {win_rates[-1]:.1%}")
        if loss_rates:
            print(f"Loss Rate: {loss_rates[-1]:.1%}")
        if push_rates:
            print(f"Push Rate: {push_rates[-1]:.1%}")
        if average_rewards:
            print(f"Average Reward: {average_rewards[-1]:.4f}")
    else:
        print("\n" + "-"*50)
        print("PERFORMANCE METRICS")
        print("-"*50)
        print("Note: Detailed performance metrics (win rates, etc.) not available in this checkpoint.")
        print("This may be from an earlier version or incomplete training session.")
    
    # Loss metrics
    print("\n" + "-"*50)
    print("TRAINING LOSS METRICS")
    print("-"*50)
    
    policy_losses = stats.get('policy_losses', [])
    value_losses = stats.get('value_losses', [])
    total_losses = stats.get('total_losses', [])
    policy_entropy = stats.get('policy_entropy', [])
    
    if policy_losses:
        print(f"Latest Policy Loss: {policy_losses[-1]:.6f}")
        print(f"Average Policy Loss: {np.mean(policy_losses):.6f}")
    if value_losses:
        print(f"Latest Value Loss: {value_losses[-1]:.6f}")
        print(f"Average Value Loss: {np.mean(value_losses):.6f}")
    if total_losses:
        print(f"Latest Total Loss: {total_losses[-1]:.6f}")
        print(f"Average Total Loss: {np.mean(total_losses):.6f}")
    if policy_entropy:
        print(f"Latest Policy Entropy: {policy_entropy[-1]:.6f}")
        print(f"Average Policy Entropy: {np.mean(policy_entropy):.6f}")
    
    # Training progress analysis
    if len(total_losses) >= 10:
        print("\n" + "-"*50)
        print("TRAINING PROGRESS ANALYSIS")
        print("-"*50)
        
        # Loss trend analysis
        recent_losses = total_losses[-10:]
        early_losses = total_losses[:10]
        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        loss_reduction = early_avg - recent_avg
        loss_reduction_pct = (loss_reduction / early_avg) * 100 if early_avg != 0 else 0
        print(f"Loss Reduction: {loss_reduction:.6f} ({loss_reduction_pct:+.1f}%)")
        print(f"Early Training Loss: {early_avg:.6f} → Recent Loss: {recent_avg:.6f}")
        
        # Performance trend (if available)
        if win_rates and len(win_rates) >= 2:
            initial_win_rate = win_rates[0]
            latest_win_rate = win_rates[-1]
            win_rate_improvement = latest_win_rate - initial_win_rate
            print(f"Win Rate Improvement: {win_rate_improvement:+.1%} ({initial_win_rate:.1%} → {latest_win_rate:.1%})")
        
        if average_rewards and len(average_rewards) >= 2:
            initial_avg_reward = average_rewards[0]
            latest_avg_reward = average_rewards[-1]
            reward_improvement = latest_avg_reward - initial_avg_reward
            print(f"Reward Improvement: {reward_improvement:+.4f} ({initial_avg_reward:.4f} → {latest_avg_reward:.4f})")
    
    print("\n" + "="*80)


def plot_comprehensive_analysis(stats: Dict) -> None:
    """Create comprehensive visualization of all training metrics"""
    # Get available data
    total_losses = stats.get('total_losses', [])
    policy_losses = stats.get('policy_losses', [])
    value_losses = stats.get('value_losses', [])
    policy_entropy = stats.get('policy_entropy', [])
    win_rates = stats.get('win_rates', [])
    loss_rates = stats.get('loss_rates', [])
    push_rates = stats.get('push_rates', [])
    average_rewards = stats.get('average_rewards', [])
    learning_rates = stats.get('learning_rates', [])
    temperatures = stats.get('temperatures', [])
    
    # Check if there's data to plot
    if not any([total_losses, policy_losses, value_losses, policy_entropy, 
                win_rates, loss_rates, push_rates, average_rewards,
                learning_rates, temperatures]):
        print("No data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('AlphaJack Training Performance Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    ax = axes.flatten()
    
    # 1. Total Loss over time
    if total_losses:
        iterations = range(1, len(total_losses) + 1)
        ax[0].plot(iterations, total_losses, 'r-', linewidth=2, alpha=0.8)
        ax[0].set_title('Total Training Loss', fontweight='bold')
        ax[0].set_xlabel('Training Iteration')
        ax[0].set_ylabel('Loss')
        ax[0].grid(True, alpha=0.3)
    else:
        ax[0].text(0.5, 0.5, 'Total Loss\nData Not Available', 
                  ha='center', va='center', transform=ax[0].transAxes, fontsize=12)
        ax[0].set_title('Total Training Loss', fontweight='bold')
    
    # 2. Policy vs Value Loss
    if policy_losses and value_losses:
        iterations = range(1, len(policy_losses) + 1)
        ax[1].plot(iterations, policy_losses, 'b-', label='Policy Loss', linewidth=2)
        ax[1].plot(iterations, value_losses, 'orange', label='Value Loss', linewidth=2)
        ax[1].set_title('Policy vs Value Loss', fontweight='bold')
        ax[1].set_xlabel('Training Iteration')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    else:
        ax[1].text(0.5, 0.5, 'Policy/Value Loss\nData Not Available', 
                  ha='center', va='center', transform=ax[1].transAxes, fontsize=12)
        ax[1].set_title('Policy vs Value Loss', fontweight='bold')
    
    # 3. Policy Entropy
    if policy_entropy:
        iterations = range(1, len(policy_entropy) + 1)
        ax[2].plot(iterations, policy_entropy, 'purple', linewidth=2)
        ax[2].set_title('Policy Entropy (Exploration)', fontweight='bold')
        ax[2].set_xlabel('Training Iteration')
        ax[2].set_ylabel('Entropy')
        ax[2].grid(True, alpha=0.3)
    else:
        ax[2].text(0.5, 0.5, 'Policy Entropy\nData Not Available', 
                  ha='center', va='center', transform=ax[2].transAxes, fontsize=12)
        ax[2].set_title('Policy Entropy (Exploration)', fontweight='bold')
    
    # 4. Win/Loss/Push Rates
    if win_rates and loss_rates and push_rates:
        iterations = range(1, len(win_rates) + 1)
        ax[3].plot(iterations, win_rates, 'g-', label='Win Rate', linewidth=2)
        ax[3].plot(iterations, loss_rates, 'r-', label='Loss Rate', linewidth=2)
        ax[3].plot(iterations, push_rates, 'gray', label='Push Rate', linewidth=2)
        ax[3].set_title('Game Outcome Rates', fontweight='bold')
        ax[3].set_xlabel('Training Iteration')
        ax[3].set_ylabel('Rate')
        ax[3].legend()
        ax[3].grid(True, alpha=0.3)
        ax[3].set_ylim(0, 1)
    else:
        ax[3].text(0.5, 0.5, 'Game Outcome Rates\nData Not Available', 
                  ha='center', va='center', transform=ax[3].transAxes, fontsize=12)
        ax[3].set_title('Game Outcome Rates', fontweight='bold')
    
    # 5. Average Reward
    if average_rewards:
        iterations = range(1, len(average_rewards) + 1)
        ax[4].plot(iterations, average_rewards, 'cyan', linewidth=2)
        ax[4].set_title('Average Reward per Game', fontweight='bold')
        ax[4].set_xlabel('Training Iteration')
        ax[4].set_ylabel('Reward')
        ax[4].grid(True, alpha=0.3)
        ax[4].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        ax[4].text(0.5, 0.5, 'Average Reward\nData Not Available', 
                  ha='center', va='center', transform=ax[4].transAxes, fontsize=12)
        ax[4].set_title('Average Reward per Game', fontweight='bold')
    
    # 6. Learning Rate (if available)
    if learning_rates:
        iterations = range(1, len(learning_rates) + 1)
        ax[5].plot(iterations, learning_rates, 'brown', linewidth=2)
        ax[5].set_title('Learning Rate Schedule', fontweight='bold')
        ax[5].set_xlabel('Training Iteration')
        ax[5].set_ylabel('Learning Rate')
        ax[5].grid(True, alpha=0.3)
        ax[5].set_yscale('log')
    else:
        ax[5].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                  ha='center', va='center', transform=ax[5].transAxes, fontsize=12)
        ax[5].set_title('Learning Rate Schedule', fontweight='bold')
    
    # 7. Temperature Decay (if available)
    if temperatures:
        iterations = range(1, len(temperatures) + 1)
        ax[6].plot(iterations, temperatures, 'magenta', linewidth=2)
        ax[6].set_title('Exploration Temperature', fontweight='bold')
        ax[6].set_xlabel('Training Iteration')
        ax[6].set_ylabel('Temperature')
        ax[6].grid(True, alpha=0.3)
    else:
        ax[6].text(0.5, 0.5, 'Temperature\nData Not Available', 
                  ha='center', va='center', transform=ax[6].transAxes, fontsize=12)
        ax[6].set_title('Exploration Temperature', fontweight='bold')
    
    # 8. Win Rate Trend Analysis
    if win_rates and len(win_rates) > 10:
        iterations = range(1, len(win_rates) + 1)
        ax[7].plot(iterations, win_rates, 'g-', alpha=0.6, linewidth=1)
        
        # Add moving average
        window_size = min(10, len(win_rates) // 4)
        if window_size > 1:
            moving_avg = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
            moving_iterations = range(window_size, len(win_rates) + 1)
            ax[7].plot(moving_iterations, moving_avg, 'darkgreen', linewidth=3, 
                      label=f'{window_size}-iter Moving Avg')
            ax[7].legend()
        
        ax[7].set_title('Win Rate Trend Analysis', fontweight='bold')
        ax[7].set_xlabel('Training Iteration')
        ax[7].set_ylabel('Win Rate')
        ax[7].grid(True, alpha=0.3)
        ax[7].set_ylim(0, 1)
    else:
        ax[7].text(0.5, 0.5, 'Insufficient Data\nfor Trend Analysis', 
                  ha='center', va='center', transform=ax[7].transAxes, fontsize=12)
        ax[7].set_title('Win Rate Trend Analysis', fontweight='bold')
    
    # 9. Loss Components Comparison
    if policy_losses and value_losses:
        policy_avg = np.mean(policy_losses)
        value_avg = np.mean(value_losses)
        
        categories = ['Policy Loss', 'Value Loss']
        values = [policy_avg, value_avg]
        colors = ['lightblue', 'lightorange']
        
        bars = ax[8].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        ax[8].set_title('Average Loss Components', fontweight='bold')
        ax[8].set_ylabel('Average Loss')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax[8].text(bar.get_x() + bar.get_width()/2., height,
                      f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax[8].text(0.5, 0.5, 'Loss Component\nData Not Available', 
                  ha='center', va='center', transform=ax[8].transAxes, fontsize=12)
        ax[8].set_title('Average Loss Components', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def analyze_training_performance(stats: Dict) -> None:
    """Comprehensive training performance analysis"""
    print_training_summary(stats)
    plot_comprehensive_analysis(stats)


def main(args):
    """Main function"""
    stats = load_training_stats(args.checkpoint)
    if stats is not None:
        analyze_training_performance(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaJack Performance Analysis")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the training checkpoint file')
    args = parser.parse_args()
    main(args)
