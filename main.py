#!/usr/bin/env python3
"""
AlphaJack - Main Execution Script

This script demonstrates how to use the AlphaJack AI system.
It provides examples of training, evaluation, and playing games.
"""

import sys
import os
import argparse
import torch
import configparser
import numpy as np

# Add the alphajack module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphajack.game_environment import BlackjackEnvironment, Action
from alphajack.neural_network import NeuralNetwork
from alphajack.mcts import MCTS
from alphajack.training import AlphaJackTrainer


def demo_game_environment():
    """Demonstrate the Blackjack game environment"""
    print("=== Blackjack Environment Demo ===")
    
    env = BlackjackEnvironment()
    env.reset()
    
    print(f"Initial state: {env.get_state()}")
    print(f"Game info: {env.get_game_info()}")
    print(f"Legal actions: {[action.name for action in env.get_legal_actions()]}")
    
    # Play a simple game
    while not env.game_over:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        # Take first legal action for demo
        action = legal_actions[0]
        next_state, reward, done = env.step(action)
        
        print(f"Action taken: {action.name}")
        print(f"New state: {next_state}")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Game info: {env.get_game_info()}")
        
        if done:
            break
    
    print("Game completed!")
    print()


def demo_neural_network():
    """Demonstrate the neural network"""
    print("=== Neural Network Demo ===")
    
    # Create a neural network
    net = NeuralNetwork(input_size=10, num_actions=4)
    net.eval()  # Set to evaluation mode to avoid batch norm issues
    
    # Create a dummy state
    dummy_state = torch.tensor([0.95, 1.0, 0.9, 0.2, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        policy_probs, value = net(dummy_state)
    
    print(f"Input state: {dummy_state}")
    print(f"Policy probabilities: {policy_probs}")
    print(f"Value prediction: {value}")
    print()


def demo_mcts():
    """Demonstrate MCTS"""
    print("=== MCTS Demo ===")
    
    # Create components
    net = NeuralNetwork(input_size=10, num_actions=4)
    net.eval()  # Set to evaluation mode
    mcts = MCTS(net, num_simulations=50)  # Fewer simulations for demo
    env = BlackjackEnvironment()
    env.reset()
    
    print(f"Initial game state: {env.get_game_info()}")
    
    # Get action from MCTS
    best_action, action_probs = mcts.search(env)
    
    print(f"MCTS recommended action: {best_action.name if best_action else 'None'}")
    print(f"Action probabilities: {action_probs}")
    print()


def train_alphajack(args):
    """Train the AlphaJack AI"""
    print("=== Training AlphaJack ===")
    
    # Apply smoke test settings if enabled
    if args.smoke:
        print("🚀 SMOKE TEST MODE: Reducing parameters for quick validation")
        args.iterations = min(args.iterations, 2)
        args.games_per_iteration = min(args.games_per_iteration, 10)
        args.epochs_per_iteration = min(args.epochs_per_iteration, 2)
        args.simulations = min(args.simulations, 25)
        args.eval_games = min(args.eval_games, 50)
        print(f"   → Iterations: {args.iterations}")
        print(f"   → Games per iteration: {args.games_per_iteration}")
        print(f"   → Epochs per iteration: {args.epochs_per_iteration}")
        print(f"   → MCTS simulations: {args.simulations}")
        print(f"   → Evaluation games: {args.eval_games}")
    
    # Create trainer with GPU/AMP support
    trainer = AlphaJackTrainer(
        state_size=10,  # Updated for enhanced state representation
        num_actions=4,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_simulations=args.simulations,
        device=args.device,
        amp=args.amp,
        seed=args.seed,
        deterministic=args.deterministic,
        max_grad_norm=args.max_grad_norm,
        compile=args.compile
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Start training
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        epochs_per_iteration=args.epochs_per_iteration
    )
    
    # Evaluate the trained model
    print("\\nEvaluating trained model...")
    trainer.evaluate(num_games=args.eval_games)
    
    # Save final model
    trainer.save_checkpoint("final_model.pth")
    print("Training completed and model saved!")


def evaluate_model(args):
    """Evaluate a trained model"""
    print("=== Evaluating Model ===")
    
    if not args.model_path:
        print("Error: Please specify --model_path for evaluation")
        return
    
    # Apply smoke test settings if enabled
    if args.smoke:
        print("🚀 SMOKE TEST MODE: Reducing evaluation games")
        args.eval_games = min(args.eval_games, 50)
        args.simulations = min(args.simulations, 25)
        print(f"   → Evaluation games: {args.eval_games}")
        print(f"   → MCTS simulations: {args.simulations}")
    
    # Create trainer and load model
    trainer = AlphaJackTrainer(num_simulations=args.simulations)
    trainer.load_checkpoint(args.model_path)
    
    # Evaluate
    results = trainer.evaluate(num_games=args.eval_games)
    
    print("\\nDetailed Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def play_interactive_game(args):
    """Play an interactive game against the AI"""
    print("=== Interactive Game ===")
    
    # Apply smoke test settings if enabled
    if args.smoke:
        print("🚀 SMOKE TEST MODE: Reducing MCTS simulations")
        args.simulations = min(args.simulations, 25)
        print(f"   → MCTS simulations: {args.simulations}")
    
    if not args.model_path:
        print("Playing against random AI (no trained model specified)")
        trainer = AlphaJackTrainer(num_simulations=args.simulations)
    else:
        print(f"Loading trained model: {args.model_path}")
        trainer = AlphaJackTrainer(num_simulations=args.simulations)
        trainer.load_checkpoint(args.model_path)
    
    env = BlackjackEnvironment()
    
    while True:
        env.reset()
        print("\\n" + "="*50)
        print("New Game Started!")
        print("="*50)
        
        game_info = env.get_game_info()
        print(f"Your hand value: {game_info['player_hand_value']}")
        print(f"Dealer up card: {game_info['dealer_up_card']}")
        print(f"Running count: {game_info['running_count']}")
        
        while not env.game_over:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            # Get AI recommendation
            action_probs = trainer.mcts.get_action_probabilities(env, temperature=0.0)
            best_ai_action_idx = max(range(len(action_probs)), key=lambda i: action_probs[i])
            
            print(f"\\nLegal actions: {[action.name for action in legal_actions]}")
            print(f"AI recommends: {Action(best_ai_action_idx).name}")
            
            # Get player input
            while True:
                try:
                    print("Choose action:")
                    for i, action in enumerate(legal_actions):
                        print(f"{i}: {action.name}")
                    
                    choice = input("Enter choice (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        return
                    
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(legal_actions):
                        selected_action = legal_actions[choice_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Take action
            next_state, reward, done = env.step(selected_action)
            game_info = env.get_game_info()
            
            print(f"\\nAction taken: {selected_action.name}")
            print(f"Your hand value: {game_info['player_hand_value']}")
            
            if done:
                print(f"\\nGame Over!")
                if game_info['dealer_hand_value'] is not None:
                    print(f"Dealer hand value: {game_info['dealer_hand_value']}")
                print(f"Final reward: {reward}")
                
                if reward > 0:
                    print("You won! 🎉")
                elif reward < 0:
                    print("You lost. 😞")
                else:
                    print("Push (tie). 🤝")
                break
        
        # Ask if player wants to continue
        play_again = input("\\nPlay another game? (y/n): ").strip().lower()
        if play_again != 'y':
            break
    
    print("Thanks for playing!")


def validate_with_real_game(args):
    """Validate model predictions with real Blackjack game data"""
    print("=== Validate with Real Game Data ===")

    if not args.model_path:
        print("Error: Please specify --model_path for validation")
        return

    # Load the model
    trainer = AlphaJackTrainer(num_simulations=args.simulations)
    trainer.load_checkpoint(args.model_path)

    env = BlackjackEnvironment()

    while True:
        try:
            input_state = input("Enter game state as 'player_value, dealer_up_card, running_count' or 'q' to quit: ").strip()
            if input_state.lower() == 'q':
                break

            player_value, dealer_up_card, running_count = map(float, input_state.split(','))
            player_value /= 21.0  # Normalize
            dealer_up_card /= 11.0  # Normalize
            running_count /= 10.0  # Normalize

            # Set the input state manually
            state = np.array([player_value, 0, dealer_up_card, running_count, 0, 0, 0, 0, 0, 0], dtype=np.float32)

            # Get AI recommendation
            with torch.no_grad():
                action_probs, _ = trainer.neural_net(torch.tensor(state).unsqueeze(0))
                best_ai_action_idx = action_probs.argmax().item()
            
            print(f"AI recommends: {Action(best_ai_action_idx).name}")

        except ValueError:
            print("Invalid input format. Please enter valid state values.")
    print("Validation completed.")


def main():
    """Main function"""
    # Load configuration file
    config = configparser.ConfigParser()
    config_file = 'config.ini'
    if os.path.exists(config_file):
        config.read(config_file)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="AlphaJack - AlphaZero-Inspired Blackjack AI")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["demo", "train", "evaluate", "play", "validate"],
                       help="Mode to run the program in")
    parser.add_argument("--config", type=str, default="config.ini",
                       help="Path to configuration file")
    
    # Training arguments (with defaults from config if available)
    parser.add_argument("--iterations", type=int, 
                       default=config.getint('train', 'iterations', fallback=50),
                       help="Number of training iterations")
    parser.add_argument("--games_per_iteration", type=int, 
                       default=config.getint('train', 'games_per_iteration', fallback=100),
                       help="Number of self-play games per iteration")
    parser.add_argument("--epochs_per_iteration", type=int, 
                       default=config.getint('train', 'epochs', fallback=10),
                       help="Number of training epochs per iteration")
    parser.add_argument("--learning_rate", type=float, 
                       default=config.getfloat('train', 'learning_rate', fallback=0.001),
                       help="Learning rate for neural network")
    parser.add_argument("--batch_size", type=int, 
                       default=config.getint('train', 'batch_size', fallback=32),
                       help="Batch size for training")
    parser.add_argument("--simulations", type=int, 
                       default=config.getint('train', 'num_simulations', fallback=100),
                       help="Number of MCTS simulations per move")
    
    # Model arguments
    parser.add_argument("--load_checkpoint", type=str, 
                       default=config.get('model', 'load_checkpoint', fallback=None) if config.get('model', 'load_checkpoint', fallback='None') != 'None' else None,
                       help="Path to checkpoint to load for continued training")
    parser.add_argument("--model_path", type=str, 
                       default=config.get('model', 'model_path', fallback=None),
                       help="Path to trained model for evaluation or playing")
    
    # Evaluation arguments
    parser.add_argument("--eval_games", type=int, 
                       default=config.getint('evaluate', 'eval_games', fallback=1000),
                       help="Number of games for evaluation")
    
    # GPU/AMP arguments
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], 
                       help="Compute device preference")
    parser.add_argument("--amp", type=int, default=1, 
                       help="Enable mixed precision (1) or disable (0)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true", 
                       help="Enable deterministic ops (slower)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                       help="Gradient clipping max norm")
    parser.add_argument("--compile", action="store_true", 
                       help="Compile model with torch.compile (PyTorch 2.x)")
    parser.add_argument("--smoke", action="store_true",
                       help="Run a fast smoke test (tiny data, few steps)")
    
    args = parser.parse_args()
    
    print("AlphaJack - AlphaZero-Inspired Blackjack AI")
    print("=" * 50)
    
    if args.mode == "demo":
        demo_game_environment()
        demo_neural_network()
        demo_mcts()
    elif args.mode == "train":
        train_alphajack(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "play":
        play_interactive_game(args)
    elif args.mode == "validate":
        validate_with_real_game(args)


if __name__ == "__main__":
    main()
