# AlphaJack User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Usage Modes](#usage-modes)
6. [Real Game Validation](#real-game-validation)
7. [Understanding the AI](#understanding-the-ai)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## Introduction

AlphaJack is an AI system that learns to play optimal Blackjack strategy through self-play, inspired by DeepMind's AlphaZero algorithm. The system combines Monte Carlo Tree Search (MCTS) with deep neural networks to develop strategy without human knowledge.

### Key Features
- Self-learning AI that improves through playing against itself
- Real-time game recommendations
- Validation against real Blackjack games
- Configurable training parameters
- Interactive gameplay modes

## Installation

### System Requirements
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for training)
- GPU optional but recommended for faster training

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   cd AlphaJack
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python main.py --mode demo
   ```

## Quick Start

### 1. Demo Mode (See All Components)
```bash
python main.py --mode demo
```
This shows how each component works: game environment, neural network, and MCTS.

### 2. Train a New Model
```bash
python main.py --mode train --iterations 10
```
Quick training with 10 iterations (about 5-10 minutes).

### 3. Play Against the AI
```bash
python main.py --mode play --model_path final_model.pth
```

### 4. Validate with Real Game Data
```bash
python main.py --mode validate --model_path final_model.pth
```

## Configuration

The `config.ini` file controls default parameters:

```ini
[train]
learning_rate = 0.001      # Neural network learning rate
batch_size = 32            # Training batch size
num_simulations = 100      # MCTS simulations per move
epochs = 10                # Training epochs per iteration
games_per_iteration = 100  # Self-play games per iteration
iterations = 50            # Total training iterations

[model]
load_checkpoint = None     # Path to resume training
model_path = "final_model.pth"  # Default model save path

[evaluate]
eval_games = 1000          # Games for evaluation

[mcts]
c_param = 1.4             # Exploration parameter
temperature = 1.0         # Action selection randomness

[blackjack]
decks = 6                 # Number of decks
shuffle_penetration = 0.75 # When to reshuffle
```

### Modifying Configuration

1. **Edit config.ini directly**
2. **Override with command-line arguments:**
   ```bash
   python main.py --mode train --iterations 100 --learning_rate 0.0005
   ```

## Usage Modes

### Demo Mode
Shows each component in action:
```bash
python main.py --mode demo
```

**What you'll see:**
- Game environment demonstration
- Neural network predictions
- MCTS decision-making process

### Train Mode
Train a new AI model:
```bash
python main.py --mode train [options]
```

**Options:**
- `--iterations N`: Number of training iterations
- `--games_per_iteration N`: Self-play games per iteration
- `--simulations N`: MCTS simulations per move
- `--learning_rate F`: Neural network learning rate
- `--batch_size N`: Training batch size
- `--load_checkpoint PATH`: Resume from checkpoint

**Example - Full Training:**
```bash
python main.py --mode train --iterations 200 --games_per_iteration 200 --simulations 200
```

### Evaluate Mode
Test a trained model's performance:
```bash
python main.py --mode evaluate --model_path MODEL_PATH [options]
```

**Options:**
- `--eval_games N`: Number of evaluation games
- `--simulations N`: MCTS simulations for evaluation

**Example:**
```bash
python main.py --mode evaluate --model_path final_model.pth --eval_games 10000
```

### Play Mode
Interactive gameplay against the AI:
```bash
python main.py --mode play --model_path MODEL_PATH
```

**How to Play:**
1. You'll see your hand value and dealer's up card
2. Choose from available actions (0: HIT, 1: STAND, etc.)
3. AI shows its recommendation
4. Game continues until completion

### Validate Mode
Test AI predictions against real Blackjack scenarios:
```bash
python main.py --mode validate --model_path MODEL_PATH
```

## Real Game Validation

The validation mode allows you to input real Blackjack game situations and see what the AI recommends.

### How to Use Validation Mode

1. **Start Validation:**
   ```bash
   python main.py --mode validate --model_path final_model.pth
   ```

2. **Input Format:**
   Enter game state as: `player_value, dealer_up_card, running_count`
   
   Example:
   ```
   Enter game state: 16, 10, -2
   AI recommends: HIT
   ```

3. **Parameters Explained:**
   - `player_value`: Your hand total (4-21)
   - `dealer_up_card`: Dealer's visible card (2-11, where 11=Ace)
   - `running_count`: Hi-Lo card count

### Real Game Scenarios

**Scenario 1: Hard 16 vs Dealer 10**
```
Input: 16, 10, 0
Expected: HIT (basic strategy)
```

**Scenario 2: Soft 18 vs Dealer 9**
```
Input: 18, 9, 0
Expected: HIT (with soft hand indicator)
```

**Scenario 3: Pair of 8s vs Dealer 6**
```
Input: 16, 6, 0
Expected: STAND or SPLIT (if available)
```

### Validation Tips

1. **Use Actual Game Data**: Record real Blackjack hands and test AI predictions
2. **Track Accuracy**: Compare AI recommendations with basic strategy
3. **Consider Count**: High positive counts may change optimal strategy

## Understanding the AI

### How AlphaJack Makes Decisions

1. **State Analysis**: Evaluates current game situation
2. **MCTS Search**: Simulates possible future outcomes
3. **Neural Network**: Provides initial policy and value estimates
4. **Action Selection**: Chooses best action based on simulations

### What the AI Considers

- Player hand value and softness
- Dealer's up card
- Card count (Hi-Lo system)
- Deck penetration
- Available actions

### Interpreting AI Recommendations

- **Confidence**: More simulations = higher confidence
- **Card Counting**: AI adapts strategy based on count
- **Risk Assessment**: Considers probability of outcomes

## Troubleshooting

### Common Issues

**1. "Module not found" Error**
```bash
# Solution: Ensure you're in the AlphaJack directory
cd AlphaJack
python main.py --mode demo
```

**2. "CUDA out of memory" (GPU training)**
```bash
# Solution: Reduce batch size
python main.py --mode train --batch_size 16
```

**3. "Config file not found"**
```bash
# Solution: Create config.ini or specify path
python main.py --config /path/to/config.ini
```

**4. Training Too Slow**
```bash
# Solution: Reduce simulations or games
python main.py --mode train --simulations 50 --games_per_iteration 50
```

### Performance Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support
2. **Parallel Training**: Use multiple CPU cores
3. **Checkpoint Regularly**: Save progress during long training

## Advanced Usage

### Custom Training Schedules

**Progressive Training:**
```bash
# Stage 1: Basic learning
python main.py --mode train --iterations 50 --simulations 50

# Stage 2: Refinement
python main.py --mode train --iterations 50 --simulations 200 --load_checkpoint checkpoint_iter_50.pth

# Stage 3: Expert level
python main.py --mode train --iterations 100 --simulations 500 --load_checkpoint checkpoint_iter_100.pth
```

### Analyzing Training Progress

Monitor these metrics:
- Win rate progression
- Average reward
- Policy entropy (exploration vs exploitation)
- Value loss (prediction accuracy)

### Extending the System

**1. Add New Features:**
- Edit `game_environment.py` to track more state
- Update neural network input size accordingly

**2. Modify Game Rules:**
- Change dealer rules in `game_environment.py`
- Adjust payouts and splitting rules

**3. Enhanced Validation:**
- Create validation datasets
- Batch process multiple scenarios

### Best Practices

1. **Start Small**: Test with few iterations first
2. **Monitor Progress**: Check evaluation metrics regularly
3. **Save Checkpoints**: Don't lose training progress
4. **Validate Results**: Test against known strategies

## Appendix

### Blackjack Rules in AlphaJack

- 6-deck shoe
- Dealer stands on 17
- Blackjack pays 3:2
- One split allowed per game
- Double down on any two cards
- No insurance or surrender

### Action Meanings

- **HIT (0)**: Take another card
- **STAND (1)**: Keep current hand
- **DOUBLE_DOWN (2)**: Double bet, take one card
- **SPLIT (3)**: Split pairs into two hands

### Card Counting (Hi-Lo)

- 2-6: +1
- 7-9: 0
- 10-A: -1

Higher counts favor the player.

---

For additional help or bug reports, check the code documentation or examine the demo mode output for detailed component behavior.
