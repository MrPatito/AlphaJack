# AlphaJack - AlphaZero-Inspired Blackjack AI

An implementation of an AlphaZero-inspired artificial intelligence system that learns to play optimal Blackjack strategy through self-play using Monte Carlo Tree Search (MCTS) and deep neural networks.

## 🎯 Project Overview

AlphaJack is a complete Python application that learns Blackjack strategy from scratch through self-play. The system uses:

- **Monte Carlo Tree Search (MCTS)** for move selection and game tree exploration
- **Deep Neural Network** with policy and value heads for position evaluation
- **Self-play training** where the AI improves by playing games against itself
- **PyTorch** for neural network implementation

The AI focuses exclusively on gameplay actions (Hit, Stand, Double Down, Split) and learns optimal strategy without human intervention or pre-programmed rules.

## 🏗️ Architecture

### Component 1: Blackjack Game Environment (`game_environment.py`)
- Complete Blackjack simulation with standard rules
- 6-deck shoe with automatic shuffling
- State representation including card counting (Hi-Lo system)
- Support for all standard actions including splitting

### Component 2: Neural Network (`neural_network.py`)
- PyTorch-based neural network with shared layers
- **Policy Head**: Outputs probability distribution over actions
- **Value Head**: Predicts game outcome (-1 to +1)
- 3 hidden layers with 128 neurons each

### Component 3: Monte Carlo Tree Search (`mcts.py`)
- UCT (Upper Confidence Bound for Trees) based selection
- Neural network guided expansion and evaluation
- Configurable number of simulations per move
- Temperature-based action selection

### Component 4: Training Loop (`training.py`)
- Self-play data collection
- Neural network training with combined policy and value loss
- Checkpoint saving and loading
- Training statistics and evaluation

## 🚀 Quick Start

### Installation

1. Clone or download the project:
```bash
cd AlphaJack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage Examples

#### 1. Run Demonstration
```bash
python main.py --mode demo
```
This will demonstrate each component of the system.

#### 2. Train the AI
```bash
# Quick training (50 iterations)
python main.py --mode train --iterations 50

# Extended training with custom parameters
python main.py --mode train --iterations 200 --games_per_iteration 200 --simulations 200
```

#### 3. Evaluate a Trained Model
```bash
python main.py --mode evaluate --model_path final_model.pth --eval_games 10000
```

#### 4. Play Against the AI
```bash
python main.py --mode play --model_path final_model.pth
```

## 📊 Training Process

The training follows these steps:

1. **Self-Play**: The AI plays games against itself using MCTS
2. **Data Collection**: States, action probabilities, and outcomes are stored
3. **Network Training**: The neural network learns from collected data
4. **Iteration**: Process repeats with the improved network

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 50 | Number of training iterations |
| `--games_per_iteration` | 100 | Self-play games per iteration |
| `--epochs_per_iteration` | 10 | Neural network training epochs |
| `--simulations` | 100 | MCTS simulations per move |
| `--learning_rate` | 0.001 | Neural network learning rate |
| `--batch_size` | 32 | Training batch size |

## 🎮 Game Features

### State Representation
The AI observes:
- Player hand value (4-21)
- Whether hand is "soft" (contains Ace as 11)
- Dealer's up-card value
- Running card count (Hi-Lo system)
- Deck penetration (fraction of cards dealt)

### Actions
- **Hit**: Take another card
- **Stand**: End turn with current hand
- **Double Down**: Double bet and take exactly one more card
- **Split**: Split pair into two separate hands (one split per game)

### Rewards
- Win: +1 (or +1.5 for blackjack)
- Loss: -1
- Push: 0
- Rewards doubled for doubled-down hands

## 📈 Expected Learning Progression

1. **Early Training (0-20 iterations)**: Random play, learning basic game rules
2. **Basic Strategy (20-100 iterations)**: Learning when to hit vs. stand
3. **Advanced Strategy (100+ iterations)**: Learning optimal doubling and splitting
4. **Expert Play (200+ iterations)**: Near-optimal strategy with card counting integration

## 🔧 Customization

### Modifying Game Rules
Edit `game_environment.py` to change:
- Number of decks
- Dealer hitting rules
- Blackjack payout ratios
- Splitting restrictions

### Network Architecture
Modify `neural_network.py` to experiment with:
- Layer sizes and depths
- Activation functions
- Additional input features

### MCTS Parameters
Adjust in `mcts.py`:
- UCT exploration parameter (c_param)
- Number of simulations
- Temperature schedules

## 📁 Project Structure

```
AlphaJack/
├── alphajack/
│   ├── __init__.py
│   ├── game_environment.py    # Blackjack game simulation
│   ├── neural_network.py      # PyTorch neural network
│   ├── mcts.py               # Monte Carlo Tree Search
│   └── training.py           # Training orchestration
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🎯 Performance Expectations

A well-trained AlphaJack model should achieve:
- **Win Rate**: ~42-45% (typical for optimal Blackjack play)
- **Expected Value**: Close to -0.5% (theoretical Blackjack house edge)
- **Strategy Accuracy**: >95% agreement with basic strategy charts

## 🔬 Advanced Features

### Card Counting Integration
The system includes Hi-Lo card counting:
- Cards 2-6: +1
- Cards 7-9: 0  
- Cards 10,J,Q,K,A: -1

The running count is part of the state representation, allowing the AI to learn betting and strategy adjustments.

### Checkpoint System
- Automatic checkpoints every 10 iterations
- Resume training from saved checkpoints
- Model evaluation and comparison tools

## 🤝 Contributing

This is a complete, self-contained implementation. Potential improvements:
- Add insurance betting
- Implement surrender option
- Multi-hand play simulation
- Advanced card counting systems
- Tournament play modes

## 📜 License

This project is provided as-is for educational and research purposes. The implementation demonstrates core AlphaZero concepts applied to Blackjack.

## 🙏 Acknowledgments

Inspired by:
- DeepMind's AlphaZero algorithm
- Classic Monte Carlo Tree Search research
- Traditional Blackjack strategy analysis

## 📞 Support

For questions about the implementation:
1. Check the inline code documentation
2. Review the demo modes in `main.py`
3. Examine the training statistics and logs

---

**Happy Learning!** 🃏🤖
