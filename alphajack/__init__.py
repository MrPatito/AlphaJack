"""
AlphaJack Package

An AlphaZero-inspired Blackjack AI that learns optimal strategy through self-play.
"""

from .game_environment import BlackjackEnvironment, Action, Card, Hand
from .neural_network import NeuralNetwork
from .mcts import MCTS, MCTSNode
from .training import AlphaJackTrainer, TrainingData

__version__ = "1.0.0"
__author__ = "AlphaJack Development Team"

__all__ = [
    'BlackjackEnvironment',
    'Action',
    'Card', 
    'Hand',
    'NeuralNetwork',
    'MCTS',
    'MCTSNode',
    'AlphaJackTrainer',
    'TrainingData'
]
