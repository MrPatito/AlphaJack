import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphajack.game_environment import BlackjackEnvironment, Action, Card
from alphajack.neural_network import NeuralNetwork
from alphajack.mcts import MCTS
import numpy as np
import torch

class TestBlackjackEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = BlackjackEnvironment()
        self.env.reset()  # Initialize the game state

    def test_initial_state(self):
        state = self.env.get_state()
        self.assertEqual(state.shape[0], 10)  # Check state size
        self.assertFalse(self.env.game_over)

    def test_legal_actions(self):
        actions = self.env.get_legal_actions()
        self.assertIn(Action.HIT, actions)
        self.assertIn(Action.STAND, actions)

    def test_bust(self):
        # Clear existing cards and manually add cards to test bust
        self.env.player_hand.cards.clear()
        self.env.player_hand.add_card(Card('♣', '10'))
        self.env.player_hand.add_card(Card('♦', '10'))
        self.env.player_hand.add_card(Card('♠', '5'))
        self.assertTrue(self.env.player_hand.is_busted())
        self.assertEqual(self.env.player_hand.get_value(), 25)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork(10, 4)  # 10 input features, 4 actions
        self.net.eval()  # Set to evaluation mode to avoid batch norm issues

    def test_forward(self):
        dummy_input = torch.tensor([0.5]*10, dtype=torch.float32)
        with torch.no_grad():
            policy, value = self.net(dummy_input)
        self.assertEqual(policy.shape[1], 4)  # Check number of actions
        self.assertEqual(value.shape[0], 1)  # Check value output


class TestMCTS(unittest.TestCase):
    def setUp(self):
        net = NeuralNetwork(10, 4)
        net.eval()  # Set to evaluation mode
        self.mcts = MCTS(net, num_simulations=10)  # Fewer simulations for testing
        self.env = BlackjackEnvironment()
        self.env.reset()  # Initialize the game state

    def test_search(self):
        best_action, action_probs = self.mcts.search(self.env)
        self.assertIsNotNone(best_action)
        self.assertEqual(len(action_probs), 4)  # Check action probabilities


if __name__ == '__main__':
    unittest.main()

