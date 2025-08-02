"""
Component 1: Blackjack Game Environment

This module implements the complete Blackjack game simulation with state representation
and action handling for the AlphaJack AI system.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from enum import Enum


class Action(Enum):
    """Enumeration of possible player actions"""
    HIT = 0
    STAND = 1
    DOUBLE_DOWN = 2
    SPLIT = 3


class Card:
    """Represents a playing card"""
    
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
    
    def get_value(self) -> int:
        """Returns the base value of the card (Ace = 1, Face cards = 10)"""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 1
        else:
            return int(self.rank)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"


class Hand:
    """Represents a hand of cards"""
    
    def __init__(self):
        self.cards: List[Card] = []
        self.is_split_hand = False
        self.has_doubled = False
    
    def add_card(self, card: Card):
        """Add a card to the hand"""
        self.cards.append(card)
    
    def get_value(self) -> int:
        """Calculate the best value of the hand"""
        total = 0
        aces = 0
        
        for card in self.cards:
            value = card.get_value()
            total += value
            if card.rank == 'A':
                aces += 1
        
        # Optimize Ace values
        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1
        
        return total
    
    def is_soft(self) -> bool:
        """Check if hand is soft (contains an Ace counted as 11)"""
        total = sum(card.get_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')
        
        # If we can count an Ace as 11 without busting, it's soft
        return aces > 0 and total + 10 <= 21 and total + 10 == self.get_value()
    
    def is_busted(self) -> bool:
        """Check if hand is busted (over 21)"""
        return self.get_value() > 21
    
    def is_blackjack(self) -> bool:
        """Check if hand is a natural blackjack"""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def can_split(self) -> bool:
        """Check if hand can be split"""
        return (len(self.cards) == 2 and 
                self.cards[0].get_value() == self.cards[1].get_value() and
                not self.is_split_hand)


class BlackjackEnvironment:
    """Main Blackjack game environment"""
    
    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.deck: List[Card] = []
        self.cards_dealt = 0
        self.running_count = 0
        
        # Game state
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.split_hand: Optional[Hand] = None
        self.current_hand_is_split = False
        self.game_over = False
        
        self._create_deck()
        self._shuffle_deck()
    
    def _create_deck(self):
        """Create a fresh deck of cards"""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        self.deck = []
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.deck.append(Card(suit, rank))
    
    def _shuffle_deck(self):
        """Shuffle the deck and reset counters"""
        random.shuffle(self.deck)
        self.cards_dealt = 0
        self.running_count = 0
    
    def _deal_card(self) -> Card:
        """Deal a card from the deck and update count"""
        if len(self.deck) - self.cards_dealt < 52:  # Reshuffle when low
            self._create_deck()
            self._shuffle_deck()
        
        card = self.deck[self.cards_dealt]
        self.cards_dealt += 1
        
        # Update Hi-Lo count
        if card.rank in ['2', '3', '4', '5', '6']:
            self.running_count += 1
        elif card.rank in ['10', 'J', 'Q', 'K', 'A']:
            self.running_count -= 1
        
        return card
    
    def reset(self):
        """Reset the game for a new hand"""
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.split_hand = None
        self.current_hand_is_split = False
        self.game_over = False
        
        # Deal initial cards
        self.player_hand.add_card(self._deal_card())
        self.dealer_hand.add_card(self._deal_card())
        self.player_hand.add_card(self._deal_card())
        self.dealer_hand.add_card(self._deal_card())
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state representation for the neural network.
        Returns a fixed-length vector containing:
        - Player's hand value
        - Is player's hand soft (0 or 1)
        - Dealer's up-card value
        - Running count
        - Deck penetration
        - Number of cards in player's hand
        - Can double down (0 or 1)
        - Can split (0 or 1)
        - Is split hand active (0 or 1)
        - Has doubled down (0 or 1)
        """
        current_hand = self.split_hand if self.current_hand_is_split else self.player_hand
        
        player_value = current_hand.get_value()
        is_soft = 1 if current_hand.is_soft() else 0
        dealer_up_card = self.dealer_hand.cards[0].get_value()
        deck_penetration = self.cards_dealt / (self.num_decks * 52)
        num_cards = len(current_hand.cards)
        
        # Check if actions are legal
        legal_actions = self.get_legal_actions()
        can_double = 1 if Action.DOUBLE_DOWN in legal_actions else 0
        can_split = 1 if Action.SPLIT in legal_actions else 0
        is_split_hand = 1 if self.split_hand is not None else 0
        has_doubled = 1 if current_hand.has_doubled else 0
        
        state = np.array([
            player_value / 21.0,  # Normalize to [0, 1]
            is_soft,
            dealer_up_card / 11.0,  # Normalize to [0, 1]
            self.running_count / 10.0,  # Normalize count
            deck_penetration,
            num_cards / 10.0,  # Normalize number of cards
            can_double,
            can_split,
            is_split_hand,
            has_doubled
        ], dtype=np.float32)
        
        return state
    
    def get_legal_actions(self) -> List[Action]:
        """Get list of legal actions in current state"""
        if self.game_over:
            return []
        
        current_hand = self.split_hand if self.current_hand_is_split else self.player_hand
        legal_actions = [Action.HIT, Action.STAND]
        
        # Double down only allowed on first two cards and if not already doubled
        if len(current_hand.cards) == 2 and not current_hand.has_doubled:
            legal_actions.append(Action.DOUBLE_DOWN)
        
        # Split only allowed if hand can be split and we haven't split yet
        if current_hand.can_split() and self.split_hand is None:
            legal_actions.append(Action.SPLIT)
        
        return legal_actions
    
    def step(self, action: Action) -> Tuple[np.ndarray, float, bool]:
        """
        Execute an action and return (next_state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0.0, True
        
        current_hand = self.split_hand if self.current_hand_is_split else self.player_hand
        
        if action == Action.HIT:
            current_hand.add_card(self._deal_card())
            if current_hand.is_busted():
                if self.split_hand and not self.current_hand_is_split:
                    # Switch to split hand if available
                    self.current_hand_is_split = True
                    return self.get_state(), 0.0, False
                else:
                    self.game_over = True
                    return self.get_state(), self._calculate_final_reward(), True
        
        elif action == Action.STAND:
            if self.split_hand and not self.current_hand_is_split:
                # Switch to split hand if available
                self.current_hand_is_split = True
                return self.get_state(), 0.0, False
            else:
                # Play dealer's hand
                self._play_dealer()
                self.game_over = True
                return self.get_state(), self._calculate_final_reward(), True
        
        elif action == Action.DOUBLE_DOWN:
            current_hand.has_doubled = True
            current_hand.add_card(self._deal_card())
            if self.split_hand and not self.current_hand_is_split:
                # Switch to split hand if available
                self.current_hand_is_split = True
                return self.get_state(), 0.0, False
            else:
                # Play dealer's hand
                self._play_dealer()
                self.game_over = True
                return self.get_state(), self._calculate_final_reward(), True
        
        elif action == Action.SPLIT:
            # Create split hand
            self.split_hand = Hand()
            self.split_hand.is_split_hand = True
            
            # Move one card to split hand
            card_to_move = current_hand.cards.pop()
            self.split_hand.add_card(card_to_move)
            
            # Deal new cards to both hands
            current_hand.add_card(self._deal_card())
            self.split_hand.add_card(self._deal_card())
        
        return self.get_state(), 0.0, False
    
    def _play_dealer(self):
        """Play the dealer's hand according to standard rules"""
        while self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self._deal_card())
    
    def _calculate_final_reward(self) -> float:
        """Calculate the final reward for the game"""
        dealer_value = self.dealer_hand.get_value()
        dealer_busted = self.dealer_hand.is_busted()
        
        total_reward = 0.0
        hands_to_evaluate = [self.player_hand]
        
        if self.split_hand:
            hands_to_evaluate.append(self.split_hand)
        
        for hand in hands_to_evaluate:
            player_value = hand.get_value()
            player_busted = hand.is_busted()
            multiplier = 2.0 if hand.has_doubled else 1.0
            
            if player_busted:
                total_reward -= multiplier
            elif dealer_busted:
                if hand.is_blackjack():
                    total_reward += multiplier * 1.5
                else:
                    total_reward += multiplier
            elif player_value > dealer_value:
                if hand.is_blackjack():
                    total_reward += multiplier * 1.5
                else:
                    total_reward += multiplier
            elif player_value < dealer_value:
                total_reward -= multiplier
            # Push (tie) - no reward change
        
        # Normalize reward to [-1, 1] range
        max_possible_reward = 2 * 1.5  # Two hands, both blackjack
        return np.clip(total_reward / max_possible_reward, -1.0, 1.0)
    
    def get_game_info(self) -> dict:
        """Get current game information for debugging/display"""
        current_hand = self.split_hand if self.current_hand_is_split else self.player_hand
        
        return {
            'player_hand_value': current_hand.get_value(),
            'player_hand_soft': current_hand.is_soft(),
            'dealer_up_card': self.dealer_hand.cards[0].get_value(),
            'dealer_hand_value': self.dealer_hand.get_value() if self.game_over else None,
            'running_count': self.running_count,
            'deck_penetration': self.cards_dealt / (self.num_decks * 52),
            'game_over': self.game_over,
            'has_split': self.split_hand is not None,
            'current_hand_is_split': self.current_hand_is_split
        }
