from abc import ABC, abstractmethod
import numpy as np
import random
from src.style import *

# Reproducibility
np.random.seed(42)
random.seed(42)

# Konstanta
ALPHA = 0.3      # Learning rate: -> 0 lebih stabil, -> 1 lebih cepat belajar
GAMMA = 0.9      # Faktor diskon: -> 0 lebih fokus reward jangka pendek, -> 1 lebih fokus reward jangka panjang
EPSILON = 0.2    # Epsilon untuk epsilon-greedy policy: -> 0 lebih eksploitasi (q-table), -> 1 lebih eksplorasi (random)
EPISODES = 1000  # Jumlah iterasi pelatihan

class ReinforcementLearning(ABC):
    """Abstract base class #PenggemarBeratOOP"""
    
    def __init__(self, env):
        self.env = env

        # Q-table: (rows, columns, has_gold, has_stench, has_breeze, has_glitter, actions)
        self.q_table = np.zeros((env.rows, env.columns, 2, 2, 2, 2, len(env.actions)))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.episodes = EPISODES
        self.convergence_episode = None
    
    def get_state_index(self, row, col, has_gold, sensors):
        return (row, col, int(has_gold), 
                int(sensors['stench']), 
                int(sensors['breeze']), 
                int(sensors['glitter']))
    
    def choose_action(self, row, col, has_gold, sensors, epsilon):
        """Epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, len(self.env.actions) - 1)
        
        state_idx = self.get_state_index(row, col, has_gold, sensors)
        return np.argmax(self.q_table[state_idx])
    
    @abstractmethod
    def update_q_value(self, state, action, reward, next_state, next_action=None):
        pass
    
    @abstractmethod
    def train(self):
        pass

    def best_path(self):

        row, col = self.env.start_point
        has_gold = False
        path = [(row, col, has_gold)]
        total_reward = 0
        risk_score = 0
        
        for step in range(100):
            sensors = self.env.get_sensors(row, col)
            action_idx = self.choose_action(row, col, has_gold, sensors, 0.0)  # Pure eksploitasi
            action_name = self.env.actions[action_idx]
            
            new_row, new_col, new_has_gold, new_sensors, reward, done = self.env.execute_action(
                row, col, action_idx, has_gold)
            
            path.append((new_row, new_col, new_has_gold, action_name, reward))
            total_reward += reward
            risk_score += self._calculate_risk(new_row, new_col)
            
            row, col, has_gold = new_row, new_col, new_has_gold
            
            if done:
                return path, total_reward, risk_score

        return path, total_reward, risk_score
    
    def _calculate_risk(self, row, col):
        risk = 0
        
        # Risk dari wumpus
        wumpus_distance = abs(row - self.env.wumpus[0]) + abs(col - self.env.wumpus[1])
        
        if wumpus_distance == 0:
            risk += 5
        elif wumpus_distance == 1:
            risk += 3
        elif wumpus_distance == 2:
            risk += 1

        # Risk dari pits
        for pit in self.env.pits:
            pit_distance = abs(row - pit[0]) + abs(col - pit[1])
            if pit_distance == 0:
                risk += 5
            elif pit_distance == 1:
                risk += 3
            elif pit_distance == 2:
                risk += 1

        return risk