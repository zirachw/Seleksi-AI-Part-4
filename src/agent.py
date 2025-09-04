from src.abstract import ReinforcementLearning
import numpy as np

# Reproducibility
np.random.seed(42)

class QLearning(ReinforcementLearning):
    
    def update_q_value(self, state, action, reward, next_state, next_action=None):
        """Q-Learning update (off-policy)"""
        state_idx = state
        next_state_idx = next_state
        
        old_q = self.q_table[state_idx][action]
        max_next_q = np.max(self.q_table[next_state_idx])
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_idx][action] = new_q
    
    def train(self):
        successful_episodes = 0
        
        for episode in range(self.episodes):
            row, col = self.env.start_point
            has_gold = False
            episode_reward = 0
            
            # Maksimal 100 step per episode, biar ga infinite loop ae
            for step in range(100):  
                sensors = self.env.get_sensors(row, col)
                action_idx = self.choose_action(row, col, has_gold, sensors, self.epsilon)
                
                new_row, new_col, new_has_gold, new_sensors, reward, done = self.env.execute_action(
                    row, col, action_idx, has_gold)
                
                episode_reward += reward
                
                state_idx = self.get_state_index(row, col, has_gold, sensors)
                next_state_idx = self.get_state_index(new_row, new_col, new_has_gold, new_sensors)
                
                # Q-value terbaik dari next_state (off-policy)
                self.update_q_value(state_idx, action_idx, reward, next_state_idx)
                
                row, col, has_gold = new_row, new_col, new_has_gold
                
                if done:
                    success = has_gold and (row, col) == self.env.start_point
                    if success:
                        successful_episodes += 1
                        if self.convergence_episode is None and successful_episodes >= 5:
                            self.convergence_episode = episode + 1
                    break

class SARSA(ReinforcementLearning):
    
    def update_q_value(self, state, action, reward, next_state, next_action=None):
        """SARSA update (on-policy)"""
        state_idx = state
        next_state_idx = next_state
        
        old_q = self.q_table[state_idx][action]
        next_q = self.q_table[next_state_idx][next_action] if next_action is not None else 0
        new_q = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
        self.q_table[state_idx][action] = new_q
    
    def train(self):
        successful_episodes = 0

        for episode in range(self.episodes):

            # Inisialisasi state dan action pertama
            row, col = self.env.start_point
            has_gold = False
            sensors = self.env.get_sensors(row, col)
            action_idx = self.choose_action(row, col, has_gold, sensors, self.epsilon)
            episode_reward = 0
            
            # Maksimal 100 step per episode, biar ga infinite loop ae
            for step in range(100):

                # Eksekusi action dan ambil state baru
                new_row, new_col, new_has_gold, new_sensors, reward, done = self.env.execute_action(
                    row, col, action_idx, has_gold)
                
                episode_reward += reward
                
                state_idx = self.get_state_index(row, col, has_gold, sensors)
                next_state_idx = self.get_state_index(new_row, new_col, new_has_gold, new_sensors)
                
                if done:
                    self.update_q_value(state_idx, action_idx, reward, next_state_idx, 0)
                    
                    success = has_gold and (new_row, new_col) == self.env.start_point
                    if success:
                        successful_episodes += 1
                        if self.convergence_episode is None and successful_episodes >= 5:
                            self.convergence_episode = episode + 1
                    break
                
                # Q-value dari next_action yang sebenarnya dipilih pada next_state (on-policy)
                next_action_idx = self.choose_action(new_row, new_col, new_has_gold, new_sensors, self.epsilon)
                self.update_q_value(state_idx, action_idx, reward, next_state_idx, next_action_idx)
                
                row, col, has_gold = new_row, new_col, new_has_gold
                sensors = new_sensors
                action_idx = next_action_idx