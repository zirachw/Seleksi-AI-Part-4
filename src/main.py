from src.environment import WumpusEnvironment
from src.agent import QLearning, SARSA
from src.style import welcome, success, show_results, comparison

# Semoga jodoh lab AI, amiin

env = WumpusEnvironment()
welcome(env)

q_agent = QLearning(env)
q_agent.train()
print(success('[!] Q-Learning training completed!\n'))
q_risk = show_results(q_agent, "Q-Learning")

sarsa_agent = SARSA(env)
sarsa_agent.train()
print(success('[!] SARSA training completed!\n'))
sarsa_risk = show_results(sarsa_agent, "SARSA")

comparison(q_agent, sarsa_agent, q_risk, sarsa_risk)