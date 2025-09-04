import numpy as np

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'

BRIGHT_BLACK = '\033[90m'
BRIGHT_RED = '\033[91m'
BRIGHT_GREEN = '\033[92m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_CYAN = '\033[96m'
BRIGHT_WHITE = '\033[97m'

SUCCESS = BRIGHT_GREEN
DANGER = BRIGHT_RED
WARNING = BRIGHT_YELLOW
TITLE = BRIGHT_CYAN + BOLD
HEADER = BRIGHT_YELLOW
BEST_ACTION = BRIGHT_GREEN + BOLD
NEGATIVE_VALUE = BRIGHT_RED
POSITIVE_VALUE = BRIGHT_GREEN

SYMBOL_START = f"{BRIGHT_CYAN}S{RESET}"
SYMBOL_WUMPUS = f"{BRIGHT_RED}W{RESET}"
SYMBOL_GOLD = f"{BRIGHT_YELLOW}*{RESET}"
SYMBOL_PIT = f"{BRIGHT_BLACK}P{RESET}"
SYMBOL_EMPTY = f"{DIM}.{RESET}"
SYMBOL_AGENT = f"{BRIGHT_GREEN}A{RESET}"

SYMBOL_UP = f"{BRIGHT_BLUE}↑{RESET}"
SYMBOL_DOWN = f"{BRIGHT_BLUE}↓{RESET}"
SYMBOL_RIGHT = f"{BRIGHT_BLUE}→{RESET}"
SYMBOL_LEFT = f"{BRIGHT_BLUE}←{RESET}"
SYMBOL_GRAB = f"{BRIGHT_YELLOW}G{RESET}"
SYMBOL_CLIMB = f"{BRIGHT_GREEN}C{RESET}"

def colored(text, color):
    return f"{color}{text}{RESET}"

def title(text):
    return f"{TITLE}{text}{RESET}"

def success(text):
    return f"{SUCCESS}{text}{RESET}"

def danger(text):
    return f"{DANGER}{text}{RESET}"

def warning(text):
    return f"{WARNING}{text}{RESET}"

def info(text):
    return f"{text}{RESET}"

def print_q_table(agent, show_detailed=False):
    if show_detailed:
        print(info("\n[~] Q-Table pas belum bawa gold:\n"))
        display_table(agent, 0)
        
        print(info("[~] Q-Table pas udah bawa gold:\n"))
        display_table(agent, 1)

def display_table(agent, has_gold_state):

    actions = ['Up', 'Down', 'Right', 'Left', 'Grab', 'Climb']
    widths = [6]

    # Q-values sensor diambil rata-rata untuk display
    for i in range(len(actions)):
        max_width = len(actions[i])
        for row in range(agent.env.rows):
            for col in range(agent.env.columns):
                q_sum = 0
                count = 0
                for stench in [0, 1]:
                    for breeze in [0, 1]:
                        for glitter in [0, 1]:
                            q_sum += agent.q_table[row, col, has_gold_state, stench, breeze, glitter, i]
                            count += 1
                avg_q = q_sum / count if count > 0 else 0
                val_len = len(f"{avg_q:.2f}")
                max_width = max(max_width, val_len)
        widths.append(max_width)
    
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(f"{border}{RESET}")
    
    header = f"|{RESET} {HEADER}{'Pos':^{widths[0]}}{RESET} |{RESET}"
    for i, action in enumerate(actions):
        header += f" {HEADER}{action:^{widths[i+1]}}{RESET} |{RESET}"
    print(header)
    print(f"{border}{RESET}")
    
    for row in range(agent.env.rows):
        for col in range(agent.env.columns):
            
            # Q-values sensor diambil rata-rata untuk display
            q_values = []
            
            for i in range(len(actions)):
                q_sum = 0
                count = 0
                for stench in [0, 1]:
                    for breeze in [0, 1]:
                        for glitter in [0, 1]:
                            q_sum += agent.q_table[row, col, has_gold_state, stench, breeze, glitter, i]
                            count += 1
                avg_q = q_sum / count if count > 0 else 0
                q_values.append(avg_q)
            
            best_idx = np.argmax(q_values)
            
            pos_str = f"({row},{col})"
            line = f"|{RESET} {pos_str:^{widths[0]}}{RESET} |{RESET}"
            
            for i, q_val in enumerate(q_values):
                if i == best_idx:
                    color = BEST_ACTION
                elif q_val < 0:
                    color = NEGATIVE_VALUE
                else:
                    color = POSITIVE_VALUE if q_val > 0 else RESET
                
                line += f" {color}{q_val:^{widths[i+1]}.2f}{RESET} |{RESET}"
            print(line)
    
    print(f"{border}{RESET}")
    print()

def print_policy(agent):

    symbols = [SYMBOL_UP, SYMBOL_DOWN, SYMBOL_RIGHT, SYMBOL_LEFT, SYMBOL_GRAB, SYMBOL_CLIMB]
    print(info("[~] Policy pas belum bawa gold:\n"))

    for row in range(agent.env.rows):
        for col in range(agent.env.columns):
            if (row, col) == agent.env.start_point:
                print(f' {SYMBOL_START} ', end='')
            elif (row, col) == agent.env.wumpus:
                print(f' {SYMBOL_WUMPUS} ', end='')
            elif (row, col) == agent.env.gold:
                print(f' {SYMBOL_GOLD} ', end='')
            elif (row, col) in agent.env.pits:
                print(f' {SYMBOL_PIT} ', end='')
            else:
                # Q-values sensor diambil rata-rata untuk display
                percept_combinations = []
                for stench in [0, 1]:
                    for breeze in [0, 1]:
                        for glitter in [0, 1]:
                            percept_combinations.append((stench, breeze, glitter))
                
                q_sum = np.zeros(len(agent.env.actions))
                for stench, breeze, glitter in percept_combinations:
                    q_sum += agent.q_table[row, col, 0, stench, breeze, glitter]
                best_action = np.argmax(q_sum)
                print(f' {symbols[best_action]} ', end='')
        print()
    
    print(info("\n[~] Policy pas udah bawa gold:\n"))
    for row in range(agent.env.rows):
        for col in range(agent.env.columns):
            if (row, col) == agent.env.start_point:
                print(f' {SYMBOL_START} ', end='')
            elif (row, col) == agent.env.wumpus:
                print(f' {SYMBOL_WUMPUS} ', end='')
            elif (row, col) == agent.env.gold:
                print(f' {SYMBOL_GOLD} ', end='')
            elif (row, col) in agent.env.pits:
                print(f' {SYMBOL_PIT} ', end='')
            else:
                # Q-values sensor diambil rata-rata untuk display
                percept_combinations = []
                for stench in [0, 1]:
                    for breeze in [0, 1]:
                        for glitter in [0, 1]:
                            percept_combinations.append((stench, breeze, glitter))
                
                q_sum = np.zeros(len(agent.env.actions))
                for stench, breeze, glitter in percept_combinations:
                    q_sum += agent.q_table[row, col, 1, stench, breeze, glitter]
                best_action = np.argmax(q_sum)
                print(f' {symbols[best_action]} ', end='')
        print()

def welcome(env):

    print(f'\n{BRIGHT_BLACK}Wumpus World RL Agents, made by luv ~ rzi{RESET}\n')
    print(title('[#] Environment Specification:'))
    print(f'    Grid size: {SUCCESS}{env.rows}x{env.columns}{RESET}')
    print(f'    Start position: {BRIGHT_CYAN}{env.start_point}{RESET}')
    print(f'    Wumpus position: {DANGER}{env.wumpus}{RESET}')
    print(f'    Gold position: {BRIGHT_YELLOW}{env.gold}{RESET}')
    print(f'    Pit positions: {DANGER}{env.pits}{RESET}')
    print()
    
    print(title('[#] Environment Layout:'))
    print(f'Legend: {BRIGHT_CYAN}S{RESET} (Start), {BRIGHT_RED}W{RESET} (Wumpus), {BRIGHT_YELLOW}*{RESET} (Gold), {BRIGHT_BLACK}P{RESET} (Pit), {DIM}.{RESET} (Empty)\n')
    for row in range(env.rows):
        for col in range(env.columns):
            if (row, col) == env.start_point:
                print(f' {SYMBOL_START} ', end='')
            elif (row, col) == env.wumpus:
                print(f' {SYMBOL_WUMPUS} ', end='')
            elif (row, col) == env.gold:
                print(f' {SYMBOL_GOLD} ', end='')
            elif (row, col) in env.pits:
                print(f' {SYMBOL_PIT} ', end='')
            else:
                print(f' {SYMBOL_EMPTY} ', end='')
        print()
    print()

def show_results(agent, algorithm_name):
    
    print(title(f'[#] {algorithm_name} results:'))
    
    convergence_ep = agent.convergence_episode if agent.convergence_episode else "No convergence"
    print(info(f"    Model menang di episode: {success(str(convergence_ep))}"))
    
    print_q_table(agent, show_detailed=True)
    print_policy(agent)
    
    print(title(f"\n[#] {algorithm_name} optimal path:"))
    path_result = agent.best_path()
    path, total_reward, risk_score = path_result
    
    for i, step in enumerate(path):
        if len(step) == 3:
            row, col, has_gold = step
            gold_status = success("Yes") if has_gold else danger("No")
            print(f"    Step {i}:{BRIGHT_BLUE} Start at {colored(f'({row},{col})', BRIGHT_CYAN)}, Gold: {gold_status}")
        else:
            row, col, has_gold, action, reward = step
            gold_status = success("Yes") if has_gold else danger("No")
            reward_color = POSITIVE_VALUE if reward > 0 else NEGATIVE_VALUE if reward < 0 else RESET
            print(f"    Step {i}:{RESET} {colored(action, BRIGHT_BLUE)} -> {colored(f'({row},{col})', BRIGHT_CYAN)}, Gold: {gold_status}, Reward: {colored(str(reward), reward_color)}")
    
    print(title(f"\n[#] Risiko path dari {algorithm_name}"))
    reward_color = POSITIVE_VALUE if total_reward > 0 else NEGATIVE_VALUE
    print(info(f"[-] Total reward: {colored(str(total_reward), reward_color)}"))
    print(info(f"[-] Risk score: {colored(f'{risk_score:.2f}', WARNING)}"))
    print()
    
    return risk_score

def comparison(q_agent, sarsa_agent, q_risk, sarsa_risk):

    print(f'{BRIGHT_BLACK}[#] Perbandingan antara Q-Learning dan SARSA{RESET}')
    
    q_conv = q_agent.convergence_episode if q_agent.convergence_episode else "No convergence"
    sarsa_conv = sarsa_agent.convergence_episode if sarsa_agent.convergence_episode else "No convergence"
     
    print(info(f"    Q-Learning menang di episode {success(str(q_conv))}"))
    print(info(f"    SARSA menang di episode {success(str(sarsa_conv))}"))
    print(info(f"    Q-Learning memiliki skor risiko: {colored(f'{q_risk:.2f}', WARNING)}"))
    print(info(f"    SARSA memiliki skor risiko: {colored(f'{sarsa_risk:.2f}', WARNING)}"))