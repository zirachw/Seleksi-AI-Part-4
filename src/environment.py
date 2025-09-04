ROWS = 4
COLS = 4

def spec_to_array(spec_pos):
    """ Konversi koordinat dari bawah kiri indeks 1 ke atas kiri indeks 0"""
    spec_col, spec_row,  = spec_pos
    return (4 - spec_row, spec_col - 1)

class WumpusEnvironment:
    
    def __init__(self):
        self.rows = ROWS
        self.columns = COLS
        self.start_point = spec_to_array([1, 1])     # Jadi (3, 0)
        self.wumpus = spec_to_array([1, 3])          # Jadi (1, 0) 
        self.gold = spec_to_array([2, 3])            # Jadi (1, 1)
        self.pits = [spec_to_array([3, 1]), 
                     spec_to_array([3, 3]), 
                     spec_to_array([4, 4])]          # Jadi [(3, 2), (1, 2), (0, 3)]
        
        self.external_env()
        self.actions = ['up', 'down', 'right', 'left', 'grab', 'climb']

    def external_env(self):

        self.stench = set()
        self.breeze = set()
        self.glitter = set()

        wumpus_row, wumpus_col = self.wumpus
        adjacent_to_wumpus = [(wumpus_row-1, wumpus_col), (wumpus_row+1, wumpus_col), 
                             (wumpus_row, wumpus_col-1), (wumpus_row, wumpus_col+1)]
        for row, col in adjacent_to_wumpus:
            if 0 <= row < self.rows and 0 <= col < self.columns:
                self.stench.add((row, col))

        for pit in self.pits:
            pit_row, pit_col = pit
            adjacent_to_pit = [(pit_row-1, pit_col), (pit_row+1, pit_col), 
                              (pit_row, pit_col-1), (pit_row, pit_col+1)]
            for row, col in adjacent_to_pit:
                if 0 <= row < self.rows and 0 <= col < self.columns:
                    self.breeze.add((row, col))
        
        self.glitter.add(self.gold)
    
    def get_sensors(self, row, col):
        sensors = {
            'stench': (row, col) in self.stench,
            'breeze': (row, col) in self.breeze, 
            'glitter': (row, col) in self.glitter
        }
        return sensors

    def execute_action(self, row, col, action_idx, has_gold):

        new_row, new_col, new_has_gold = row, col, has_gold
        done = False
        reward = -1
        action = self.actions[action_idx]
        
        if action == 'up' and row > 0:
            new_row -= 1

        elif action == 'down' and row < self.rows - 1:
            new_row += 1

        elif action == 'right' and col < self.columns - 1:
            new_col += 1

        elif action == 'left' and col > 0:
            new_col -= 1

        elif action == 'grab':
            if (row, col) == self.gold and not has_gold:
                new_has_gold = True
                reward = 1000
            # Asumsi tambahan ya kak: penalti besar pas grab di posisi salah
            else:
                reward = -5

        elif action == 'climb':
            if (row, col) == self.start_point and has_gold:
                done = True
                reward = 0
            # Asumsi tambahan ya kak: penalti besar pas climb di posisi salah
            else:
                reward = -5
        
        # Kasus mati
        if action in ['up', 'down', 'right', 'left']:
            if (new_row, new_col) in self.pits or (new_row, new_col) == self.wumpus:
                reward = -1000
                done = True
        
        # Sensor agent
        sensors = self.get_sensors(new_row, new_col)
        
        return new_row, new_col, new_has_gold, sensors, reward, done