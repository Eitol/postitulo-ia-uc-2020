"""
Original author: Alexandre Bergel
Adaptation: Hector Oliveros
"""

import enum
from typing import List, Tuple
from copy import deepcopy
import numpy as np
import random

# Si quieres que todas las corridas del programa tengan el
# mismo resultado entonces asigna esta variable en True
EXPECT_REPRODUCIBILITY = True

# Semilla por defecto
DEFAULT_SEED = 42

if EXPECT_REPRODUCIBILITY:
    random.seed(DEFAULT_SEED)  # for reproducibility
# How many episodes we are considering? More episodes
# means more exploration, and therefore a more efficient model
N_EPISODES = 20

# How many steps, at maximum, per episode? We need to
# make sure that this number is large enough for end the episode
MAX_EPISODE_STEPS = 100
MIN_ALPHA = 0.02

# np.linspace: Crea un array de "N_EPISODES" size y
# donde el primer elemento es 1.0 y el último es MIN_ALPHA
# El resto son valores intermedios
# [
#    1.0,         0.94842105, 0.89684211, 0.84526316, 0.79368421,
#    0.74210526,  0.69052632, 0.63894737, 0.58736842, 0.53578947,
#    0.48421053,  0.43263154, 0.38105263, 0.32947368, 0.27789474,
#    0.22631579,  0.17473684, 0.12315781, 0.07157895, 0.02
# ]
ALPHAS = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
GAMMA = 1.0

# Porcentaje de las veces en la que se elige una acción aleatoria
EXPLORATION_FACTOR = 0.2


class GridItem(enum.Enum):
    ZOMBIE = 'z'
    CAR = 'c'
    ICE_CREAM = 'i'
    EMPTY = '*'
    
    def __repr__(self):
        return self.value


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTIONS = [
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT
]

Grid = List[List[GridItem]]
Coordinate = List[int]

X = 0
Y = 1


class State(object):
    q_table = dict()
    
    def __init__(self, grid: Grid, car_pos: Coordinate):
        self.car_pos = car_pos
        self.grid = grid
    
    def __eq__(self, other: 'State') -> bool:
        return isinstance(other, State) \
               and self.grid == other.grid \
               and self.car_pos == other.car_pos
    
    def __hash__(self) -> int:
        return self.get_id()
    
    def get_id(self) -> int:
        """
        Retorna un identificador único del estado
        """
        return hash(str(self.grid) + str(self.car_pos))
    
    def __str__(self) -> str:
        return f'State(grid={self.grid}, car_post={self.car_pos})'
    
    @staticmethod
    def compute_new_car_pos(state: 'State', action: Action) -> Coordinate:
        """
        Crea una nueva coordenada según una nueva acción
        
        Por ejemplo:
        "CAR" es 'c' está en la coordenada [ 1, 1 ]
          
          0   1
         _______
        |_i_|_*_| <- 0
        |_z_|_c_| <- 1
        
        Entonces si la acción es 'UP', CAR terminará en la coordenada [ 1, 0 ]
        
          0   1
         ______
        |_i_|_c_| <- 0
        |_z_|_*_| <- 1
        
        :param state:
        :param action:
        :return:
        """
        new_coord = deepcopy(state.car_pos)
        
        if action == Action.UP:
            new_coord[X] = max(0, new_coord[X] - 1)
        elif action == Action.DOWN:
            new_coord[X] = min(len(state.grid) - 1, new_coord[X] + 1)
        elif action == Action.LEFT:
            new_coord[Y] = max(0, new_coord[Y] - 1)
        elif action == Action.RIGHT:
            new_coord[Y] = min(len(state.grid[0]) - 1, new_coord[Y] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return new_coord
    
    def act(self, state: 'State', action: Action) -> Tuple['State', int, bool]:
        """
        Toma un estado y una acción, recompensa y si este episodio se ha completado
        
        :param state: El estado actual
        :param action: Acción a ajecutar
        :return:
        """
        coord = self.compute_new_car_pos(state, action)
        grid_item = state.grid[coord[X]][coord[Y]]
        
        new_grid = deepcopy(state.grid)
        
        if grid_item == GridItem.ZOMBIE:
            reward = -100
            is_done = True
            new_grid[coord[X]][coord[Y]] = GridItem.CAR
        elif grid_item == GridItem.ICE_CREAM:
            reward = 1000
            is_done = True
            new_grid[coord[X]][coord[Y]] = GridItem.CAR
        elif grid_item == GridItem.EMPTY:
            reward = -1
            is_done = False
            old = state.car_pos
            new_grid[old[X]][old[Y]] = GridItem.EMPTY
            new_grid[coord[X]][coord[Y]] = GridItem.CAR
        elif grid_item == GridItem.CAR:
            reward = -1
            is_done = False
        else:
            raise ValueError(f"Unknown grid item {grid_item}")
        return State(grid=new_grid, car_pos=coord), reward, is_done
    
    def q(self, state: 'State') -> np.ndarray:
        """
        Permite acceder a la fila de elementos para un valor de la q_table
        
        Cuando el id del elemento no existe en las llaves de la tabla,
        entonces inicializa dicho valor con un arreglo de ceros
        """
        state_id = state.get_id()
        if state_id not in self.q_table:
            self.q_table[state_id] = np.zeros(len(ACTIONS))
        return self.q_table[state_id]
    
    def choose_action(self, state: 'State') -> Action:
        # El "EPS" de las veces se toma una acción aleatória
        # Esto sirve para explorar otras soluciones que pueden.
        # Por ejemplo si EPS es 0.2 entonces el 20% de las veces
        # no se toma en cuenta la tabla y se elige una acción de forma
        # aleatoria
        # Se basa en: "Epsilon-Greedy Algorithm"
        # see: https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning
        if random.uniform(0, 1) < EXPLORATION_FACTOR:
            return Action(random.choice(ACTIONS))
        
        # np.argmax(array): retorna el índice del mayor valor.
        #                           0  1  2  3
        # Por ejemplo en array == [ 8, 3, 9, 5 ]
        # retornará 2
        return Action(np.argmax(self.q(state)))
    
    @staticmethod
    def compute_bellman_optimality_equation(current_q: float, next_row: np.ndarray,
                                            alpha: float, reward: float, gamma: float) -> float:
        """
        # Bellman optimality equation for q*
        :param current_q: the current desition quality
        :param next_row: the next row state
        :param alpha: the current alpha
        :param reward: the current reward
        :param gamma:
        :return:
        """
        return current_q + alpha * (reward + gamma * np.max(next_row) - current_q)
    
    def run_episode(self, alpha: float, gamma: float, state: 'State', total_rw: float) -> Tuple['State', bool, float]:
        """
        Corre un episodio
        
        :param alpha: The current episode alpha
        :param gamma:
        :param state:
        :param total_rw:
        :return: Una tupla que:
         [0]: indica el estado final del episodio
         [1]: Si debe terminar la corrida
         [2]: El total reward
        """
        action = self.choose_action(state)
        next_state, reward, done = self.act(state, action)
        total_rw += reward
        current_q = self.q(state)[action.value]
        next_row = self.q(next_state)
        
        quality = self.compute_bellman_optimality_equation(current_q, next_row, alpha, reward, gamma)
        self.q(state)[action.value] = quality
        return next_state, done, total_rw
    
    def train(self, n_episodes: int, max_episode_steps: int, gamma: float, alphas: np.ndarray):
        for e in range(n_episodes):
            state = self
            total_reward = 0
            alpha = alphas[e]
            for _ in range(max_episode_steps):
                state, done, total_reward = self.run_episode(alpha, gamma, state, total_reward)
                if done:
                    break
            print(f"Episode {e + 1}: total reward -> {total_reward}")


if __name__ == '__main__':
    grid_ = [
        [GridItem.ICE_CREAM, GridItem.EMPTY],
        [GridItem.ZOMBIE, GridItem.CAR]
    ]
    initial_car_pos = [1, 1]
    start_state = State(grid=grid_, car_pos=initial_car_pos)
    start_state.train(N_EPISODES, MAX_EPISODE_STEPS, GAMMA, ALPHAS)
