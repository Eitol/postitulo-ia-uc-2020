# copy para copiar nuestros objetos de un lado para otro
from copy import deepcopy, copy
# numpy es una libreria numerica que permite facil trabajo con 
# matrices, como matlab. La usamos para trabajos numericos
import numpy as np
# para tener cosas random!
import random
# para ayudarnos con los tipos, nos ayuda a tener mas claro que retorna que
from typing import List, Tuple, Any, Union, NewType, Dict

# siempre es bueno usar una semilla cuando hacemos experimentos, es la unica
# forma confiable que tenemos para asegurarnos que nuestros experimentos 
# son reproducibles
random.seed(42)  # no importa que numero elijamos, pero lo dejamos fijo

# Vamos a definir algunas cosas con `monitos` para que nuestro mapa se
# vea mas entretenido

# Partiremos con estos
ZOMBIE = "🧟"
HERO = "🙃"
TROPHIE = "🏆"
EMPTY = "⚪"

# despues ustedes tendran que agregar estos!
BLOCK = "🚫"
KEY = "🔑"
DOOR = "🚪"
SWORD = "🗡️"

# Nuestro agente tiene que saber las condiciones del mundo donde existe
# en este mundo solo hay 4 acciones que puede hacer.
# Moverse hacia `arriba`, `abajo`, `derecha` e `izquierda`, nada mas.
# En algun otro ambiente, podriamos agregar mas acciones como saltar
# atacar, comprar, etc. Pero mantendremos la simplicidad aqui.
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
# Juntamos nuestras acciones para que queden ordenadas.
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Aqui vamos a crear nuestros tipos, esto nos ayudara a entender que hace
# cada metodo y funcion que usemos mas claramente.
Action = NewType('Action', int)
GridElement = NewType('GridElement', str)


# Aqui definimos nuestra grilla que acturara como la base de nuestro mapa
class Grid:
    # Nuestro constructor toma una lista u otra grilla y la guarda
    # se preocupa de copiarla para que no modifiquemos la anterior
    def __init__(self, grid: Union['Grid', List[List[GridElement]]] = None) -> None:
        assert grid is not None
        if isinstance(grid, list):
            self.grid = deepcopy(grid)
        elif isinstance(grid, Grid):
            self.grid = deepcopy(grid.grid)
        
        # Guardamos el tamaño de la grilla para trabajar mas rapido
        self.x_lim = len(self.grid[0])
        self.y_lim = len(self.grid)
    
    # Nuestro metodo para comparar una grilla con otra
    def __eq__(self, other: 'Grid') -> bool:
        return isinstance(other, Grid) and self.grid == other.grid
    
    # Simpre es importante que si modificamos nuestra igualdad, tambien
    # adaptemos nuestro hash
    def __hash__(self) -> int:
        return hash(str(self.grid))
    
    # Cuando imprimimos una grilla, esta se mostrara como un
    # mapa, como una matriz
    def __str__(self) -> str:
        return '\n'.join([' '.join(str(e) for e in row) for row in self.grid])
    
    # Este es un metodo muy util para indexar partes de la grilla
    def __getitem__(self, position: Tuple[int, int]) -> GridElement:
        assert type(position) == tuple
        # necesitamos 2 coordenadas para saber que hay en esa posicion
        assert len(position) == 2
        x, y = position
        # verificamos que las coordenadas esten dentro de la grilla
        assert 0 < x <= self.x_lim
        assert 0 < y <= self.y_lim
        # retornamos el elemento que hay en esa posicion
        return self.grid[self.y_lim - y][x - 1]
    
    # Este es un metodo muy util para insertar elementos en la grilla
    def __setitem__(self, position: Tuple[int, int], value: GridElement) -> None:
        assert type(position) == tuple
        assert len(position) == 2
        x, y = position
        assert 0 < x <= self.x_lim
        assert 0 < y <= self.y_lim
        # igual que antes, pero ahora asignamos un elemento en vez de retornarlo
        self.grid[self.y_lim - y][x - 1] = value
    
    # Una forma `fancy` de acceder a variables de una clase sin un `getter`
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.x_lim, self.y_lim)


# La clase `State` representara un estado del heroe
class State:
    # En nuestro constructor solo asignamos nuestras variables
    def __init__(self, grid: Union[Grid, List[List[GridElement]]] = None,
                 hero_pos: Tuple[int, int] = (1, 1)) -> None:
        
        self.grid = Grid(grid=grid)
        self.x_lim, self.y_lim = self.grid.shape
        self.hero_x, self.hero_y = hero_pos
    
    # Misma forma `fancy` de acceder a la posicion del heroe
    @property
    def hero_pos(self) -> Tuple[int, int]:
        return (self.hero_x, self.hero_y)
    
    # Para dibujar nuestro mapa con el heroe en la posicion actual
    def __str__(self) -> str:
        grid = deepcopy(self.grid)
        grid[self.hero_x, self.hero_y] = HERO
        return grid.__str__()
    
    # Un estado es igual a otro si las grillas y posiciones de los heroes son
    # las mismas
    def __eq__(self, other: 'State') -> bool:
        return isinstance(other, State) and self.hero_pos == other.hero_pos and \
               self.grid == other.grid
    
    # Igual que antes, por completitud debemos implementar cuando 2
    # estados tienen el mismo hash
    def __hash__(self) -> int:
        return hash(str(self.grid) + str(self.hero_pos))
    
    # Este metodo nos ayuda a obtener que elemento se encuentra en una posicion
    # determinada. Necesitamos el estado pasado para comparar ya que no tenemos
    # historia, pero es una forma simple de implementar el mapa sin
    # mucho codigo. Recuerda que estamos en una cadena de Markov, aqui no tenemos
    # los estados pasados! Estos no afectan la decision que tomaremos ahora.
    def get_element(self, position: Tuple[int, int], state: 'State') -> GridElement:
        assert type(position) == tuple
        assert len(position) == 2
        x, y = position
        assert 0 < x <= self.x_lim
        assert 0 < y <= self.y_lim
        
        # Por limitaciones de la implementacion, debemos saber si el heroe se
        # movio a la posicion en la que esta, o estaba ahi desde antes
        # otra implementacion podria solucionar este problema de mejor manera
        # pero es mas compleja de entender
        if position == state.hero_pos:
            return HERO
        return self.grid[x, y]
    
    # De nuestras acciones tenemos que elegir una y actuar acorde a ella.
    # por ejemplo, si le pedimos al estado que suba, entonces tenemos que 
    # enviar la accion `UP`.
    # Cuando llamamos a este metodo, creamos un nuevo estado con la
    # accion aplicada
    def action_dispatch(self, action: Action) -> 'State':
        if action == UP:
            return self.moveUp()
        elif action == DOWN:
            return self.moveDown()
        elif action == LEFT:
            return self.moveLeft()
        elif action == RIGHT:
            return self.moveRight()
        else:
            raise ValueError(f"Unknown action {action}")
    
    # Este metodo solo copia el estado actual y crea uno nuevo para aplicar
    # los cambios pedidos por la accion ingresada
    def register(self) -> 'State':
        past_state = copy(self)
        return State(grid=past_state.grid, hero_pos=past_state.hero_pos)
    
    # Los siguientes metodos mueven nuestro personaje en las direcciones
    # que definimos antes, arriba, abajo, derecha e izquierda
    
    def moveUp(self) -> 'State':
        new_state = self.register()
        new_state.hero_y = new_state.hero_y + 1 if new_state.hero_y < new_state.y_lim else new_state.hero_y
        
        return new_state
    
    def moveDown(self) -> 'State':
        new_state = self.register()
        new_state.hero_y = new_state.hero_y - 1 if new_state.hero_y > 1 else new_state.hero_y
        
        return new_state
    
    def moveRight(self) -> 'State':
        new_state = self.register()
        new_state.hero_x = new_state.hero_x + 1 if new_state.hero_x < new_state.x_lim else new_state.hero_x
        
        return new_state
    
    def moveLeft(self) -> 'State':
        new_state = self.register()
        new_state.hero_x = new_state.hero_x - 1 if new_state.hero_x > 1 else new_state.hero_x
        
        return new_state


# Creamos una lista de listas (una matriz) que represente a nuestro mapa
mapa_ejemplo = [
    [TROPHIE, EMPTY, EMPTY],
    [ZOMBIE, EMPTY, ZOMBIE],
    [EMPTY, EMPTY, EMPTY]
]
# Digamos que nuestro heroe parte en la posicion (1,1)
estado_ejemplo = State(grid=mapa_ejemplo, hero_pos=(1, 1))

# Veamos como se ve!
print(estado_ejemplo)


# Definimos una funcion que represente un `acto`. Es decir
# dado un estado y una accion, que ocurre.
# Este "que ocurre" es bastante variado, podemos movernos,
# ganar puntaje, perder el juego, o cualquier cosa que decidamos

# Aqui es donde debemos decidir cuando y cuanta recompensa o castigo
# debemos dar a nuestro agente.
# Esta funcion retornara 3 cosas. El nuevo estado en que quedo nuestro heroe,
# una recompensa por su esfuerzo (puede ser negativa), y un booleano indicando
# si el juego termino o no. Este `termino` puede ser porque ganamos o perdimos.
def act(state: State, action: Action, condition=None):
    # Le decimos a nuestro estado que se mueva en la direccion pedida
    # esto nos da un nuevo estado
    new_state = state.action_dispatch(action)
    
    # ahora le pedimos al nuevo estado que nos diga que hay 
    # en la posicion que quedamos
    # De nuevo, por un tema de implementacion tenemos que saber si donde estamos
    # ahora estaba ocupado por otro elemento, o si siempre estuvimos nosotros ahi.
    grid_item = new_state.get_element(new_state.hero_pos, state)
    
    # Si nos encontramos un zombie, que hacemos?
    if grid_item == ZOMBIE:
        
        # como no tenemos como defendernos, perdemos
        # le daremos una recompensa negativa al agente por que se equivoco
        
        # le daremos -100 como recompensa.
        reward = -100
        # el juego se acabo...
        is_done = True
    
    # Si nos encontramos con un trofeo, que hacemos?
    elif grid_item == TROPHIE:
        # pues ganamos! Le damos 1000 de recompensa al agente
        
        # le daremos 1000 como recompensa.
        reward = 1000
        # el juego se acabo... pero ahora ganamos!
        is_done = True
    
    # Si el espacio esta vacio, que hacemos?
    elif grid_item == EMPTY:
        # nada, simplemente nos movemos a ese lugar.
        
        # por que es negativa la recompensa? Lo veremos en la pregunta 7!
        reward = -1
        # no se ha terminado el juego
        is_done = False
    
    # Si el heroe ya estba en ese espacio, que hacemos?
    elif grid_item == HERO:
        # nada, simplemente nos quedamos igual.
        
        # por que es negativa la recompensa? Lo veremos en la pregunta 7!
        reward = -1
        # no se ha terminado el juego
        is_done = False
    
    else:
        raise ValueError(f"Unknown grid item {grid_item}")
    
    return new_state, reward, is_done


# Declaramos nuestra tabla `q_table` como un diccionario vacio
# Nuestra tabla se vera de la siguiente forma:
# estado: [lista de acciones posibles]
# Esta lista de acciones posibles es una lista de puntajes para cada
# accion dado un estado.
q_table = {}


# Luego hacemos nuestra funcion de busqueda `q`.
# Esta tiene 2 funciones:
# 1. Dado  un estado, retorna una lista con los puntajes para cada accion en ese
# estado. Es decir, una lista con puntajes para decidir que hacer
# 2. Dado un estado y una accion, retorna el puntaje asociado a realizar
# esa accion en ese estado.
def q(state: State, action: Action = None) -> Union[float, np.ndarray]:
    if state not in q_table:
        # Si no hemos visto este estado, lo creamos
        # como no sabemos que hacer aun, decimos que todas las acciones
        # tienen beneficio 0, ya que no lo hemos evaluado aun
        q_table[state] = np.zeros(len(ACTIONS))
    
    if action is None:
        return q_table[state]
    
    return q_table[state][action]


# Este es un metodo conveniente para no estar borrando manualmente
# la tabla cada vez que queremos hacer algo nuevo
def reset_table():
    q_table = {}


# El total de episodios donde nuestro agente aprendera
N_EPISODES = 20

# El maximo numero de pasos por episodio
MAX_EPISODE_STEPS = 10

# Debemos definir nuestro conjunto de pesos de entrenamiento.
# En un comienzo nuestro agente aprendera mucho, ya que sus primeros
# acercamientos al juego son mas valiosos. Pero mientras mas veces jugamos
# lo que aprendemos por cada jugada es cada vez menos. Es importante hacer esta
# diferencia o una jugada muy avanzada, por intentar explorar, podria
# arruinar todo lo que habiamos aprendido antes.

# Siempre aprenderemos aun que sea un poco
MIN_ALPHA = 0.02
# Aprenderemos desde TODO, hasta un 2% de lo que veamos.
# Esta decision es arbitraria, intenten cambiarlo a ver que pasa!
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)

# Un factor de descuento. Esto lo usamos para balancear entre la recompensa
# maxima a corto plazo, o a largo plazo. Si lo dejamos solo a corto plazo
# es poco probable que aprendamos algo util a futuro. Pero si lo dejamos en
# 100% entonces estamos pensando demasiado en el futuro y no nos estamos
# preocupando del presente.
# Generalmente este valor esta entre 80 y 99%
gamma = 0.9

# Si solo nos guiamos por la mejor accion y no exploramos ni nos arriesgamos
# es poco probable que aprendamos mucho del mundo. Por esto, es importante
# poner un poco de aleatoriedad en esto. Existe un 20% de probabilidades
# de que elijamos una accion al azar dado el estado que estamos. En contraste
# existe un 80% de probabilidad de que elijamos la mejor accion que conocemos.
eps = 0.2


# Aqui simulamos la eleccion de una accion. Dado un estado, nos dice que
# accion tomar. Existe un `eps` probabilidad de que tomemos una accion
# al azar.
def choose_action(state: State) -> Action:
    if random.random() < eps:
        return random.choice(ACTIONS)
    else:
        return np.argmax(q(state))


def q_learning(start_state: State, episodes: int, steps: int,
               table: Dict[State, np.ndarray], learning_rate: np.ndarray,
               discount: float) -> None:
    for ep in range(episodes):
        
        # Creamos una copia para no modificar nuestro estado original
        state = deepcopy(start_state)
        
        # Partimos con una recompensa 0
        total_reward = 0
        
        # Cada episodio tiene una tasa de aprendizaje distinto
        alpha = learning_rate[ep]
        
        # Para cada paso, vamos a ir actualizando nuestra tabla
        # para encontrar los movimientos que nos llevaran a ganar el juego
        for _ in range(steps):
            
            # Tomamos una accion de nuestro banco de acciones
            # dado nuestro estado
            action = choose_action(state)
            
            # Llamamos un `acto`, para ver si lo hicimos bien
            # necesitamos indicarle el estado donde estamos y la accion a
            # realizar
            # Esto nos da un nuevo estado, una recompensa y nos dice
            # si se termino el juego o no.            
            next_state, reward, done = act(state, action, condition=None)
            
            # Vamos guardando nuestras recompensas
            total_reward += reward
            
            # Actualizamos nuestros estados con la formula que vimos antes
            q(state)[action] = q(state, action) + \
                               alpha * (reward + gamma * np.max(q(next_state)) - q(state, action))
            
            # estamos listos para el siguiente paso
            state = next_state
            
            # si el juego termino, dejamos los pasos y comenzamos con un
            # nuevo episodio
            if done:
                break
        
        print(f"Episode {ep + 1}: total reward -> {total_reward}")
    
    # Dibujamos nuestro mapa. Pueden dibujar lo que quieran, a ver si el


# agente se la puede
grid = [
    [TROPHIE, EMPTY, EMPTY],
    [ZOMBIE, ZOMBIE, EMPTY],
    [EMPTY, EMPTY, EMPTY]
]

# Y aqui definimos nuestro estado inicial. Pueden poner al heroe donde
# quieran, pero cuiden que no parte sobre un zombie!
initial_state = State(grid=grid, hero_pos=(1, 1))

print(initial_state)


# Vamos a dejar esto aqui para que lo completen en la pregunta 2!

# RELLENE AQUI PARA RESPONDER A LA PREGUNTA 2
class StateCondition:
    def __init__(self):  # quizas recibe algo?
        ...
        pass
    
    # esto nos permite llamar al objeto como si fuera una funcion
    def __call__(self, interaction: GridElement) -> Tuple[int, bool]:
        
        if interaction == ...:
            ...
            return ...
        
        if interaction == ...:
            ...
            return ...
        
        return ...


# por si acaso, antes de entrenar reiniciamos nuestra tabla para no tener
# informacion demas
reset_table()

# Le pasamos los argumentos a nuestra funcion de aprendizaje y estamos
# listos para ver los resultados de nuestro agente
q_learning(start_state=initial_state,
           episodes=N_EPISODES,
           steps=MAX_EPISODE_STEPS,
           table=q_table,
           learning_rate=alphas,
           discount=gamma)

# Solo en caso de que nuestro heroe se quede pegado, ponemos un maximo
# de escenarios a mostrar, aqui tenemos maximo 100
show_max = 100

# Partimos con el estado inicial, preguntemos a donde podemos movernos
possible_actions = q(initial_state)
print(initial_state)
print(f"up={possible_actions[UP]}, "
      f"down={possible_actions[DOWN]}, "
      f"left={possible_actions[LEFT]}, "
      f"right={possible_actions[RIGHT]}")

# Seleccionamos la accion con el mejor puntaje, y le pedimos al
# heroe que se mueva en esa direccion. Esto nos da un nuevo estado
s, _, done = act(initial_state, np.argmax(possible_actions))

# Mientras no haya terminado el juego, o nos hayamos pasado del maximo de
# acciones a mostar definido mas arriba, seguimos moviendonos con la 
# accion mas favorable para ese estado
while not done and show_max:
    # Mostramos el estado actual
    print(s)
    # vemos nuestras acciones
    possible_actions = q(s)
    # Que accion deberiamos tomar?
    print(f"up={possible_actions[UP]}, "
          f"down={possible_actions[DOWN]}, "
          f"left={possible_actions[LEFT]}, "
          f"right={possible_actions[RIGHT]}")
    # Elejimos la mejor accion y continuamos
    s, _, done = act(s, np.argmax(possible_actions))
    
    show_max -= 1
print(s)

# # ESCRIBA AQUI SU RESPUESTA A LA PREGUNTA 1
# # import matplotlib
# # import matplotlib.pyplot as plt
# #
# plt...
#
# grid = [
#     [TROPHIE, EMPTY, EMPTY, EMPTY, ZOMBIE, TROPHIE, BLOCK, BLOCK],
#     [ZOMBIE, ZOMBIE, BLOCK, EMPTY, BLOCK, BLOCK, BLOCK, BLOCK],
#     [DOOR, EMPTY, EMPTY, EMPTY, BLOCK, BLOCK, BLOCK, BLOCK],
#     [TROPHIE, BLOCK, EMPTY, EMPTY, BLOCK, BLOCK, BLOCK, BLOCK],
#     [ZOMBIE, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, BLOCK],
#     [EMPTY, EMPTY, BLOCK, EMPTY, BLOCK, BLOCK, EMPTY, BLOCK],
#     [EMPTY, ZOMBIE, BLOCK, KEY, BLOCK, BLOCK, EMPTY, BLOCK],
#     [EMPTY, EMPTY, EMPTY, BLOCK, BLOCK, BLOCK, SWORD, BLOCK]
# ]
# print(Grid(grid))
