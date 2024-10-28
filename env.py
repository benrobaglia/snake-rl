import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=20, max_steps=100):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_steps = 0
        # Actions: 0: Straight, 1: Right, 2: Left
        self.action_space = spaces.Discrete(3)
        # State = Observation = 0: Empty space 1: Snake body 2: Snake head 3: Fruit
        # self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.int8)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(11,), dtype=np.float32)
        # Initialize the game state variables
        self.snake = None
        self.food_position = None
        self.direction = None
        self.score = 0

        # Rendering variables
        self.render_mode = 'human'
        self.window = None
        self.clock = None

        # Initialize the random number generator
        self.np_random = None

        self.direction_deltas = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }

        # Seed the environment
        self.seed()

        # Reset the environment to start
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        # Snake head position
        head_x, head_y = self.snake[0]

        # Food position
        food_x, food_y = self.food_position

        # Direction encoding
        direction = self.direction

        # Relative position to food
        relative_x = food_x - head_x
        relative_y = food_y - head_y

        # Obstacle detection
        obstacle_up = int(head_y == 0 or self.grid[head_x, head_y - 1] != 0)
        obstacle_right = int(head_x == self.grid_size - 1 or self.grid[head_x + 1, head_y] != 0)
        obstacle_down = int(head_y == self.grid_size - 1 or self.grid[head_x, head_y + 1] != 0)
        obstacle_left = int(head_x == 0 or self.grid[head_x - 1, head_y] != 0)

        # Construct the observation vector
        observation = np.array([
            head_x, head_y,
            food_x, food_y,
            direction,
            relative_x, relative_y,
            obstacle_up, obstacle_right, obstacle_down, obstacle_left
        ], dtype=np.float32)

        return observation

    def _is_collision(self, position):
        """
        Check if the given position (y, x) is a collision with the snake's body or the wall.
        """
        y, x = position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if position in self.snake[1:]:
            return True
        else:
            return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Initialize the snake at random position
        start_x = np.random.randint(self.grid_size)
        start_y = np.random.randint(self.grid_size)
        self.snake = [(start_y, start_x)]  # List of (row, column) tuples
        self.grid[start_y, start_x] = 2  # Snake head
        self.current_steps = 0

        # Place the initial food
        self._place_food()

        # Set the initial direction (e.g., up)
        self.direction = 0  # 0: up, 1: right, 2: down, 3: left

        self.score = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def _place_food(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        food_position = empty_cells[self.np_random.choice(len(empty_cells))]
        self.food_position = food_position
        self.grid[food_position] = 3  # Place food on the grid

    def step(self, action):

        self.current_steps += 1

        # Update the direction based on the action
        if action == 0:
            # Continue straight
            new_direction = self.direction
        elif action == 1:
            # Turn right
            new_direction = (self.direction + 1) % 4
        elif action == 2:
            # Turn left
            new_direction = (self.direction - 1) % 4
        else:
            raise ValueError(f"Invalid action: {action}")

        self.direction = new_direction

        # Move the snake's head
        head_y, head_x = self.snake[0]
        delta_y, delta_x = self.direction_deltas[self.direction]
        new_head = (head_y + delta_y, head_x + delta_x)

        # Check if there is a collision
        terminated = self._is_collision(new_head)
        truncated = self.current_steps >= self.max_steps * len(self.snake)
        
        if terminated or truncated:
            observation = self._get_observation()
            reward = -10
            return observation, reward, terminated, truncated, {}
        else:
            terminated = False
            truncated = False
    
            # Move the snake
            self.grid[new_head] = 2
            self.snake.insert(0, new_head)

            # Check if the snake has eaten the food
            if new_head == self.food_position:
                reward = 10
                self.score += 1
                self._place_food()
            else:
                # Remove the tail
                tail = self.snake.pop()
                self.grid[tail] = 0
                reward = 0

            if len(self.snake) > 1:
                self.grid[self.snake[1]] = 1

            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

    def _get_action_from_direction(self, desired_direction):
        """
        Map the desired absolute direction to a relative action based on the current direction.

        Directions:
        0: Up
        1: Right
        2: Down
        3: Left
        """
        current_direction = self.direction

        if desired_direction == current_direction:
            return 0  # Straight
        elif (desired_direction - current_direction) % 4 == 1:
            return 1  # Turn right
        elif (desired_direction - current_direction) % 4 == 3:
            return 2  # Turn left
        else:
            # Opposite direction or invalid turn; keep moving straight
            return 0

    def render(self, mode='human'):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * 20, self.grid_size * 20))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 25)

        # Handle events to allow closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.window.fill((0, 0, 0))
        # Draw the grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_value = self.grid[y, x]
                if cell_value == 0:
                    continue  # Empty cell; skip drawing
                elif cell_value == 1:
                    color = (0, 255, 0)  # Snake body - Green
                elif cell_value == 2:
                    color = (0, 200, 0)  # Snake head - Darker Green
                elif cell_value == 3:
                    color = (255, 0, 0)  # Food - Red
                else:
                    color = (255, 255, 255)  # Just in case

                rect = pygame.Rect(x * 20, y * 20, 20, 20)
                pygame.draw.rect(self.window, color, rect)

        # Optionally, draw the score
        score_text = self.font.render(
            f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, [0, 0])

        # Update the display
        pygame.display.flip()
        self.clock.tick(10)
