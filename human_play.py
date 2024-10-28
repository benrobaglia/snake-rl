import pygame
from env import SnakeEnv

def human_play():
    # Initialize the environment
    env = SnakeEnv(grid_size=20)
    observation, info = env.reset()
    done = False

    # Initialize Pygame for capturing key presses
    pygame.init()
    pygame.display.set_caption('Snake Game - Human Play')
    clock = pygame.time.Clock()

    action = 0  # Default action (straight)
    total_reward = 0
    
    while not done:
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Map arrow keys to desired directions
                if event.key == pygame.K_UP:
                    desired_direction = 0  # Up
                elif event.key == pygame.K_RIGHT:
                    desired_direction = 1  # Right
                elif event.key == pygame.K_DOWN:
                    desired_direction = 2  # Down
                elif event.key == pygame.K_LEFT:
                    desired_direction = 3  # Left
                else:
                    desired_direction = None

                if desired_direction is not None:
                    # Get the relative action from the desired direction
                    action = env._get_action_from_direction(desired_direction)

        # Take a step in the environment
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        # Reset action to 'straight' unless a new key is pressed
        action = 0

        # Control the frame rate
        clock.tick(15)

        if done:
            print(f"Game Over! Final Score: {env.score}")
            print(f"Total Reward: {total_reward}\n")
            
            env.reset()
            total_reward = 0
            done = False



if __name__ == "__main__":
    human_play()
