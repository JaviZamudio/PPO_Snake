from uuid import uuid4

import pygame

from Models import Critic, flatten_state
from Agents import PPOAgent
from configs import DEBUG_MODE, STATE_SIZE

class Player:
    """Base class for a player in the Snake game. Subclasses should implement the get_move method."""

    def get_move(self, state: list[list[int]], grid_size: int) -> int:
        """Given the current game state, return a direction code (0=up, 1=right, 2=down, 3=left)."""
        raise NotImplementedError("Subclasses must implement get_move method")

    def handle_eat(self, state):
        """Called when the snake eats an apple. Can be used to update internal state if needed."""
        pass

    def handle_bite(self, state):
        """Called when the snake bites itself. Can be used to update internal state if needed."""
        pass

    def handle_crash(self, state):
        """Called when the snake crashes into a wall. Can be used to update internal state if needed."""
        pass

    def handle_invalid_move(self, state):
        """Called when the snake attempts an invalid move (e.g., reversing direction). Can be used to update internal state if needed."""
        pass

class HumanPlayer(Player):
    """Human-controlled player that returns direction codes on WASD input.

    Directions are encoded as:
    0 = up, 1 = right, 2 = down, 3 = left
    """

    KEY_TO_DIR = {
        pygame.K_w: 0,
        pygame.K_d: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
    }

    def __init__(self):
        self.critic = Critic("snake_critic.keras")

    def get_move(self, state, grid_size):
        """Block until a valid movement key is pressed, then return direction code."""

        for r in range(STATE_SIZE):
            for c in range(STATE_SIZE):
                if r >= grid_size or c >= grid_size:
                    state[r][c] = 1 

        if DEBUG_MODE:
            value = self.critic.predict(flatten_state(state))
            print(f"Critic value for current state: {value[0][0]:.4f}")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit("Game closed by user")
                    return None

                if event.type == pygame.KEYDOWN:
                    if event.key in self.KEY_TO_DIR:
                        return self.KEY_TO_DIR[event.key]

        return None
    
class AIPlayer(Player):
    """Placeholder for an AI-controlled player. Currently returns a random valid move."""

    def __init__(self, ppo_agent: PPOAgent):
        self.ppo_agent = ppo_agent
        self.id = str(uuid4())  # Unique ID for this player to track rewards in PPOAgent
        
    def get_move(self, state, grid_size):
        self.ppo_agent.set_reward(0.05, self.id)  # Set alive reward for taking a step
        return self.ppo_agent.select_action(state, grid_size, self.id)
    
    def handle_eat(self, state):
        self.ppo_agent.set_reward(5.0, self.id)

    def handle_bite(self, state):
        self.ppo_agent.set_reward(-1.0, self.id)
        self.ppo_agent.handle_game_end(self.id)
    
    def handle_crash(self, state):
        self.ppo_agent.set_reward(-1.0, self.id)
        self.ppo_agent.handle_game_end(self.id)

    def handle_invalid_move(self, state):
        self.ppo_agent.set_reward(-0.5, self.id)
        