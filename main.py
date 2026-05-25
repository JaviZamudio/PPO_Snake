from time import sleep
from Game import Game
from Agents import PPOAgent
from Players import AIPlayer, HumanPlayer
import configs

if __name__ == "__main__":
    grid_size = 10

    ppo_agent = PPOAgent(
        actor_path="snake_actor.keras",
        critic_path="snake_critic.keras",
        memories_until_training=grid_size**2 * 2**5,
        epsilon=0.1,
        # human_input=True,
    )

    ai_player = AIPlayer(ppo_agent)

    outer_corners = [
        (0, 0),  # top left
        (0, grid_size - 1),  # top right
        (grid_size - 1, 0),  # bottom left
        (grid_size - 1, grid_size - 1),  # bottom right
    ]
    inner_corners = [
        (1, 1),  # top left
        (1, grid_size - 2),  # top right
        (grid_size - 2, 1),  # bottom left
        (grid_size - 2, grid_size - 2),  # bottom right
    ]
    outer_centers = [
        (0, grid_size // 2),  # top center
        (grid_size - 1, grid_size // 2),  # bottom center
        (grid_size // 2, 0),  # left center
        (grid_size // 2, grid_size - 1),  # right center
    ]
    inner_centers = [
        (1, grid_size // 2),  # top center
        (grid_size - 2, grid_size // 2),  # bottom center
        (grid_size // 2, 1),  # left center
        (grid_size // 2, grid_size - 2),  # right center
    ]
    center = (grid_size // 2, grid_size // 2)

    preferred_apple_positions = (
        outer_corners + inner_corners + outer_centers + inner_centers + [center]
    )

    game = Game(
        ai_player,
        grid_size=grid_size,
        # initial_apple_pos=(grid_size // 2, grid_size // 2 - 3),
        # preferred_apple_positions=preferred_apple_positions,
    )

    while True:
        game.run_game_loop()
        game.reset()
