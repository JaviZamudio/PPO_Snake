from time import sleep
from Game import Game
from Agents import PPOAgent
from Players import AIPlayer, HumanPlayer
from configs import DEBUG_MODE


if __name__ == "__main__":
    human_player = HumanPlayer()

    ppo_agent = PPOAgent(
        actor_path="snake_actor.keras",
        critic_path="snake_critic.keras",
        memories_until_training=1024,
    )
    ai_player = AIPlayer(ppo_agent)
    DEBUG_MODE = False
    grid_size = 10
    game = Game(ai_player, grid_size=grid_size, prefered_apple_positions=[
        (1, 1),
        (1, grid_size - 2),
        (grid_size - 2, 1),
        (grid_size - 2, grid_size - 2),
    ])

    while True:
        game.run_game_loop()
        game.reset()
