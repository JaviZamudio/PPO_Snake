from time import sleep
from Game import Game
from Agents import PPOAgent
from Players import AIPlayer, HumanPlayer


if __name__ == "__main__":
    human_player = HumanPlayer()

    ppo_agent = PPOAgent(
        actor_path="snake_actor.keras",
        critic_path="snake_critic.keras",
        memories_until_training=256,
    )
    ai_player = AIPlayer(ppo_agent)

    game = Game(ai_player, grid_size=6)

    while True:
        game.run_game_loop()
        game.reset()
