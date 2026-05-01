from Models import Actor, Critic, Memory, flatten_state


import numpy as np

import Players
from configs import STATE_SIZE, print_debug


class PPOAgent:
    def __init__(
        self,
        actor_path,
        critic_path,
        leash_threshold=0.2,
        gamma=0.99,
        memories_until_training=1,
        epsilon=0.1,
        human_input=False,
    ):
        self.actor = Actor(actor_path, leash_threshold)
        self.critic = Critic(critic_path, gamma)
        self.memories: list[Memory] = []
        self.memories_until_training = memories_until_training
        self.epsilon = epsilon
        self.human_input = human_input

    def update_next_states(self, player_id: str | None = None):
        """
        Meant to run every time a terminal state is reached.
        It updates the next_state field of each memory experience with the state of the next experience with the same player_id.
        If there is no next experience with the same player_id, it sets the next_state to the same as the current state (for terminal states).

        If never used, the next_state field will remain None, and the critic will throw an error

        @param player_id: If provided, only updates next_state for experiences of the specified player_id. If None, updates next_state for all experiences.
        """

        for i, experience in enumerate(self.memories):
            if experience["next_state"] is None and (
                experience["player_id"] == player_id or player_id is None
            ):
                current_player_id = experience["player_id"]
                next_state_found = False

                for j in range(i + 1, len(self.memories)):
                    if self.memories[j]["player_id"] == current_player_id:
                        experience["next_state"] = self.memories[j]["state"]
                        next_state_found = True
                        break

                if not next_state_found:
                    experience["next_state"] = experience["state"]

    def select_action(self, state: list[list[int]], grid_size, player_id: str) -> int:
        # If previous reward hasn't been set, make it 0
        self.set_reward(0.0, player_id, override=False)

        # Make everything outside size x size be treated as body (1) for the agent's perception of the state
        for r in range(STATE_SIZE):
            for c in range(STATE_SIZE):
                if r >= grid_size or c >= grid_size:
                    state[r][c] = 1  # Treat out of bounds as body (1)

        flat_state = flatten_state(state)

        # Predict action probabilities from the actor
        action_probs = self.actor.predict(flat_state)[0]

        # Get the critic's value estimation for this state (before taking the action)
        value_estimation = 0.0
        if self.human_input:
            value_estimation = self.critic.predict(flat_state)[0][0]
            print(f" ------ \nPredicted state value estimation: {value_estimation:.4f}")
            print(f"Action probabilities: {action_probs}")

        # Epsilon-greedy action selection for exploration
        if self.human_input:
            action = Players.HumanPlayer.static_get_move()
        elif np.random.rand() < self.epsilon:
            # Choose a random action (exploration)
            action = np.random.choice(len(action_probs))
            print(f"Exploring: selected random action {action}")
            self.epsilon *= 0.995  # Decay epsilon after each exploration
        else:
            action = np.random.choice(len(action_probs), p=action_probs)        

        # Create a memory experience for this action selection
        memory_experience: Memory = {
            "action": action,
            "action_prob": action_probs[action],
            "state": flat_state,
            "player_id": player_id,
            "reward": None,  # Will be updated later when the reward is received
            "next_state": None,  # Will be updated later by update_next_states()
            "advantage": 0.0,  # Will be updated later by the critic
            "value_estimation": value_estimation,  # Will be updated later by the critic
            "next_value_estimation": 0.0,  # Will be updated later by the critic
        }

        # if self.human_input:
        #     print("---- Storing memory")
        #     print(f"Chosen action: {action} with probability {action_probs[action]:.4f} (probabilities: {action_probs})")
        #     print(f"Predicted state value estimation: {value_estimation:.4f}")

        self.memories.append(memory_experience)

        return action

    def set_reward(self, reward: float, player_id: str, override: bool = False):
        # Set the reward for the most recent memory experience of the specified player_id
        for experience in reversed(self.memories):
            if experience["player_id"] == player_id:
                if override or experience["reward"] == None:
                    experience["reward"] = reward
                    if self.human_input:
                        print(
                            f"Setting reward for players' last action: {reward} (override={override})"
                        )
                break

    def handle_game_end(self, player_id: str, ready_to_train: bool = True):
        # Update next states for all experiences of this player_id
        self.update_next_states(player_id)

        if len(self.memories) > self.memories_until_training or self.human_input:
            print("Training PPO agent...")
            self.train()

    def clear_incomplete_experiences(self):
        # Remove any experiences that don't have a reward set (incomplete games)
        self.memories = [exp for exp in self.memories if exp["reward"] is not None]

    def train(self):
        # First, update next states for all experiences
        self.update_next_states()

        # Then, clear any incomplete experiences that don't have rewards set (incomplete games)
        self.clear_incomplete_experiences()

        # Then run the critic to calculate advantages
        updated_memories = self.critic.run_critics(self.memories)

        # Then train the critic with the updated memories
        self.critic.train_critic(updated_memories)

        # Finally, train the actor with the updated memories
        self.actor.train_actor(updated_memories)

        # Clear memories after training
        self.memories.clear()
