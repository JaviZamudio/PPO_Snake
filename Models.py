from typing import TypedDict

import numpy as np
import tensorflow as tf

from configs import print_debug


class Memory(TypedDict):
    state: np.ndarray  # Flattened state of the board before taking the action
    next_state: (
        np.ndarray | None
    )  # Flattened state of the board after taking the action
    action: int  # Column index (0-6) where the piece was dropped
    action_prob: float  # Probability of the action taken according to the actor at the time of action selection
    reward: float | None  # Reward received after taking the action
    value_estimation: float  # Value estimation of the state before taking the action (from the critic)
    advantage: float  # Advantage calculated by the critic for the action taken
    next_value_estimation: float  # Value estimation of the next state (from the critic)
    player_id: (
        str | None
    )  # Identifier for the player who took the action (e.g., "Player 1" or "Player 2")


class Model:
    def __init__(self, path):
        self.model: tf.keras.Model
        self.path = path
        self.load(path)

    def load(self, path: str):
        try:
            self.model = tf.keras.models.load_model(path)
            print(f"Model {self.__class__.__name__} loaded successfully from {path}.")
        except Exception as e:
            print(f"Failed to load model: {e}. Creating a new model.")
            self.model = self.create_model()

    def create_model(self) -> tf.keras.Model: ...

    def save(self, path: str):
        try:
            self.model.save(path)
            print(f"Model {self.__class__.__name__} saved successfully to {path}.")
        except Exception as e:
            print(f"Failed to save model: {e}.")

    def predict(self, state: np.ndarray):
        return self.model.predict(state, verbose=0)

    def get_model(self):
        return self.model


class Actor(Model):
    def __init__(self, path, leash_threshold=0.2):
        super().__init__(path)
        self.leash_threshold = leash_threshold  # Threshold for leash mechanism (e.g., 20% deviation from expected reward)
        self.entropy = 0.01  # Entropy coefficient for encouraging exploration

    def create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Input(shape=(412,), dtype=tf.float32)
        )  # 20x20 board cells flattened (values 0=empty, 1=body, 2=head, 3=apple) + 8 surrounding cells + 4 direction indicators
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(
            tf.keras.layers.Dense(4, activation="softmax")
        )  # 4 possible actions: up, right, down, left
        return model

    def train_actor(self, memory: list[Memory], save=True, load=False):
        # Load the current model weights before training
        if load:
            self.load(self.path)

        if not save:
            return

        epoch_count = 0
        learning_rate = 0.00005
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        while epoch_count < 5:
            epoch_count += 1

            with tf.GradientTape() as tape:
                epoch_loss_list = []
                for experience in memory:
                    state_tensor: tf.Tensor = tf.convert_to_tensor(
                        experience["state"], dtype=tf.float32
                    )
                    advantage: float = experience["advantage"]
                    original_action_prob: float = experience["action_prob"]
                    original_action_index: int = experience["action"]

                    # Predict new action and calculate drift
                    new_action_probs: tf.Tensor = self.model(
                        state_tensor, training=True
                    )[0]
                    new_prob = new_action_probs[original_action_index]
                    drift = new_prob / original_action_prob

                    lower_bound = 1 - self.leash_threshold
                    upper_bound = 1 + self.leash_threshold

                    clipped_drift: tf.Tensor = tf.clip_by_value(
                        drift, lower_bound, upper_bound
                    )

                    # Calculate the PPO loss using the clipped drift
                    loss = -tf.minimum(drift * advantage, clipped_drift * advantage)
                    # Add entropy bonus to encourage exploration
                    # loss = loss - self.entropy * tf.reduce_sum(
                    #     new_action_probs * tf.math.log(new_action_probs + 1e-8)
                    # )
                    epoch_loss_list.append(loss)

                # Calculate the mean loss for the epoch
                final_loss = tf.reduce_mean(epoch_loss_list)

            # Apply the epoch loss
            gradients = tape.gradient(final_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Save the model after training
        if save:
            self.save(self.path)


class Critic(Model):
    def __init__(self, path, gamma=0.99):
        super().__init__(path)
        self.gamma = gamma
        
        # Compile the model with mean squared error loss and an optimizer
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Input(shape=(412,), dtype=tf.float32)
        )  # 20x20 board cells flattened (values 0=empty, 1=body, 2=head, 3=apple) + 8 surrounding cells + 4 direction indicators
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))  # state value
        return model

    def run_critics(self, memories: list[Memory]) -> list[Memory]:
        # states are already flattened and made into np arrays in the memory...
        # but we need to stack them into a single array for batch prediction
        states = np.array([experience["state"] for experience in memories]).reshape(
            len(memories), -1
        )
        next_states = np.array(
            [experience["next_state"] for experience in memories]
        ).reshape(len(memories), -1)

        value_estimations = self.predict(states)
        next_value_estimations = self.predict(next_states)

        # For each experience in memory, calculate the advantage and update the experience with the advantage
        for i, experience in enumerate(memories):
            state = experience["state"]
            reward = experience["reward"]
            next_state = memories[i + 1]["state"] if i + 1 < len(memories) else state

            # Get value estimation for current state
            value_estimation = value_estimations[i]

            # Get value estimation for next state if it exists
            next_value_estimation = next_value_estimations[i]

            # Calculate advantage
            advantage = reward + (self.gamma * next_value_estimation) - value_estimation

            # Update experience with advantage, value estimation
            memories[i]["advantage"] = advantage
            memories[i]["value_estimation"] = value_estimation
            memories[i]["next_value_estimation"] = next_value_estimation

        return memories

    def train_critic(self, memories: list[Memory], save=True):
        # Train the critic to predict the value estimation + advantage
        states = np.array([experience["state"] for experience in memories]).reshape(
            len(memories), -1
        )

        # The target for the critic is the value estimation + advantage (which should equal the reward + discounted next value estimation)
        targets = np.array(
            [
                experience["value_estimation"] + experience["advantage"]
                for experience in memories
            ]
        ).reshape(-1, 1)

        self.model.fit(states, targets, epochs=5, verbose=0)

        if save:
            self.save(self.path)  # Save the updated model after training


def flatten_state(state: list[list[int]]) -> np.ndarray:
    # Find head and apple positions in the state
    head_pos = None
    head_surroundings = [0] * 8  # Initialize surroundings with 8 zeros
    apple_pos = None
    for r in range(len(state)):
        for c in range(len(state[r])):
            cell_value = state[r][c]
            if cell_value == 2:  # HEAD
                head_pos = (r, c)
            if cell_value == 3:  # APPLE
                apple_pos = (r, c)
        if head_pos and apple_pos:
            break

    # If head is found, get the values of the 8 surrounding cells (if out of bounds, treat surrounding as body (1))
    if head_pos:
        hr, hc = head_pos
        directions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]
        for i, (dr, dc) in enumerate(directions):
            r, c = hr + dr, hc + dc
            if 0 <= r < len(state) and 0 <= c < len(state[r]):
                head_surroundings[i] = state[r][c]
            else:
                head_surroundings[i] = 1  # Out of bounds is treated as body

    # pass direction indicators (up, right, down, left) based on apple position relative to head
    direction_indicators = [0, 0, 0, 0]
    if head_pos and apple_pos:
        hr, hc = head_pos
        ar, ac = apple_pos
        if ar < hr:
            direction_indicators[0] = 1  # Apple is up
        elif ar > hr:
            direction_indicators[2] = 1  # Apple is down
        if ac > hc:
            direction_indicators[1] = 1  # Apple is right
        elif ac < hc:
            direction_indicators[3] = 1  # Apple is left

    # Print state, head surroundings, and direction indicators for debugging
    print_debug("State:")
    for row in state:
        print_debug(row)
    print_debug(f"Head surroundings:")
    print_debug(
        head_surroundings[-1], head_surroundings[0], head_surroundings[1]
    )  # up-left, up, up-right
    print_debug(
        head_surroundings[6], "X", head_surroundings[2]
    )  # left, X (head), right
    print_debug(
        head_surroundings[5], head_surroundings[4], head_surroundings[3]
    )  # down-left, down, down-right
    print_debug(
        f"Direction indicators Up: {direction_indicators[0]}, Right: {direction_indicators[1]}, Down: {direction_indicators[2]}, Left: {direction_indicators[3]}"
    )

    # Flatten the 20x20 state into a 400-length array and convert to float32
    flat_state = np.array(state).flatten().astype(np.float32).reshape(1, -1)
    flat_state = np.concatenate(
        [
            flat_state,
            np.array(head_surroundings).reshape(1, -1),
            np.array(direction_indicators).reshape(1, -1),
        ],
        axis=1,
    )
    return flat_state
