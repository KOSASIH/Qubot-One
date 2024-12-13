# src/ai/models/reinforcement_learning.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95):
        """Initialize the Q-learning agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            learning_rate (float): The learning rate for Q-learning.
            discount_factor (float): The discount factor for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, epsilon):
        """Choose an action based on epsilon-greedy policy.

        Args:
            state (int): The current state.
            epsilon (float): The exploration rate.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value based on the action taken.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state after taking the action.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.learning_rate * (td_target - self.q_table[state][action])

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """Initialize the Deep Q-Network.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """Initialize the DQN agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            learning_rate (float): The learning rate for the optimizer.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The initial exploration rate.
            epsilon_decay (float): The decay rate for epsilon.
            min_epsilon (float): The minimum value for epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Choose an action based on the current state and epsilon-greedy policy.

        Args:
            state (np.array): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self, batch_size):
        """Train the model using a batch of experiences from memory.

        Args:
            batch_size (int): The number of experiences to sample from memory.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.discount_factor * torch.max(self.target_model(next_state_tensor)).item()
            target_f = self.model(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(env, num_episodes=1000, batch_size=32):
    """Train the DQN agent in the given environment.

    Args:
        env: The environment to train the agent in.
        num_episodes (int): The number of episodes to train.
        batch_size (int): The batch size for training.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.update_target_model()

        print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Example usage
if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v1')  # Example environment
    train_dqn_agent(env)
    env.close()
    
