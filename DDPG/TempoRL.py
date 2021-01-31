"""
Adaptation of the vanilla DDPG code to allow for TempoRL modification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# we use exactly the same Actor and Critic networks and training methods for both as in the vanilla implementation
from DDPG.vanilla import DDPG as VanillaDDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, skip_dim, non_linearity=F.relu):
        super(Q, self).__init__()
        # We follow the architecture of the Actor and Critic networks in terms of depth and hidden units
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, skip_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class DDPG(VanillaDDPG):
    def __init__(self, state_dim, action_dim, max_action, skip_dim, discount=0.99, tau=0.005):
        # We can fully reuse the vanilla DDPG and simply stack TempoRL on top
        super(DDPG, self).__init__(state_dim, action_dim, max_action, discount, tau)

        # Create Skip Q network
        self.skip_Q = Q(state_dim, action_dim, skip_dim)
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    def select_skip(self, state, action):
        """
        Select the skip action.
        Has to be called after select_action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.skip_Q(torch.cat([state, action], 1)).cpu().data.numpy().flatten()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * np.power(self.discount, skip + 1) * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()

    def save(self, filename):
        super().save(filename)

        torch.save(self.skip_Q.state_dict(), filename + "_skip")
        torch.save(self.skip_optimizer.state_dict(), filename + "_skip_optimizer")

    def load(self, filename):
        super().load(filename)

        self.skip_Q.load_state_dict(torch.load(filename + "_skip"))
        self.skip_optimizer.load_state_dict(torch.load(filename + "_skip_optimizer"))
