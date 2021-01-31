"""
Adaptation of the vanilla DDPG code to allow for FiGAR modification as presented
in the FiGAR paper https://arxiv.org/pdf/1702.06054.pdf
"""


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We have to modify the Actor to also generate the repetition output
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, rep_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, rep_dim)

    def forward(self, state):
        # As suggested by the FiGAR authors, the input layer is shared
        shared = F.relu(self.l1(state))
        a = F.relu(self.l2(shared))

        r = F.relu(self.l5(shared))
        return self.max_action * torch.tanh(self.l3(a)), F.log_softmax(self.l6(r), dim=1)


# The Critic has to be modified to be able to accept the additional repetition output of the actor
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, repetition_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim + repetition_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action, repetition):
        q = F.relu(self.l1(torch.cat([state, action, repetition], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, repetition_dim, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action, repetition_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, repetition_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        # The select action method has to be adjusted to also sample from the repetition distribution
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, repetition_prob = self.actor(state)
        repetition_dist = torch.distributions.Categorical(repetition_prob)
        repetition = repetition_dist.sample()
        return (action.cpu().data.numpy().flatten(),
                repetition.cpu().data.numpy().flatten(),
                repetition_prob.cpu().data.numpy().flatten())

    def train(self, replay_buffer, batch_size=100):
        # The train method has to be adapted to take changes to Actor and Critic into account
        # Sample replay buffer
        state, action, repetition, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, *self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action, repetition)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, *self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
