import os
import json
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time
from mountain_car import MountainCarEnv
from utils import experiments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class NatureDQN(nn.Module):
    """
    DQN following the DQN implementation from
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """

    def __init__(self, in_channels=4, num_actions=18):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)


class NatureTQN(nn.Module):
    """
    Network to learn the skip behaviour using the same architecture as the original DQN but with additional context.
    The context is expected to be the chosen behaviour action on which the skip-Q is conditioned.

    This Q function is expected to be used solely to learn the skip-Q function
    """

    def __init__(self, in_channels=4, num_actions=18):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureTQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.skip = nn.Linear(1, 10)  # Context layer

        self.fc4 = nn.Linear(7 * 7 * 64 + 10, 512)  # Combination layer
        self.fc5 = nn.Linear(512, num_actions)  # Output

    def forward(self, x, action_val=None):
        # Process input image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Process behaviour context
        x_ = F.relu(self.skip(action_val))

        # Combine both streams
        x = F.relu(self.fc4(
            torch.cat([x.reshape(x.size(0), -1), x_], 1)))  # This layer concatenates the context and CNN part
        return self.fc5(x)


class NatureWeightsharingTQN(nn.Module):
    """
    Network to learn the skip behaviour using the same architecture as the original DQN but with additional context.
    The context is expected to be the chosen behaviour action on which the skip-Q is conditioned.
    This implementation allows to share weights between the behaviour network and the skip network
    """

    def __init__(self, in_channels=4, num_actions=18, num_skip_actions=10):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureWeightsharingTQN, self).__init__()
        # shared input-layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # skip-layers
        self.skip = nn.Linear(1, 10)  # Context layer
        self.skip_fc4 = nn.Linear(7 * 7 * 64 + 10, 512)
        self.skip_fc5 = nn.Linear(512, num_skip_actions)

        # behaviour-layers
        self.action_fc4 = nn.Linear(7 * 7 * 64, 512)
        self.action_fc5 = nn.Linear(512, num_actions)

    def forward(self, x, action_val=None):
        # Process input image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if action_val is not None:  # Only if an action_value was provided we evaluate the skip output layers Q(s,j|a)
            x_ = F.relu(self.skip(action_val))
            x = F.relu(self.skip_fc4(
                torch.cat([x.reshape(x.size(0), -1), x_], 1)))  # This layer concatenates the context and CNN part
            return self.skip_fc5(x)
        else:  # otherwise we simply continue as in standard DQN and compute Q(s,a)
            x = F.relu(self.action_fc4(x.reshape(x.size(0), -1)))
            return self.action_fc5(x)


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class TQ(nn.Module):
    """
    Q-Function that takes the behaviour action as context.
    This Q is solely inteded to be used for computing the skip-Q Q(s,j|a).

    Basically the same architecture as Q but with context input layer.
    """

    def __init__(self, state_dim, skip_dim, non_linearity=F.relu, hidden_dim=50):
        super(TQ, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.skip_fc2 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_fc3 = nn.Linear(hidden_dim + 10, skip_dim)  # output layer taking context and state into account
        self._non_linearity = non_linearity

    def forward(self, x, a=None):
        # Process the input state
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))

        # Process behaviour-action as context
        x_ = self._non_linearity(self.skip_fc2(a))
        return self.skip_fc3(torch.cat([x, x_], -1))  # Concatenate both to produce the final output


class WeightSharingTQ(nn.Module):
    """
    Q-function with shared state representation but two independent output streams (action, skip)
    """

    def __init__(self, state_dim, action_dim, skip_dim, non_linearity=F.relu, hidden_dim=50):
        super(WeightSharingTQ, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.skip_fc2 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_fc3 = nn.Linear(hidden_dim, action_dim)
        self.skip_fc3 = nn.Linear(hidden_dim + 10, skip_dim)
        self._non_linearity = non_linearity

    def forward(self, x, a=None):
        # Process input state with shared layers
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))

        if a is not None:  # Only compute Skip Output if the behaviour action is given as context
            x_ = self._non_linearity(self.skip_fc2(a))
            return self.skip_fc3(torch.cat([x, x_], -1))

        # Only compute Behaviour output
        return self.action_fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)


class SkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects "concatenated states" which already contain the behaviour-action for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length of the transition
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths)


class NoneConcatSkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects states in which the behaviour-action is not siply concatenated for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    Additionally stores the behaviour_action which is the context for this skip-transition.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths", "behaviour_action"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[],
                                behaviour_action=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length, behaviour):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length
        self._data.behaviour_action.append(behaviour)  # Behaviour action to condition skip on
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)
            self._data.behaviour_action.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        batch_behavoiurs = np.array([self._data.behaviour_action[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths), tt(batch_behavoiurs)


class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env, vision: bool = False):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # For featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)
        else:  # For image states, i.e. Atari
            self._q = NatureDQN(state_dim, action_dim).to(device)
            self._q_target = NatureDQN(state_dim, action_dim).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :return:
        """
        total_steps = 0
        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            for t in range(max_env_time_steps):
                a = self.get_action(s, epsilon)
                ns, r, d, _ = self._env.step(a)
                total_steps += 1

                ########### Begin Evaluation
                if (total_steps % eval_every_n_steps) == 0:
                    eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                        avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                        avg_rew_per_eval_ep=float(np.mean(eval_r)),
                        std_rew_per_eval_ep=float(np.std(eval_r)),
                        eval_eps=eval_eps
                    )

                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(64)

                ########### Begin double Q-learning update
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()

                soft_update(self._q_target, self._q, 0.01)
                ########### End double Q-learning update

                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # Final evaluation
        if (total_steps % eval_every_n_steps) != 0:
            eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )

            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

    def eval(self, episodes: int, max_env_time_steps: int):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, 0)
                    ed += 1

                    ns, r, d, _ = self._eval_env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))


class DAR:
    """
    Simple Dynamic Action Repetition Agent based on double DQN
    """

    def __init__(self, state_dim: int, action_dim: int,
                 num_output_duplication: int, skip_map: dict,
                 gamma: float, env: gym.Env, eval_env: gym.Env):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param num_output_duplication: integer that determines how often to duplicate output heads (original is 2)
        :param skip_map: determines the skip value associated with each output head
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        """

        # TODO make DAR work for image states to use with ATARI
        self._q = Q(state_dim, action_dim * num_output_duplication).to(device)
        self._q_target = Q(state_dim, action_dim * num_output_duplication).to(device)

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_map = skip_map
        self._dup_vals = num_output_duplication
        self._env = env
        self._eval_env = eval_env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        total_steps = 0
        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for t in range(max_env_time_steps):
                a = self.get_action(s, epsilon)

                # convert action id int corresponding behaviour action and skip value
                act = a // self._dup_vals  # behaviour
                rep = a // self._env.action_space.n  # skip id
                skip = self._skip_map[rep]  # skip id to corresponding skip value

                for _ in range(skip + 1):  # repeat chosen behaviour action for "skip" steps
                    ns, r, d, _ = self._env.step(act)
                    total_steps += 1
                    es += 1

                    ########### Begin Evaluation
                    if (total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                            avg_rew_per_eval_ep=float(np.mean(eval_r)),
                            std_rew_per_eval_ep=float(np.std(eval_r)),
                            eval_eps=eval_eps
                        )

                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                    ########### End Evaluation

                    ### Q-update based double Q learning
                    self._replay_buffer.add_transition(s, a, ns, r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(64)

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                        break

                    s = ns
                if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # Final evaluation
        if (total_steps % eval_every_n_steps) != 0:
            eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )

            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

    def eval(self, episodes: int, max_env_time_steps: int):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    # print(self._q(tt(s)))
                    a = self.get_action(s, 0)
                    act = a // self._dup_vals
                    rep = a // self._eval_env.action_space.n
                    skip = self._skip_map[rep]

                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(act)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))


class TQN:
    """
    TempoRL DQN agent that maintains separate skip and behaviour Q-networks.
    Only works for featurized data as it expects the possibility to concatenate the behaviour action to the state vector
    as input for the skip-Q
    """

    def __init__(self, state_dim: int, action_dim: int, skip_dim: int, gamma: float, env: gym.Env, eval_env: gym.Env,
                 vision: bool = False):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)

            self._skip_q = Q(state_dim + 1, skip_dim).to(device)
        else:
            raise NotImplementedError

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = SkipReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x
        """
        u = np.argmax(self._skip_q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

    def eval(self, episodes: int, max_env_time_steps: int):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    a = self.get_action(s, 0)
                    skip_state = np.hstack([s, [a]])
                    skip = self.get_skip(skip_state, 0)
                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        total_steps = 0
        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for _ in count():
                a = self.get_action(s, epsilon)
                skip_state = np.hstack([s, [a]])  # concatenate action to the state
                skip = self.get_skip(skip_state, epsilon)

                d = False
                skip_states, skip_rewards = [], []
                for _ in range(skip + 1):  # play the same action a "skip" times
                    ns, r, d, _ = self._env.step(a)
                    total_steps += 1
                    es += 1
                    skip_states.append(np.hstack([s, [a]]))  # keep track of all states that are visited inbetween
                    skip_rewards.append(r)

                    #### Evaluation
                    if (total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                            avg_rew_per_eval_ep=float(np.mean(eval_r)),
                            std_rew_per_eval_ep=float(np.std(eval_r)),
                            eval_eps=eval_eps
                        )

                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                    ### Evaluation

                    # Update the skip buffer with all observed transitions
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip:]):
                            skip_reward += np.power(self._gamma, exp) * r  # make sure to properly discount rewards

                        self._skip_replay_buffer.add_transition(start_state, skip_id, ns, skip_reward, d, skip_id + 1)
                    skip_id += 1

                    # Skip Q update based on double DQN where the target is the behaviour network
                    batch_states, batch_actions, batch_next_states, batch_rewards, \
                    batch_terminal_flags, batch_lengths = self._skip_replay_buffer.random_next_batch(64)

                    target = batch_rewards + (1 - batch_terminal_flags) * np.power(self._gamma, batch_lengths) * \
                             self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._skip_q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                    loss = self._skip_loss_function(current_prediction, target.detach())

                    self._skip_q_optimizer.zero_grad()
                    loss.backward()
                    self._skip_q_optimizer.step()

                    # Action Q update based on double DQN with standard target network
                    self._replay_buffer.add_transition(s, a, ns, r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(64)

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                        break

                    s = ns
                if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # Final evaluatoin
        if (total_steps % eval_every_n_steps) != 0:
            eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )

            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

    def save_model(self, path):
        # self.save_always = {self._q, self._skip_q}
        # self.save_optional = {self._q_optimizer, self._skip_q_optimizer,
        #                       self._replay_buffer, self._skip_replay_buffer}
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))
        torch.save(self._skip_q.state_dict(), os.path.join(path, 'TQ'))


class TDQN:
    """
    TempoRL DQN agent capable of handling more complex state inputs through use of contextualized behaviour actions.
    """

    def __init__(self, state_dim, action_dim, skip_dim, gamma, env, eval_env, vision=False, shared=True):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the action output
        :param skip_dim: dimenionality of the skip output
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param vision: boolean flag to indicate if the input state is an image or not
        :param shared: boolean flag to indicate if a weight sharing input representation is used or not.
        """
        if not vision:
            if shared:
                self._q = WeightSharingTQ(state_dim, action_dim, skip_dim).to(device)
                self._q_target = WeightSharingTQ(state_dim, action_dim, skip_dim).to(device)
            else:
                self._q = Q(state_dim, action_dim).to(device)
                self._q_target = Q(state_dim, action_dim).to(device)
        else:
            self._q = NatureWeightsharingTQN(state_dim, action_dim, skip_dim).to(device)
            self._q_target = NatureWeightsharingTQN(state_dim, action_dim, skip_dim).to(device)

        if shared:
            self._skip_q = self._q
        else:
            self._skip_q = TQ(state_dim, skip_dim).to(device)
        print('Using {} as Q'.format(str(self._q)))
        print('Using {} as skip-Q\n{}'.format(str(self._skip_q), '#' * 80))

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._skip_loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._skip_dim = skip_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def get_skip(self, x: np.ndarray, a: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get the skip epsilon-greedy based on observation x conditioned on behaviour action a
        """
        u = np.argmax(self._skip_q(tt(x), tt(a)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

    def eval(self, episodes: int, max_env_time_steps: int):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = self._eval_env.reset()
                for _ in count():
                    # print(self._q(tt(s)))
                    # a = self.get_action(np.array([s]), 0)
                    a = self.get_action(s, 0)
                    # print(self._skip_q(tt(s), tt(np.array([a]))))
                    # print('')
                    # print('')
                    # print('')
                    # skip = self.get_skip(np.array([s]), np.array([[a]]), 0)
                    skip = self.get_skip(s, np.array([a]), 0)
                    ed += 1

                    d = False
                    for _ in range(skip + 1):
                        ns, r, d, _ = self._eval_env.step(a)
                        er += r
                        es += 1
                        if es >= max_env_time_steps or d:
                            break
                        s = ns
                    if es >= max_env_time_steps or d:
                        break
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        """
        total_steps = 0
        start_time = time.time()
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            es = 0
            for _ in count():
                a = self.get_action(s, epsilon)
                skip = self.get_skip(s, np.array([a]), epsilon)  # get skip with the selected action as context

                d = False
                skip_states, skip_rewards = [], []
                for _ in range(skip + 1):  # repeat the selected action for "skip" times
                    ns, r, d, _ = self._env.step(a)
                    total_steps += 1
                    es += 1
                    skip_states.append(s)  # keep track of all observed skips
                    skip_rewards.append(r)

                    #### Begin Evaluation
                    if (total_steps % eval_every_n_steps) == 0:
                        eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                            avg_rew_per_eval_ep=float(np.mean(eval_r)),
                            std_rew_per_eval_ep=float(np.std(eval_r)),
                            eval_eps=eval_eps
                        )

                        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                    #### End Evaluation

                    # Update the skip replay buffer with all observed skips.
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip:]):  # make sure to properly discount
                            skip_reward += np.power(self._gamma, exp) * r

                        self._skip_replay_buffer.add_transition(start_state, skip_id, ns, skip_reward, d, skip_id + 1,
                                                                np.array([a]))  # also keep track of the behavior action
                    skip_id += 1

                    # Skip Q update based on double DQN where target is behavior Q
                    batch_states, batch_actions, batch_next_states, batch_rewards,\
                        batch_terminal_flags, batch_lengths, batch_behaviours = \
                        self._skip_replay_buffer.random_next_batch(64)

                    target = batch_rewards + (1 - batch_terminal_flags) * np.power(self._gamma, batch_lengths) * \
                             self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._skip_q(batch_states, batch_behaviours)[
                        torch.arange(64).long(), batch_actions.long()]

                    loss = self._skip_loss_function(current_prediction, target.detach())

                    self._skip_q_optimizer.zero_grad()
                    loss.backward()
                    self._skip_q_optimizer.step()

                    # Action Q update based on double DQN with normal target
                    self._replay_buffer.add_transition(s, a, ns, r, d)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(64)

                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                        break

                    s = ns
                if es >= max_env_time_steps or d or total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # final evaluation
        if (total_steps % eval_every_n_steps) != 0:
            eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )

            with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')

    def save_model(self, path):
        # self.save_always = {self._q, self._skip_q}
        # self.save_optional = {self._q_optimizer, self._skip_q_optimizer,
        #                       self._replay_buffer, self._skip_replay_buffer}
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))
        torch.save(self._skip_q.state_dict(), os.path.join(path, 'TQ'))


if __name__ == "__main__":
    import argparse

    outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y%m%dT%H%M%S.%f',
                          'seed': '{:d}', 'params': '{:d}_{:d}_{:d}',
                          'paramsseed': '{:d}_{:d}_{:d}_{:d}'}
    parser = argparse.ArgumentParser('TempoRL')
    parser.add_argument('--episodes', '-e',
                        default=100,
                        type=int,
                        help='Number of training episodes.')
    parser.add_argument('--training-steps', '-t',
                        default=1_000_000,
                        type=int,
                        help='Number of training episodes.')

    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='paramsseed',
                        type=str,
                        choices=list(outdir_suffix_dict.keys()),
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=12345,
                        type=int,
                        help='Seed')
    parser.add_argument('--eval-after-n-steps',
                        default=10 ** 3,
                        type=int,
                        help='After how many steps to evaluate')
    parser.add_argument('--eval-n-episodes',
                        default=1,
                        type=int,
                        help='How many episodes to evaluate')

    # DQN -> normal double DQN agent
    # DAR -> Dynamic action repetition agent based on normal DDQN with repeated output heads for different skip values
    # tqn -> TempoRL DDQN only for featurized states where the behaviour action can be concatenated with the state
    # tdqn -> TempoRL DDQN with shared state representation for behaviour and skip Qs
    # t-dqn -> TempoRL DDQN without shared state representation for behaviour and skip Qs
    parser.add_argument('--agent',
                        choices=['dqn', 'tqn', 'dar', 'tdqn', 't-dqn'],
                        type=str.lower,
                        help='Which agent to train',
                        default='tqn')
    parser.add_argument('--skip-net-max-skips',
                        type=int,
                        default=10,
                        help='Maximum skip-size')
    parser.add_argument('--env-max-steps',
                        default=200,
                        type=int,
                        help='Maximal steps in environment before termination.',
                        dest='env_ms')
    parser.add_argument('--dar-base', default=None,
                        type=int,
                        help='DAR base')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--dar-A', default=None, type=int)
    parser.add_argument('--dar-B', default=None, type=int)
    parser.add_argument('--env', choices=['mountain', 'moon', 'pong'], default='mountain')

    # setup output dir
    args = parser.parse_args()
    outdir_suffix_dict['seed'] = outdir_suffix_dict['seed'].format(args.seed)
    epis = args.episodes if args.episodes else -1
    outdir_suffix_dict['params'] = outdir_suffix_dict['params'].format(
        epis, args.skip_net_max_skips, args.env_ms)
    outdir_suffix_dict['paramsseed'] = outdir_suffix_dict['paramsseed'].format(
        epis, args.skip_net_max_skips, args.env_ms, args.seed)

    out_dir = experiments.prepare_output_dir(args, user_specified_dir=args.out_dir,
                                             time_format=outdir_suffix_dict[args.out_dir_suffix])

    if args.env == 'pong':  # attempt at getting it to work with Pong
        from utils.env_wrappers import make_env  # TODO figure out if this wrapping is correct

        # Setup Envs
        env = make_env("PongNoFrameskip-v4")
        eval_env = make_env("PongNoFrameskip-v4")

        # Setup Agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        if args.agent == 'dqn':
            agent = DQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env, vision=True)
        elif args.agent == 'tqn':
            agent = TQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, eval_env=eval_env,
                        vision=True)
        elif args.agent == 'tdqn':
            agent = TDQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, eval_env=eval_env,
                         vision=True)
        elif args.agent == 'dar':
            if args.dar_A is not None and args.dar_B is not None:
                skip_map = {0: args.dar_A, 1: args.dar_B}
            elif args.dar_base:
                skip_map = {a: args.dar_base ** a for a in range(args.skip_net_max_skips)}
            else:
                skip_map = {a: a for a in range(args.skip_net_max_skips)}
            agent = DAR(state_dim, action_dim, args.skip_net_max_skips, skip_map, gamma=0.99, env=env,
                        eval_env=eval_env)
    else:  # Simple featurized environments
        # Setup Env
        if args.env == 'mountain':
            if args.sparse:
                from gym.envs.classic_control import MountainCarEnv
            env = MountainCarEnv()
            eval_env = MountainCarEnv()
        elif args.env == 'moon':
            env = gym.make('LunarLander-v2')
            eval_env = gym.make('LunarLander-v2')

        # Setup agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        if args.agent == 'dqn':
            agent = DQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env)
        elif args.agent == 'tqn':
            agent = TQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, eval_env=eval_env)
        elif args.agent == 'tdqn':
            agent = TDQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env, eval_env=eval_env)
        elif args.agent == 't-dqn':
            agent = TDQN(state_dim, action_dim, args.skip_net_max_skips, gamma=0.99, env=env,
                         eval_env=eval_env, shared=False)
        elif args.agent == 'dar':
            if args.dar_A is not None and args.dar_B is not None:
                skip_map = {0: args.dar_A, 1: args.dar_B}
            elif args.dar_base:
                skip_map = {a: args.dar_base ** a for a in range(args.skip_net_max_skips)}
            else:
                skip_map = {a: a for a in range(args.skip_net_max_skips)}
            agent = DAR(state_dim, action_dim, args.skip_net_max_skips, skip_map, gamma=0.99, env=env,
                        eval_env=eval_env)
        else:
            raise NotImplementedError

    episodes = args.episodes
    max_env_time_steps = args.env_ms
    epsilon = 0.2

    agent.train(episodes, max_env_time_steps, epsilon, args.eval_n_episodes, args.eval_after_n_steps,
                max_train_time_steps=args.training_steps)
    os.mkdir(os.path.join(out_dir, 'final'))
    agent.save_model(os.path.join(out_dir, 'final'))
