import os
import pickle
from collections import defaultdict

import numpy as np
from utils import experiments

from grid_envs import GridCore


def make_epsilon_greedy_policy(Q: defaultdict, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.

    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
            Q[observation] == Q[observation].max()
        ))
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule

    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError


def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float):
    """ Simple TD update rule """
    # TD update
    best_next_action = np.random.choice(np.flatnonzero(q[next_state] == q[next_state].max()))  # greedy best next
    td_target = reward + gamma * q[next_state][best_next_action]
    td_delta = td_target - q[state][action]
    return q[state][action] + alpha * td_delta


def q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10,
        render_eval: bool = True):
    """
    Vanilla tabular Q-learning algorithm
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    for i_episode in range(num_episodes + 1):
        # print('#' * 100)
        epsilon = epsilon_schedule[min(i_episode, num_episodes - 1)]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha)

            if policy_done:
                break
            policy_state = s_
        rewards.append(cummulative_reward)
        lens.append(episode_length)
        train_steps_list.append(environment.total_steps)

        # evaluation with greedy policy
        test_steps = 0
        if i_episode % eval_every == 0:
            policy_state = environment.reset()
            episode_length, cummulative_reward = 0, 0
            if render_eval:
                environment.render()
            while True:  # roll out episode
                policy_action = np.random.choice(np.flatnonzero(Q[policy_state] == Q[policy_state].max()))
                environment.total_steps -= 1  # don't count evaluation steps
                s_, policy_reward, policy_done, _ = environment.step(policy_action)
                test_steps += 1
                if render_eval:
                    environment.render()
                s_ = s_
                cummulative_reward += policy_reward
                episode_length += 1
                if policy_done:
                    break
                policy_state = s_
            test_rewards.append(cummulative_reward)
            test_lens.append(episode_length)
            test_steps_list.append(test_steps)
            print('Done %4d/%4d episodes' % (i_episode, num_episodes))
    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list), Q


class SkipTransition:
    """
    Simple helper class to keep track of all transitions observed when skipping through an MDP
    """

    def __init__(self, skips, df):
        self.state_mat = np.full((skips, skips), -1, dtype=int)  # might need to change type for other envs
        self.reward_mat = np.full((skips, skips), np.nan, dtype=float)
        self.idx = 0
        self.df = df

    def add(self, reward, next_state):
        """
        Add reward and next_state to triangular matrix
        :param reward: received reward
        :param next_state: state reached
        """
        self.idx += 1
        for i in range(self.idx):
            self.state_mat[self.idx - i - 1, i] = next_state
            # Automatically discount rewards when adding to corresponding skip
            self.reward_mat[self.idx - i - 1, i] = reward * self.df ** i + np.nansum(self.reward_mat[self.idx - i - 1])


def temporl_q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        decay_stops: int = None,
        eval_every: int = 10,
        render_eval: bool = True,
        max_skip: int = 7):
    """
    Implementation of tabular TempoRL
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param decay_stops: Episode after which to stop epsilon decay
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :param max_skip: Maximum skip size to use.
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    temporal_actions = max_skip
    action_Q = defaultdict(lambda: np.zeros(environment.action_space.n))
    temporal_Q = defaultdict(lambda: np.zeros(temporal_actions))
    if not decay_stops:
        decay_stops = num_episodes

    epsilon_schedule_action = get_decay_schedule(epsilon, decay_starts, decay_stops, epsilon_decay)
    epsilon_schedule_temporal = get_decay_schedule(epsilon, decay_starts, decay_stops, epsilon_decay)
    rewards = []
    lens = []
    test_rewards = []
    test_lens = []
    train_steps_list = []
    test_steps_list = []
    for i_episode in range(num_episodes + 1):

        # setup exploration policy for this episode
        epsilon_action = epsilon_schedule_action[min(i_episode, num_episodes - 1)]
        epsilon_temporal = epsilon_schedule_temporal[min(i_episode, num_episodes - 1)]
        action_policy = make_epsilon_greedy_policy(action_Q, epsilon_action, environment.action_space.n)
        temporal_policy = make_epsilon_greedy_policy(temporal_Q, epsilon_temporal, temporal_actions)

        episode_r = 0
        state = environment.reset()  # type: list
        action_pol_len = 0
        while True:  # roll out episode
            action = np.random.choice(list(range(environment.action_space.n)), p=action_policy(state))
            temporal_state = (state, action)
            action_pol_len += 1
            temporal_action = np.random.choice(list(range(temporal_actions)), p=temporal_policy(temporal_state))

            s_ = None
            done = False
            tmp_state = state
            skip_transition = SkipTransition(temporal_action + 1, discount_factor)
            reward = 0
            for tmp_temporal_action in range(temporal_action + 1):
                if not done:
                    # only perform action if we are not done. If we are not done "skipping" though we have to
                    # still add reward and same state to the skip_transition.
                    s_, reward, done, _ = environment.step(action)
                skip_transition.add(reward, tmp_state)

                # 1-step update of action Q (like in vanilla Q)
                action_Q[tmp_state][action] = td_update(action_Q, tmp_state, action,
                                                        reward, s_, discount_factor, alpha)

                count = 0
                # For all sofar observed transitions compute all forward skip updates
                for skip_num in range(skip_transition.idx):
                    skip = skip_transition.state_mat[skip_num]
                    rew = skip_transition.reward_mat[skip_num]
                    skip_start_state = (skip[0], action)

                    # Temporal TD update
                    best_next_action = np.random.choice(
                        np.flatnonzero(action_Q[s_] == action_Q[s_].max()))  # greedy best next
                    td_target = rew[skip_transition.idx - 1 - count] + (
                            discount_factor ** (skip_transition.idx - 1)) * action_Q[s_][best_next_action]
                    td_delta = td_target - temporal_Q[skip_start_state][skip_transition.idx - count - 1]
                    temporal_Q[skip_start_state][skip_transition.idx - count - 1] += alpha * td_delta
                    count += 1

                tmp_state = s_
            state = s_
            if done:
                break
        rewards.append(episode_r)
        lens.append(action_pol_len)
        train_steps_list.append(environment.total_steps)

        # ---------------------------------------------- EVALUATION -------------------------------------------------
        # ---------------------------------------------- EVALUATION -------------------------------------------------
        test_steps = 0
        if i_episode % eval_every == 0:
            episode_r = 0
            state = environment.reset()  # type: list
            if render_eval:
                environment.render(in_control=True)
            action_pol_len = 0
            while True:  # roll out episode
                action = np.random.choice(np.flatnonzero(action_Q[state] == action_Q[state].max()))
                temporal_state = (state, action)
                action_pol_len += 1

                # Examples of different action selection schemes when greedily following a policy
                # temporal_action = np.random.choice(
                #     np.flatnonzero(temporal_Q[temporal_state] == temporal_Q[temporal_state].max()))
                temporal_action = np.max(  # if there are ties use the larger action
                    np.flatnonzero(temporal_Q[temporal_state] == temporal_Q[temporal_state].max()))
                # temporal_action = np.min(  # if there are ties use the smaller action
                #     np.flatnonzero(temporal_Q[temporal_state] == temporal_Q[temporal_state].max()))

                for i in range(temporal_action + 1):
                    environment.total_steps -= 1  # don't count evaluation steps
                    s_, reward, done, _ = environment.step(action)
                    test_steps += 1
                    if render_eval:
                        environment.render(in_control=False)
                    episode_r += reward
                    if done:
                        break
                if render_eval:
                    environment.render(in_control=True)
                state = s_
                if done:
                    break
            test_rewards.append(episode_r)
            test_lens.append(action_pol_len)
            test_steps_list.append(test_steps)
            print('Done %4d/%4d episodes' % (i_episode, num_episodes))
    return (rewards, lens), (test_rewards, test_lens), (train_steps_list, test_steps_list), (action_Q, temporal_Q)


if __name__ == '__main__':
    import argparse

    outdir_suffix_dict = {'none': '', 'empty': '', 'time': '%Y%m%dT%H%M%S.%f',
                          'seed': '{:d}', 'params': '{:d}_{:d}_{:d}',
                          'paramsseed': '{:d}_{:d}_{:d}_{:d}'}
    parser = argparse.ArgumentParser('Skip-MDP Tabular-Q')
    parser.add_argument('--episodes', '-e',
                        default=10_000,
                        type=int,
                        help='Number of training episodes')
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
    parser.add_argument('--env-max-steps',
                        default=100,
                        type=int,
                        help='Maximal steps in environment before termination.',
                        dest='env_ms')
    parser.add_argument('--agent-eps-decay',
                        default='linear',
                        choices={'linear', 'log', 'const'},
                        help='Epsilon decay schedule',
                        dest='agent_eps_d')
    parser.add_argument('--agent-eps',
                        default=1.0,
                        type=float,
                        help='Epsilon value. Used as start value when decay linear or log. Otherwise constant value.',
                        dest='agent_eps')
    parser.add_argument('--agent',
                        default='sq',
                        choices={'sq', 'q'},
                        type=str.lower,
                        help='Agent type to train')
    parser.add_argument('--env',
                        default='lava',
                        choices={'lava', 'lava2',
                                 'lava_perc', 'lava2_perc',
                                 'lava_ng', 'lava2_ng',
                                 'lava3', 'lava3_perc', 'lava3_ng'},
                        type=str.lower,
                        help='Enironment to use')
    parser.add_argument('--eval-eps',
                        default=100,
                        type=int,
                        help='After how many episodes to evaluate')
    parser.add_argument('--stochasticity',
                        default=0,
                        type=float,
                        help='probability of the selected action failing and instead executing any of the remaining 3')
    parser.add_argument('--no-render',
                        action='store_true',
                        help='Deactivate rendering of environment evaluation')
    parser.add_argument('--max-skips',
                        type=int,
                        default=7,
                        help='Max skip size for tempoRL')

    # setup output dir
    args = parser.parse_args()
    outdir_suffix_dict['seed'] = outdir_suffix_dict['seed'].format(args.seed)
    outdir_suffix_dict['params'] = outdir_suffix_dict['params'].format(
        args.episodes, args.max_skips, args.env_ms)
    outdir_suffix_dict['paramsseed'] = outdir_suffix_dict['paramsseed'].format(
        args.episodes, args.max_skips, args.env_ms, args.seed)

    if not args.no_render:
        # Clear screen in ANSI terminal
        print('\033c')
        print('\x1bc')

    out_dir = experiments.prepare_output_dir(args, user_specified_dir=args.out_dir,
                                             time_format=outdir_suffix_dict[args.out_dir_suffix])

    np.random.seed(args.seed)  # seed nump
    d = None

    if args.env.startswith('lava'):
        import gym
        from grid_envs import Bridge6x10Env, Pit6x10Env, ZigZag6x10, ZigZag6x10H

        perc = args.env.endswith('perc')
        ng = args.env.endswith('ng')
        if args.env.startswith('lava2'):
            d = Bridge6x10Env(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng,
                              act_fail_prob=args.stochasticity, numpy_state=False)
        elif args.env.startswith('lava3'):
            d = ZigZag6x10(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9),
                           act_fail_prob=args.stochasticity, numpy_state=False)
        elif args.env.startswith('lava4'):
            d = ZigZag6x10H(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng, goal=(5, 9),
                            act_fail_prob=args.stochasticity, numpy_state=False)
        else:
            d = Pit6x10Env(max_steps=args.env_ms, percentage_reward=perc, no_goal_rew=ng,
                           act_fail_prob=args.stochasticity, numpy_state=False)

    # setup agent
    if args.agent == 'sq':
        train_data, test_data, num_steps, (action_Q, t_Q) = temporl_q_learning(d, args.episodes,
                                                                               epsilon_decay=args.agent_eps_d,
                                                                               epsilon=args.agent_eps,
                                                                               discount_factor=.99, alpha=.5,
                                                                               eval_every=args.eval_eps,
                                                                               render_eval=not args.no_render,
                                                                               max_skip=args.max_skips)
    elif args.agent == 'q':
        train_data, test_data, num_steps, Q = q_learning(d, args.episodes,
                                                         epsilon_decay=args.agent_eps_d,
                                                         epsilon=args.agent_eps,
                                                         discount_factor=.99,
                                                         alpha=.5, eval_every=args.eval_eps,
                                                         render_eval=not args.no_render)
    else:
        raise NotImplemented

    # TODO save resulting Q-function for easy reuse
    with open(os.path.join(out_dir, 'train_data.pkl'), 'wb') as outfh:
        pickle.dump(train_data, outfh)
    with open(os.path.join(out_dir, 'test_data.pkl'), 'wb') as outfh:
        pickle.dump(test_data, outfh)
    with open(os.path.join(out_dir, 'steps_per_episode.pkl'), 'wb') as outfh:
        pickle.dump(num_steps, outfh)

    if args.agent == 'q':
        with open(os.path.join(out_dir, 'Q.pkl'), 'wb') as outfh:
            pickle.dump(dict(Q), outfh)
    elif args.agent == 'sq':
        with open(os.path.join(out_dir, 'Q.pkl'), 'wb') as outfh:
            pickle.dump(dict(action_Q), outfh)
        with open(os.path.join(out_dir, 'J.pkl'), 'wb') as outfh:
            pickle.dump(dict(t_Q), outfh)
