"""
Based on code from Scott Fujimoto https://github.com/sfujim/TD3
We adapted the DDPG code he provides to allow for FiGAR and TempoRL variants
This code is originally under the MIT license https://github.com/sfujim/TD3/blob/master/LICENSE
"""

import argparse

import gym
import numpy as np
import torch

from DDPG import utils
from DDPG.FiGAR import DDPG as FiGARDDPG
from DDPG.TempoRL import DDPG as TempoRLDDPG
from DDPG.vanilla import DDPG
from utils import experiments


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, FiGAR=False, TempoRL=False):
    eval_env = gym.make(env_name)
    special = 'PendulumDecs-v0' == env_name
    if special:
        eval_env = utils.Render(eval_env, episode_modulo=10)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    avg_steps = 0.
    avg_decs = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        repetition = 1
        while not done:
            if FiGAR:
                action, repetition, rps = policy.select_action(np.array(state))
                repetition = repetition[0] + 1

            elif TempoRL:
                action = policy.select_action(np.array(state))
                repetition = np.argmax(policy.select_skip(np.array(state), action)) + 1

            else:
                action = policy.select_action(np.array(state))

            if special:
                eval_env.set_decision_point(True)
            avg_decs += 1

            for _ in range(repetition):
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                avg_steps += 1
                if done:
                    break
        eval_env.close()

    avg_reward /= eval_episodes
    avg_decs /= eval_episodes
    avg_steps /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_decs, avg_steps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument("--policy", default="TempoRLDDPG")  # Policy name (DDPG, FiGARDDPG or our TempoRLDDPG)
    parser.add_argument("--env", default="Pendulum-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e4, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--max-skip", "--max-rep", default=20, type=int,
                        dest='max_rep')  # Maximum Skip length to use with FiGAR or TempoRL
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    outdir_suffix_dict = dict()
    outdir_suffix_dict['seed'] = '{:d}'.format(args.seed)
    out_dir = experiments.prepare_output_dir(args, user_specified_dir=args.out_dir,
                                             time_format=outdir_suffix_dict['seed'])

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    max_rep = args.max_rep

    # Initialize policy
    if args.policy == "DDPG":
        policy = DDPG(**kwargs)
    elif args.policy.startswith('FiGAR'):
        kwargs['repetition_dim'] = max_rep
        policy = FiGARDDPG(**kwargs)
    elif args.policy.startswith('TempoRL'):
        kwargs['skip_dim'] = max_rep
        policy = TempoRLDDPG(**kwargs)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy_file = args.load_model
        policy.load(f"{out_dir}/{policy_file}")

    skip_replay_buffer = None
    if 'FiGAR' in args.policy:
        replay_buffer = utils.FiGARReplayBuffer(state_dim, action_dim, rep_dim=max_rep)
    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    if 'TempoRL' in args.policy:
        skip_replay_buffer = utils.FiGARReplayBuffer(state_dim, action_dim, rep_dim=1)

    # Evaluate untrained policy
    evaluations = [[0, *eval_policy(policy, args.env, args.seed, FiGAR='FiGAR' in args.policy,
                                    TempoRL='TempoRL' in args.policy)]]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    t = 0
    while t < int(args.max_timesteps):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:  # Before learning starts we sample actions uniformly at random
            action = env.action_space.sample()
            if args.policy.startswith('FiGAR'):
                # FiGAR uses a second actor network to learn the repetition value so we have to create
                # initial distirbution over the possible repetition values
                repetition_probs = np.random.random(max_rep)


                def softmax(x):
                    """Compute softmax values for each sets of scores in x."""
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()


                repetition_probs = softmax(repetition_probs)
                repetition = np.argmax(repetition_probs)
            elif args.policy.startswith('TempoRL'):
                # TempoRL uses a simple DQN for which we can simply sample from the possible skip values
                repetition = np.random.randint(max_rep) + 1
            else:
                repetition = 1
        else:
            # Get Action and skip values
            if 'FiGAR' in args.policy:
                # For FiGAR we treat the action policy exploration as in standard DDPG
                action, repetition, repetition_probs = policy.select_action(np.array(state))
                action = (
                        action + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # The Repetition policy however uses epsilon greedy exploration as described in the original paper
                # https://arxiv.org/pdf/1702.06054.pdf
                if np.random.random() < args.expl_noise:
                    repetition = np.random.randint(max_rep) + 1  # + 1 since randint samples from [0, max_rep)
                else:
                    repetition = repetition[0]
            elif 'TempoRL' in args.policy:
                # TempoRL does not interfere with the action policy and its exploration
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

                # the skip policy uses epsilon greedy exploration for learning
                repetition = policy.select_skip(state, action)
                if np.random.random() < args.expl_noise:
                    repetition = np.random.randint(max_rep) + 1  # + 1 sonce randint samples from [0, max_rep)
                else:
                    repetition = np.argmax(repetition) + 1  # + 1 since indices start at 0
            else:
                # Standard DDPG
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                repetition = 1  # Never skip with vanilla DDPG

        # Perform action
        skip_states, skip_rewards = [], []  # only used for TempoRL to build the local conectedness graph
        for curr_skip in range(repetition):
            next_state, reward, done, _ = env.step(action)
            t += 1
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            skip_states.append(state)
            skip_rewards.append(reward)

            # Store data in replay buffer
            if 'FiGAR' in args.policy:
                # To train the second actor with FiGAR, we need to keep track of its output "repetition_probs"
                replay_buffer.add(state, action, repetition_probs, next_state, reward, done_bool)
            else:
                # Vanilla DDPG
                replay_buffer.add(state, action, next_state, reward, done_bool)
                # In addition to the normal replay_buffer
                # TempoRL uses a second replay buffer that is only used for training the skip network
                if 'TempoRL' in args.policy:
                    # Update the skip buffer with all observed transitions in the local connectedness graph
                    skip_id = 0
                    for start_state in skip_states:
                        skip_reward = 0
                        for exp, r in enumerate(skip_rewards[skip_id:]):
                            skip_reward += np.power(policy.discount, exp) * r  # make sure to properly discount rewards
                        skip_replay_buffer.add(start_state, action, curr_skip - skip_id, next_state, skip_reward, done)
                        skip_id += 1

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
                if 'TempoRL' in args.policy:
                    policy.train_skip(skip_replay_buffer, args.batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                break

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                evaluations.append([t, *eval_policy(policy, args.env, args.seed, FiGAR='FiGAR' in args.policy,
                                                    TempoRL='TempoRL' in args.policy)])
                np.save(f"{out_dir}/{file_name}", evaluations)
                if args.save_model:
                    policy.save(f"{out_dir}/{file_name}")
