import yaml
import os
import glob
import pickle
import numpy as np

import json
import pandas as pd


def load_config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config'), 'r') as ymlf:
        cfg = yaml.load(ymlf, Loader=yaml.FullLoader)
    return cfg


def load_data(experiment_dir="experiments_01_28", methods=['sq', 'q'], exp_version='-1.0-linear',
              episodes=50_000, max_skip=6, max_steps=100, local=True, debug=False):
    cfg = load_config()
    method_test_rewards = {}
    method_test_lengths = {}
    method_steps_per_episodes = {}
    if not local:
        print('Loading from')
        print(os.path.join(cfg['data']['local' if local else 'remote'], experiment_dir))
    for method in methods:
        print(method)
        files = glob.glob(
            os.path.join(cfg['data']['local' if local else 'remote'], experiment_dir,
                         '{:s}-experiments{:s}'.format(method, exp_version),
                         '{:d}_{:d}_{:d}_*', 'test_data.pkl'
                         ).format(
                episodes,
                max_skip,
                max_steps
            ))
        test_rewards, test_lens, steps_per_eps = [], [], []
        for file in files:
            if debug:
                print('Loading', file)
            with open(file, 'rb') as fh:
                data = pickle.load(fh)
                test_rewards.append(data[0])
                test_lens.append(data[1])
            try:
                with open(file.replace('test_data', 'steps_per_episode'), 'rb') as fh:
                    data = pickle.load(fh)
                    steps_per_eps.append(data[1])
            except FileNotFoundError:
                print('No steps data found')

        method_test_rewards[method] = np.array(test_rewards)
        method_test_lengths[method] = np.array(test_lens)
        method_steps_per_episodes[method] = np.array(steps_per_eps)
    return method_test_rewards, method_test_lengths, method_steps_per_episodes


def load_dqn_data(experiment_dir, method, max_steps=None, succ_threashold=None, debug=False):
    cfg = load_config()
    print(os.path.abspath(os.path.join(method, experiment_dir, 'eval_scores.json')))
    files = glob.glob(
        os.path.abspath(os.path.join(method, experiment_dir, 'eval_scores.json')),
        recursive=True
    )
    frames = []
    max_len = 0
    succ_count = 0
    for file in sorted(files):
        if debug:
            print('Loading', file)
        data = []
        with open(file, 'r') as fh:
            for line in fh:
                loaded = json.loads(line)
                data.append(loaded)
                if max_steps and loaded['training_steps'] >= max_steps:
                    break
            frame = pd.DataFrame(data)
            max_len = max(max_len, frame.shape[0])
        if succ_threashold:
            if loaded['avg_rew_per_eval_ep'] > succ_threashold:
                succ_count += 1
        frames.append(frame)
    rews, lens, decs, training_steps, training_eps = [], [], [], [], []
    for frame in frames:
        for (list_, array) in [(rews, frame.avg_rew_per_eval_ep), (lens, frame.avg_num_steps_per_eval_ep),
                               (decs, frame.avg_num_decs_per_eval_ep), (training_steps, frame.training_steps),
                               (training_eps, frame.training_eps)]:
            data = np.full((max_len,), np.nan)
            data[:len(array)] = array
            list_.append(data)
    mean_r, std_r = np.nanmean(rews, axis=0), np.nanstd(rews, axis=0)
    mean_l, std_l = np.nanmean(lens, axis=0), np.nanstd(lens, axis=0)
    mean_d, std_d = np.nanmean(decs, axis=0), np.nanstd(decs, axis=0)
    mean_ts, std_ts = np.nanmean(training_steps, axis=0), np.nanstd(training_steps, axis=0)
    mean_te, std_te = np.nanmean(training_eps, axis=0), np.nanstd(training_eps, axis=0)
    if succ_threashold:
        try:
            print('\t {}/{} ({}\%) runs exceeded a final performance  of {} after {} training steps'.format(
                succ_count, len(frames), succ_count / len(frames) * 100, succ_threashold, max_steps
            ))
        except ZeroDivisionError:
            pass
    if debug:
        print('#' * 80, '\n')

    return (mean_r, std_r), (mean_l, std_l), (mean_d, std_d), (mean_ts, std_ts), (mean_te, std_te)
