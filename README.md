# TempoRL

This repository contains the code for the ICML'21 paper "[TempoRL: Learning When to Act](https://ml.informatik.uni-freiburg.de/papers/21-ICML-TempoRL.pdf)".

If you use TempoRL in you research or application, please cite us:

```bibtex
@inproceedings{biedenkapp-icml21,
  author    = {André Biedenkapp and Raghu Rajan and Frank Hutter and Marius Lindauer},
  title     = {{T}empo{RL}: Learning When to Act},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning (ICML 2021)},
  year = {2021},
  month     = jul,
}
```

## Appendix
The appendix PDF has been uploaded to this repository and can be accessed [here](TempoRL_Appendix.pdf).

## Setup
This code was developed with python 3.6.12 and torch 1.4.0.
If you have the correct python version you need to install the dependencies via
```bash
pip install -r requirements.txt
```

If you only want to run quick experiments with the tabular agents you can install the minimal requirements in `tabular_requirements.txt` via
```bash
pip install -r tabular_requirements.txt
```

To make use of the provided jupyter notebook you optionally have to install jupyter
```bash
pip install jupyter
```

## How to train tabular agents
To run an agent on any of the below listed environments run
```bash
python run_tabular_experiments.py -e 10000 --agent Agent --env env_name --eval-eps 500
```
replace Agent with `q` for vanilla q-learning and `sq` for our method.

## Envs
Currently 3 simple environments available.
Per default all environments give a reward of 1 when reaching the goal (X).
The agents start in state (S) and can traverse open fields (o).
When falling into "lava" (.) the agent receives a reward of -1.
For no other transition are rewards generated. (When rendering environments the agent is marked with *)
An agent can use at most 100 steps to reach the goal.

Modifications of the below listed environments can run without goal rewards (env_name ends in _ng)
or reduce the goal reward by the number of taken steps (env_name ends in _perc).
* lava (Cliff)
    ```console
    S  o  .  .  .  .  .  .  o  X
    o  o  .  .  .  .  .  .  o  o
    o  o  .  .  .  .  .  .  o  o
    o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o
    ```

* lava2 (Bridge)
    ```console
    S  o  .  .  .  .  .  .  o  X
    o  o  .  .  .  .  .  .  o  o
    o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o
    o  o  .  .  .  .  .  .  o  o
    o  o  .  .  .  .  .  .  o  o
    ```

* lava3 (ZigZag)
    ```console
    S  o  .  .  o  o  o  o  o  o
    o  o  .  .  o  o  o  o  o  o
    o  o  .  .  o  o  .  .  o  o
    o  o  .  .  o  o  .  .  o  o
    o  o  o  o  o  o  .  .  o  o
    o  o  o  o  o  o  .  .  o  X
    ```
  
## How to train deep agents
To train an agent on featurized environments run e.g.
```bash
python run_featurized_experiments.py -e 10000 -t 1000000 --eval-after-n-steps 200 -s 1 --agent tdqn --skip-net-max-skips 10 --out-dir . --sparse
```
replace tdqn (our agent with shared network architecture) with dqn or dar to run the respective baseline agents

To train a DDPG agent run e.g.
```bash
python run_ddpg_experiments.py --policy TempoRLDDPG --env Pendulum-v0 --start_timesteps 1000 --max_timesteps 30000 --eval_freq 250 --max-skip 16 --save_model --out-dir . --seed 1
```
replace TempoRLDDPG with FiGARDDPG or DDPG to run the baseline agents.

To train an agent on atari environments run e.g.
```bash
run_atari_experiments.py --env freeway --env-max-steps 10000 --agent tdqn --out-dir experiments/atari_new/freeway/tdqn_3 --episodes 20000 --training-steps 2500000 --eval-after-n-steps 10000 --seed 12345 --84x84 --eval-n-episodes 3
```
replace tdqn (our agent with shared network architecture) with dqn or dar to run the respective baseline agents.
  
## Experiment data
##### Note: This data is required to run the plotting jupyter notebooks
We provide all learning curve data, final policy network weights as well as commands to generate that data at:
https://figshare.com/s/d9264dc125ca8ba8efd8

(Download this data and move it into the experiments folder)
