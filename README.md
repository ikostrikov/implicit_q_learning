# Offline Reinforcement Learning with Implicit Q-Learning 

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.com) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/). 

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How to run the code

### Install dependencies

```bash
pip install -r requirements.txt
```

See [instructions](https://github.com/google/jax#pip-installation-gpu-cuda) for CUDA.

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).