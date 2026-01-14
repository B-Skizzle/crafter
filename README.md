**Status:** Stable release

[![PyPI](https://img.shields.io/pypi/v/crafter.svg)](https://pypi.python.org/pypi/crafter/#history)

# Crafter

Open world survival game for evaluating a wide range of agent abilities within
a single environment.

![Crafter Terrain](https://github.com/danijar/crafter/raw/main/media/terrain.png)

## Overview

Crafter features randomly generated 2D worlds where the player needs to forage
for food and grass, find shelter to sleep, defend against monsters, collect
materials, and build tools. Crafter aims to be a fruitful benchmark for
reinforcement learning by focusing on the following design goals:

- **Research challenges:** Crafter poses substantial challenges to current
  methods, evaluating strong generalization, wide and deep exploration,
  representation learning, and long-term reasoning and credit assignment.

- **Meaningful evaluation:** Agents are evaluated by semantically meaningful
  achievements that can be unlocked in each episode, offering insights into the
  ability spectrum of both reward agents and unsupervised agents.

- **Iteration speed:** Crafter evaluates many agent abilities within a single
  env, vastly reducing the computational requirements over benchmarks suites
  that require training on many separate envs from scratch.

See the research paper to find out more: [Benchmarking the Spectrum of Agent
Capabilities](https://arxiv.org/pdf/2109.06780.pdf)

```
@article{hafner2021crafter,
  title={Benchmarking the Spectrum of Agent Capabilities},
  author={Danijar Hafner},
  year={2021},
  journal={arXiv preprint arXiv:2109.06780},
}
```

## Play Yourself

```sh
python3 -m pip install crafter  # Install Crafter
python3 -m pip install pygame   # Needed for human interface
python3 -m crafter.run_gui      # Start the game
```

<details>
<summary>Keyboard mapping (click to expand)</summary>

| Key | Action |
| :-: | :----- |
| WASD | Move around |
| SPACE| Collect material, drink from lake, hit creature |
| TAB | Sleep |
| T | Place a table |
| R | Place a rock |
| F | Place a furnace |
| P | Place a plant |
| 1 | Craft a wood pickaxe |
| 2 | Craft a stone pickaxe |
| 3 | Craft an iron pickaxe |
| 4 | Craft a wood sword |
| 5 | Craft a stone sword |
| 6 | Craft an iron sword |

</details>




## Custom maps, saving, and enemy settings (quick guide)


### 1) Editing the map (the world layout)

Where the map file is used:
1. `crafter/run_gui.py` starts the game with a map file. It passes `map_file='maps/level1.txt'` into `crafter.Env(...)`.
2. `crafter/env.py` loads that text file (called an ASCII map) when `map_file` is set, reads it into `self._ascii_map`, and also resizes the world to match the map’s width and height. 
3. `crafter/worldgen.py` turns the ASCII map text into real terrain using `parse_ascii_map(...)`, then `generate_world(...)` applies it to the world. 

What you actually edit:
1. Open the map text file (example `maps/level1.txt`) and change the characters.
2. The key legend (what each character means) is inside `parse_ascii_map` in `crafter/worldgen.py`.

Character legend:
1. `|` means grass
2. `o` means stone
3. `~` means water
4. `.` means sand
5. `@` means the player start spot (and the ground under it is water)
6. `C` means a cow (on water)
7. `Z` means a zombie (on water)
8. `S` means a skeleton (on water)

### 2) Saving and loading (F5 and F9)

How it works:
1. `run_gui.py` maps F5 to `save_game` and F9 to `load_game`. 
2. When you press F5, it saves to `saves/game_save.npz`. When you press F9, it loads from `saves/game_save.npz`. 
3. The real save and load code is in `crafter/env.py`:
   1. `Env.save_game(filepath)` writes the world, the player, and all objects (cows, zombies, plants, etc.). 
   2. `Env.load_game(filepath)` puts everything back into the world. 

### 3) Enemy settings (JSON config)

Where it is loaded:
1. `crafter/env.py` loads enemy settings from a JSON file (default name is `enemy_config.json`) when the environment starts. 
2. If the JSON file is missing, it uses built in defaults instead. 

What you can change in `enemy_config.json`:
1. Turn all enemies on or off with `enemies_enabled`. 
2. Turn zombies or skeletons on or off with `zombie_settings.enabled` and `skeleton_settings.enabled`. 
3. Change how often they show up during world generation with `worldgen_spawn_probability`.
4. World generation uses these settings when it decides whether to place zombies or skeletons while building the map. 
5. While you play, the “keep the right number of enemies around you” logic also checks the config before spawning zombies and skeletons. 


![Crafter Video](https://github.com/danijar/crafter/raw/main/media/video.gif)

## Interface

To install Crafter, run `pip3 install crafter`. The environment follows the
[OpenAI Gym][gym] interface. Observations are images of size (64, 64, 3) and
outputs are one of 17 categorical actions.

```py
import gym
import crafter

env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)

obs = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

[gym]: https://github.com/openai/gym

## Evaluation

Agents are allowed a budget of 1M environmnent steps and are evaluated by their
success rates of the 22 achievements and by their geometric mean score. Example
scripts for computing these are included in the `analysis` directory of the
repository.

- **Reward:** The sparse reward is `+1` for unlocking an achievement during
  the episode and `-0.1` or `+0.1` for lost or regenerated health points.
  Results should be reported not as reward but as success rates and score.

- **Success rates:** The success rates of the 22 achievemnts are computed
  as the percentage across all training episodes in which the achievement was
  unlocked, allowing insights into the ability spectrum of an agent.

- **Crafter score:** The score is the geometric mean of success rates, so that
  improvements on difficult achievements contribute more than improvements on
  achievements with already high success rates.

## Scoreboards

Please create a pull request if you would like to add your or another algorithm
to the scoreboards. For the reinforcement learning and unsupervised agents
categories, the interaction budget is 1M. The external knowledge category is
defined more broadly.

### Reinforcement Learning

| Algorithm | Score (%) | Reward | Open Source |
|:----------|----------:|-------:|:-----------:|
| [Curious Replay](https://arxiv.org/pdf/2306.15934.pdf) | 19.4±1.6 | - | [AutonomousAgentsLab/cr-dv3](https://github.com/AutonomousAgentsLab/cr-dv3) |
| [PPO (ResNet)](https://arxiv.org/pdf/2307.03486.pdf)| 15.6±1.6 | 10.3±0.5 | [snu-mllab/Achievement-Distillation](https://github.com/snu-mllab/Achievement-Distillation) 
| [DreamerV3](https://arxiv.org/pdf/2301.04104v1.pdf) | 14.5±1.6 | 11.7±1.9 | [danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| [LSTM-SPCNN](https://arxiv.org/pdf/2208.03374.pdf) | 12.1±0.8 | — | [astanic/crafter-ood](https://github.com/astanic/crafter-ood) |
| [EDE](https://openreview.net/pdf?id=GZDsKahGY-2) | 11.7±1.0 | — | [yidingjiang/ede](https://github.com/yidingjiang/ede) |
| [OC-SA](https://arxiv.org/pdf/2208.03374.pdf) | 11.1±0.7 | — | [astanic/crafter-ood](https://github.com/astanic/crafter-ood) |
| [DreamerV2](https://arxiv.org/pdf/2010.02193.pdf) | 10.0±1.2 | 9.0±1.7 | [danijar/dreamerv2](https://github.com/danijar/dreamerv2) |
| [PPO](https://arxiv.org/pdf/1710.02298.pdf) | 4.6±0.3 | 4.2±1.2 | [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) | 4.3±0.2 | 6.0±1.3 | [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow) |

### Unsupervised Agents

| Algorithm | Score (%) | Reward | Open Source |
|:----------|----------:|-------:|:-----------:|
| [Plan2Explore](https://arxiv.org/pdf/2010.02193.pdf) | 2.1±0.1 | 2.1±1.5 | [danijar/dreamerv2](https://github.com/danijar/dreamerv2) |
| [RND](https://arxiv.org/pdf/1810.12894.pdf) | 2.0±0.1 | 0.7±1.3 | [alirezakazemipour/PPO-RND](https://github.com/alirezakazemipour/PPO-RND) |
| Random | 1.6±0.0 | 2.1±1.3 | — |

### External Knowledge

| Algorithm | Score (%) | Reward | Uses | Interaction | Open Source |
|:----------|----------:|-------:|:-----|:-----------:|:-----------:|
| [Human](https://en.wikipedia.org/wiki/Human) | 50.5±6.8 | 14.3±2.3 | Life experience | 0 | [crafter_human_dataset](https://archive.org/details/crafter_human_dataset) |
| [SPRING](https://arxiv.org/pdf/2305.15486.pdf) | 27.3±1.2 | 12.3±0.7 | LLM, scene description, Crafter paper | 0 | ❌ |
| [Achievement Distillation](https://arxiv.org/pdf/2307.03486.pdf) | 21.8±1.4 | 12.6±0.3 | Reward structure | 1M | [snu-mllab/Achievement-Distillation](https://github.com/snu-mllab/Achievement-Distillation) |
| [ELLM](https://arxiv.org/pdf/2302.06692.pdf) | — | 6.0±0.4 | LLM, scene description | 5M | ❌ |

## Baselines

Baseline scores of various agents are available for Crafter, both with and
without rewards. The scores are available in JSON format in the `scores`
directory of the repository. For comparison, the score of human expert players
is 50.5\%. The [baseline
implementations](https://github.com/danijar/crafter-baselines) are available as
a separate repository.

<img src="https://github.com/danijar/crafter/raw/main/media/scores.png" width="400"/>
