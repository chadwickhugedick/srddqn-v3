# Self-Rewarding Double DQN (SRDDQN) for Crypto Trading

This repository contains the implementation of a Self-Rewarding Double Deep Q-Network (SRDDQN) algorithmic trading agent designed for cryptocurrency markets. It features a sophisticated Multi-Timeframe (MTF) data pipeline, an advanced TimesNet pre-trained reward network, and a stable PPO baseline setup.

## Features
- **Multi-Timeframe Pipeline**: Processes 1H, 4H, and 1D technical indicators (RSI, MACD, Bollinger Bands, etc.) into a robust 27-feature observation space.
- **TimesNet Pre-training**: Implements a TimesNet model (using Fast Fourier Transforms) to predict perfect-hindsight "Min-Max" macro trends via Supervised Learning.
- **Self-Rewarding Mechanism**: Augments the standard RL reward dynamically using the TimesNet predictions (`r_t = max(r_env, r_timesnet)`), encouraging the agent to hold through macro-trends and avoid overfitting to high-frequency noise.
- **Walk-Forward Validation**: Built-in 12-month / 3-month rolling walk-forward validation for rigorous out-of-sample benchmarking.
- **Weights & Biases Sweeps**: Out-of-the-box configurations to aggressively hyperparameter tune TimesNet architectures and Double DQN (SB3) configurations.

## Setup
Ensure you have Python 3.13+ and [Poetry](https://python-poetry.org/) installed.

```bash
poetry install
```

## Usage

**Optimize TimesNet (Reward Network)**:
```bash
wandb sweep sweep_timesnet.yaml
# Run the generated wandb agent command
```

**Optimize Double DQN**:
```bash
wandb sweep sweep_dqn.yaml
# Run the generated wandb agent command
```

**Walk-Forward Validation**:
```bash
python training/train_srddqn_wfv.py
```

## Structure
- `/baselines`: Contains legacy PPO baselines and evaluation scripts.
- `/envs`: Core Gym environments, including the `CryptoMtfEnv` and `SRDRLWrapper`.
- `/models`: Deep learning architectures (TimesNet).
- `/training`: Training pipelines, WFV logic, and Sweep entry points.
- `/data`: Parquet data pipeline storage.
