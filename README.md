# World models for Atari (DreamerV3 + JEPA experiments)

A from-scratch PyTorch implementation of DreamerV3-style model-based reinforcement learning.
The agent learns a model of the game (a "world model") and trains its policy inside that model
instead of only in the real game.

**Status:** work in progress / not finished.

## What works

- **DreamerV3 (RSSM world model)** trained on `PongNoFrameskip-v4` and `BreakoutNoFrameskip-v4`
  (500k environment steps, ~27 h on a V100).
- On Pong the agent went from −21 (random play) to positive episodes.
- The key fix was a foreground-weighted reconstruction loss: pixels that move (ball, paddles)
  get much higher weight, so the model stops ignoring the tiny ball.

## What is unfinished

- **JEPA world model** (`train_jepa.py`): replaces pixel reconstruction with prediction in an
  embedding space. The agent learned a little on Pong, but training collapsed later
  and the approach does not yet match the DreamerV3 baseline.

## Structure

```
train.py          DreamerV3 training loop
train_jepa.py     JEPA variant
models/           encoder, decoder, RSSM, JEPA predictor, actor, critic
training/         world model and actor-critic training, replay buffer
envs/             Atari wrappers (frame skip, grayscale, 64x64, frame stack)
configs/          one YAML file per experiment version
probe_*.py        tools for inspecting what the world model has learned
```

## Run

```
pip install -r requirements.txt
python train.py --config configs/pong.yaml
```

Logging uses Weights & Biases (`WANDB_API_KEY` in a local `.env` file).
