"""
Smoke test for the new train_jepa.py functions (greedy eval + divergence
telemetry) with a stubbed env — ALE is only installed on the training box.
Run from project root: python scripts/smoke_v20_trainloop.py
"""

import sys
import types

sys.path.insert(0, ".")

# Stub ale_py + gymnasium so train_jepa (-> envs.wrappers) imports on this
# machine — Atari deps are only installed on the training box. The stubs only
# need to satisfy module-level code in envs/wrappers.py (class definitions);
# make_atari_env itself is never called here.
sys.modules.setdefault("ale_py", types.ModuleType("ale_py"))
if "gymnasium" not in sys.modules:
    _gym = types.ModuleType("gymnasium")

    class _StubBase:
        def __init__(self, *args, **kwargs):
            self.env = args[0] if args else None

    _gym.Env = _StubBase
    _gym.Wrapper = _StubBase
    _gym.ObservationWrapper = _StubBase
    _spaces = types.ModuleType("gymnasium.spaces")

    class _StubSpace:
        def __init__(self, *args, **kwargs):
            pass

    _spaces.Box = _StubSpace
    _spaces.Discrete = _StubSpace
    _gym.spaces = _spaces
    sys.modules["gymnasium"] = _gym
    sys.modules["gymnasium.spaces"] = _spaces

import numpy as np
import torch
import torch.nn.functional as F

from train_jepa import run_greedy_eval, measure_imagination_diagnostics
from training.jepa_world_model import JEPAWorldModel
from models.actor import Actor
from models.critic import Critic

device = torch.device("cpu")
torch.manual_seed(0)

D, ACT, C = 256, 6, 4

cfg = {
    "obs_channels": C, "embed_dim": D, "act_dim": ACT, "cnn_depth": 48,
    "mlp_units": 400, "pred_depth": 4, "pred_heads": 4, "pred_mlp_dim": 512,
    "pred_dropout": 0.1, "batch_length": 30, "history_size": 3,
    "use_ema_target": True, "predictor_loss": "mse_target",
    "inv_dyn_weight": 0.3, "aux_recon_weight": 0.0,
}
wm = JEPAWorldModel(cfg).to(device)
actor = Actor(hidden_dim=D, stoch_dim=0, act_dim=ACT, units=400, discrete=True)
critic = Critic(hidden_dim=D, stoch_dim=0, units=400)


class FakeEnv:
    """Mimics the wrapped Atari env API: float32 (C,64,64) obs."""
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return np.random.randn(C, 64, 64).astype(np.float32) * 0.1, {}

    def step(self, action):
        assert isinstance(action, int) and 0 <= action < ACT, f"bad action {action}"
        self.t += 1
        obs = np.random.randn(C, 64, 64).astype(np.float32) * 0.1
        reward = 1.0 if self.t % 7 == 0 else 0.0
        terminated = self.t >= 25
        return obs, reward, terminated, False, {}


print("=== run_greedy_eval ===")
assert wm.encoder.training and actor.training
rewards = run_greedy_eval(FakeEnv(), actor, wm, device, n_episodes=2)
assert len(rewards) == 2 and all(r == 3.0 for r in rewards), rewards
assert wm.encoder.training and wm.encoder_bn.training and actor.training, \
    "train mode not restored after eval"
print(f"  episode rewards: {rewards}; modes restored")

print("=== measure_imagination_diagnostics ===")
N, T = 8, 30
frozen = {
    "obs": torch.randn(N, T, C, 64, 64) * 0.1,
    "action": F.one_hot(torch.randint(0, ACT, (N, T)), ACT).float(),
}
wm.train()
out = measure_imagination_diagnostics(wm, critic, frozen, device)
expected = {"diag/critic_divergence", "diag/critic_divergence_rel",
            "diag/value_mean_real", "diag/value_mean_pred",
            "diag/value_var", "diag/emb_effective_rank"}
assert set(out) == expected, set(out) ^ expected
for k, v in out.items():
    assert v == v and abs(v) != float("inf"), f"{k} not finite"
    print(f"  {k}: {v:.4f}")
assert 1 <= out["diag/emb_effective_rank"] <= D
assert wm.training, "wm train mode not restored"
print("  wm mode restored")

print("\nTrain-loop smoke tests passed!")
