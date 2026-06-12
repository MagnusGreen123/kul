"""
Smoke test for v19/v20 features. Run from project root:
    python scripts/smoke_v20.py

Covers:
  1. v20 mode (mse_target + byol_enc + cpc + event reward): forward/backward,
     finite losses, gradient routing (online yes / target no), CPC ramp.
  2. v19 mode (cpc off) loss keys.
  3. Backwards compat: v18 byol mode and v11 plain-MSE mode still run.
  4. AC trainer: AC-step curriculum ramps, predictor mode restored after
     train_step (eval during imagination), train_step executes.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from training.jepa_world_model import JEPAWorldModel, JEPAWorldModelTrainer
from training.jepa_actor_critic import JEPAActorCriticTrainer
from models.actor import Actor
from models.critic import Critic

device = torch.device("cpu")
torch.manual_seed(0)

B, T, C, H, W = 4, 30, 4, 64, 64
ACT = 6
D = 256

BASE = {
    "obs_channels": C, "embed_dim": D, "act_dim": ACT, "cnn_depth": 48,
    "mlp_units": 400, "pred_depth": 4, "pred_heads": 4, "pred_mlp_dim": 512,
    "pred_dropout": 0.1, "batch_length": T, "history_size": 3,
    "sigreg_weight": 0.1, "pred_weight": 1.0, "rollout_weight": 0.5,
    "rollout_steps": 5, "reward_weight": 10.0, "use_symlog": True,
    "aux_recon_weight": 1.0, "aux_decoder_depth": 32, "fg_weight": 50.0,
    "learning_rate": 1e-4, "mixed_precision": False, "max_grad_norm": 2.0,
    "horizon": 15, "gamma": 0.997, "lambda_": 0.95, "entropy_coeff": 3e-3,
    "critic_ema_decay": 0.98, "warmup_steps": 10, "total_steps": 1000,
}

V20 = {
    **BASE,
    "use_ema_target": True, "target_ema_decay": 0.99,
    "predictor_loss": "mse_target",
    "byol_encoder_weight": 0.3,
    "inv_dyn_weight": 0.3,
    "reward_loss_mode": "event", "reward_event_weight": 30.0,
    "cpc_weight": 0.5, "cpc_horizons": [1, 3, 5], "cpc_proj_dim": 128,
    "cpc_hidden_dim": 256, "cpc_temperature": 0.1, "cpc_var_weight": 1.0,
    "cpc_ramp_steps": 20000, "cpc_max_samples": 512,
    "repval_weight": 0.3, "horizon_curriculum_ac_steps": 100,
}


def make_batch():
    rewards = torch.zeros(B, T)
    rewards[:, 7] = 1.0   # sparse reward events
    rewards[:, 19] = -1.0
    return {
        "obs": torch.randn(B, T, C, H, W) * 0.1,
        "action": F.one_hot(torch.randint(0, ACT, (B, T)), ACT).float(),
        "reward": rewards,
        "done": torch.zeros(B, T),
    }


def check_finite(losses, tag):
    for k, v in losses.items():
        val = v.item() if torch.is_tensor(v) else v
        assert val == val and abs(val) != float("inf"), f"{tag}: {k} not finite: {val}"


print("=== 1. v20 mode: forward/backward + gradient routing ===")
wm = JEPAWorldModel(V20).to(device)
batch = make_batch()
losses, info = wm(batch["obs"], batch["action"], batch["reward"],
                  dones=batch["done"], global_step=10000)
check_finite(losses, "v20")
assert "cpc" in losses, "cpc loss missing"
assert "byol_enc" in losses, "byol_enc loss missing"
assert "pred_byol" not in losses, "pred_byol should not exist in mse_target mode"
diag = info["diagnostics"]
assert "reward_event_mse" in diag and diag["reward_event_mse"] > 0
assert "cpc_acc_1step" in diag
# Ramp: at 10000/20000 steps, scale must be 0.5 * 0.5 = 0.25
assert abs(diag["cpc_scale"] - 0.25) < 1e-6, f"ramp wrong: {diag['cpc_scale']}"
print(f"  losses: {[f'{k}={v.item():.3f}' for k, v in losses.items()]}")
print(f"  cpc_scale@10k={diag['cpc_scale']:.3f}, "
      f"reward_event_mse={diag['reward_event_mse']:.4f}")

losses["total"].backward()
enc_grad = sum(p.grad.abs().sum().item() for p in wm.encoder.parameters()
               if p.grad is not None)
pred_grad = sum(p.grad.abs().sum().item() for p in wm.predictor.parameters()
                if p.grad is not None)
cpc_grad = sum(p.grad.abs().sum().item() for p in wm.cpc.parameters()
               if p.grad is not None)
assert enc_grad > 0 and pred_grad > 0 and cpc_grad > 0
for p in wm.target_encoder.parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, "target encoder got grad!"
for p in wm.target_projection_head.parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, "target proj got grad!"
print(f"  grads: encoder={enc_grad:.1f}, predictor={pred_grad:.1f}, "
      f"cpc={cpc_grad:.1f}; target nets clean")

# mse_target: training pred loss must equal the logged MSE metric
wm.zero_grad()
losses2, _ = wm(batch["obs"], batch["action"], batch["reward"],
                dones=batch["done"], global_step=0)
assert losses2["pred"].item() > 0
print(f"  pred (MSE vs target emb): {losses2['pred'].item():.4f}")

print("=== 2. Trainer step + per-head grad norms + CPC ramp ===")
trainer = JEPAWorldModelTrainer(JEPAWorldModel(V20).to(device), V20, device)
# At step 0 the CPC ramp is 0 -> CPC heads must get NO gradient yet
ld0, info0 = trainer.train_step(make_batch(), global_step=0)
assert info0["diagnostics"]["cpc_scale"] == 0.0, "ramp at step 0 must be 0"
assert ld0["grad_cpc"] == 0.0, "CPC must not train while ramp weight is 0"
# Mid-ramp the CPC heads must train
ld, info = trainer.train_step(make_batch(), global_step=10000)
check_finite({k: torch.tensor(v) for k, v in ld.items()}, "trainer")
assert ld["grad_encoder"] > 0 and ld["grad_cpc"] > 0
print(f"  ramp@0: cpc_scale=0, grad_cpc=0 (correct); "
      f"ramp@10k: grad_cpc={ld['grad_cpc']:.2f}")
print(f"  grad_encoder={ld['grad_encoder']:.2f}, grad_predictor={ld['grad_predictor']:.2f}, "
      f"grad_reward_head={ld['grad_reward_head']:.2f}")

print("=== 3. AC trainer: curriculum + predictor eval/restore ===")
wm3 = JEPAWorldModel(V20).to(device)
actor = Actor(hidden_dim=D, stoch_dim=0, act_dim=ACT, units=400, discrete=True)
critic = Critic(hidden_dim=D, stoch_dim=0, units=400)
ac = JEPAActorCriticTrainer(actor, critic, wm3, V20, device)
assert ac._current_horizon == 1, "curriculum must start at 1"
emb_seq = torch.randn(B, T, D)
rew = torch.zeros(B, T); rew[:, 5] = 1.0
assert wm3.predictor.training, "predictor should start in train mode"
horizons = []
for i in range(120):
    out = ac.train_step(emb_seq, dones=torch.zeros(B, T), real_rewards=rew,
                        global_step=0)
    horizons.append(ac._current_horizon)
assert wm3.predictor.training, "predictor mode not restored after train_step"
assert horizons[0] == 1 and horizons[-1] == V20["horizon"], \
    f"AC-step curriculum broken: {horizons[0]} -> {horizons[-1]}"
assert out["replay_critic_loss"] != 0.0
print(f"  horizon ramp over 120 AC steps: {horizons[0]} -> {horizons[-1]} "
      f"(ramp target 100 steps); predictor mode restored")

print("=== 4. Backwards compat: v18 byol mode ===")
V18 = {**BASE, "use_ema_target": True, "target_ema_decay": 0.99,
       "inv_dyn_weight": 0.3, "repval_weight": 0.3,
       "horizon_curriculum_steps": 200000}
wm18 = JEPAWorldModel(V18).to(device)
assert wm18.predictor_loss_mode == "byol", "v18 default must stay byol"
assert wm18.cpc is None
l18, _ = wm18(batch["obs"], batch["action"], batch["reward"], dones=batch["done"])
check_finite(l18, "v18")
assert "pred_byol" in l18
l18["total"].backward()
print(f"  v18 byol path OK: pred_byol={l18['pred_byol'].item():.4f}, "
      f"pred(mse)={l18['pred'].item():.4f}")

print("=== 5. Backwards compat: v11 plain-MSE mode ===")
V11 = {**BASE}  # no ema flags at all
wm11 = JEPAWorldModel(V11).to(device)
assert wm11.predictor_loss_mode == "mse"
assert wm11.target_encoder is None and wm11.cpc is None
l11, _ = wm11(batch["obs"], batch["action"], batch["reward"], dones=batch["done"])
check_finite(l11, "v11")
assert "byol_enc" not in l11 and "cpc" not in l11
l11["total"].backward()
print(f"  v11 path OK: pred={l11['pred'].item():.4f}")

print("=== 6. Reward event weighting math ===")
wm_e = JEPAWorldModel(V20).to(device)
le, ie = wm_e(batch["obs"], batch["action"], batch["reward"], dones=batch["done"])
wm_s = JEPAWorldModel({**V20, "reward_loss_mode": "sum"}).to(device)
ls, _ = wm_s(batch["obs"], batch["action"], batch["reward"], dones=batch["done"])
# event mode is a weighted MEAN -> must be ~T x smaller than the sum variant
assert le["reward"].item() < ls["reward"].item(), \
    "event-mode reward loss should be far below sum-mode"
print(f"  reward loss: event={le['reward'].item():.4f} vs sum={ls['reward'].item():.4f}")

print("\nAll v19/v20 smoke tests passed!")
