# Novel Techniques for JEPA-RL — Research Ideas

Date: 2026-04-27. Context: v10 running (categorical two-hot critic, step ~292k, reward -9 to -14). These ideas are for future versions and paper contributions.

---

## Related Work Landscape

Key papers that define the space we're working in:

| Paper | Year | What it does | Relation to us |
|-------|------|-------------|----------------|
| LeWorldModel (2603.19312) | Mar 2026 | JEPA from pixels + SIGReg, planning (MPC) | Closest competitor. ViT encoder, continuous control — we do CNN + Atari + AC imagination |
| JEPA for RL (2504.16591) | Apr 2025 | JEPA for RL from images, Cart Pole | Early work, discusses collapse. We go much further |
| TD-JEPA (2510.00739) | Oct 2025 | TD-learning in JEPA space, zero-shot RL | Successor features in latent space, different from our rollout approach |
| ACT-JEPA (2501.14622) | Jan 2025 | JEPA + imitation learning, action chunking | Action prediction as auxiliary — could borrow this idea |
| MuDreamer (2405.15083) | May 2024 | DreamerV3 without reconstruction | Uses value + action prediction instead of decoder. Atari 100k baseline |
| Dreamer-CDP (2603.07083) | Mar 2026 | Reconstruction-free Dreamer, JEPA-style predictor | ICLR 2026 workshop tiny paper. Matches Dreamer on Crafter |
| SGF (2506.02612) | Jun 2025 | Self-supervised WM, no RNN/transformer/recon | Shows much DreamerV3 complexity is unnecessary. Atari 100k |
| TD-MPC2 (2310.16828) | 2024 | Deterministic latent WM + discrete regression (CE) for value/reward | Validates our categorical critic approach. Uses SimNorm |
| MAD-TD (2410.08896) | ICLR 2025 | Model-augmented data stabilizes value estimation | Relevant for our critic collapse problem |
| On Rollouts in MBRL (ICLR 2025) | 2025 | Infoprop: uncertainty-based rollout truncation | Adaptive horizon based on epistemic uncertainty |
| Plasticine (2504.17490) | Apr 2025 | Benchmark for plasticity loss in deep RL, 13+ methods | Framework for testing reset strategies |
| Primacy Bias (Nikishin, ICML 2022) | 2022 | Periodic network resets combat overfitting to early data | Foundation for the "planned resets" idea |
| C-CHAIN (ICML 2025) | 2025 | Reduce churn to prevent rank collapse and plasticity loss | Churn = unnecessary value estimate changes. Relevant to our collapse |
| Stay Hungry / SBP (ICML 2025) | 2025 | Cycle reset + inner distillation for sustained plasticity | Informed reset that preserves useful knowledge |
| DINO-WM (2411.04983) | ICML 2025 | World model on DINOv2 features, zero-shot planning | Pretrained representations — different paradigm but good comparison |

**Our unique position:** JEPA + AC imagination for discrete Atari. LeWM does planning (not AC). MuDreamer uses RSSM (not JEPA). Nobody does our exact combination.

---

## Novel Technique Ideas

### 1. Contrastive Dynamics Prediction (replaces MSE prediction loss)

**Status quo:** All JEPA world models (ours, LeWM, MuDreamer) use MSE for prediction loss.

**Idea:** Use InfoNCE contrastive loss instead. The predictor should *recognize* the correct future embedding among N distractors, not regress to the exact value.

```python
# Current
pred_loss = F.mse_loss(predictor(z_t, a_t), z_next)

# Proposed
pred = predictor(z_t, a_t)                     # (B, D)
positives = z_next                              # (B, D)
negatives = z_next[torch.randperm(B)]           # shuffled batch as negatives
logits_pos = (pred * positives).sum(-1)         # (B,)
logits_neg = (pred * negatives).sum(-1)         # (B,)
pred_loss = -F.logsigmoid(logits_pos).mean() - F.logsigmoid(-logits_neg).mean()
```

**Why it helps our specific problem:**
- MSE compounding error is *systematic* in deterministic predictors — errors accumulate in one direction. Contrastive loss cares about *ranking*, not exact position. Less compounding.
- DreamerV3 escapes this because stochastic sampling breaks systematic error. We have no sampling. Contrastive loss fills that gap.
- Nobody has done this in JEPA-RL. CPC was used for representation learning, never as dynamics-prediction loss in a world model running AC imagination.

**Paper angle:** "Contrastive dynamics prediction eliminates the need for stochastic latents in imagination-based MBRL"

**Novelty:** Very high. **Effort:** Medium. **Direct help for v10 problem:** Yes — less compounding error in rollouts.

---

### 2. Predictor Ensemble for Uncertainty-Weighted Imagination

**Idea:** Run 3-5 predictor heads (shared backbone, different final layers) during imagination. Variance between them = natural uncertainty estimate.

Use this for three things simultaneously:
- **Weight lambda-returns:** Low confidence → more critic bootstrap, less model reward
- **Adaptive rollout length:** Cut when variance > threshold
- **Scale actor gradient:** Uncertain steps → weaker policy update

```python
preds = [head(ctx_embs, ctx_acts) for head in predictor_heads]  # list of (B, D)
mean_pred = torch.stack(preds).mean(0)
variance = torch.stack(preds).var(0).mean(-1)  # (B,) scalar uncertainty per sample

# Uncertainty-weighted lambda returns
confidence = 1.0 / (1.0 + variance / variance_threshold)
effective_reward = confidence * model_reward + (1 - confidence) * 0  # fade reward to 0 when uncertain
```

**Why it's new:** Ensembles are used in model-free RL (SAC Q-ensembles, TD-MPC2), but nobody has used *predictor ensembles in JEPA imagination* to steer actor-critic training. Uncertainty in *dynamics* is different from uncertainty in *values*.

**Why it helps us:** v6 collapsed because the predictor was systematically wrong after regime shift, and the critic had no way to know. Ensemble variance lets you *see* imagination becoming unreliable and dampen gradient updates automatically.

**Novelty:** High. **Effort:** Medium (3-5 extra linear heads). **Direct help:** Yes — automatic collapse detection.

---

### 3. Imagination Horizon Curriculum

**Idea:** Start with horizon=1, gradually increase to 15. Surprisingly, nobody does this explicitly.

```python
horizon = min(1 + global_step // 20000, max_horizon)  # 1 at start, 15 by 280k
```

**Why it works:**
- Horizon=1 is nearly model-free: one-step model reward + critic bootstrap. Almost no compounding error.
- Critic learns correct short-horizon values first. Then you extend to 2, and critic already has a solid foundation to bootstrap from.
- The transition from "short-horizon negative returns" to "long-horizon potentially positive returns" is gradual.

**Why it helps us:** v6 collapse happened because critic suddenly got 15-step rollouts with completely different return distributions. With curriculum, the regime shift would be gradual.

**Nobody does this.** DreamerV3 uses fixed horizon=15. TD-MPC uses fixed planning horizon. It's a blind spot in the field.

**Novelty:** High (surprising blind spot). **Effort:** Very low (5 lines). **Direct help:** Yes — smooths regime transitions.

---

### 4. Embedding Augmentation During Imagination ("Soft Stochasticity")

**Idea:** Add controlled noise to predicted embeddings during imagination, without changing the architecture:

```python
next_emb = predictor(ctx_embs, ctx_acts)
if imagining:
    noise_scale = 0.05  # tunable, or schedule from 0.1 → 0.01
    next_emb = next_emb + noise_scale * torch.randn_like(next_emb)
```

**Key insight:** This is NOT the same as having a stochastic model. SIGReg forces embeddings to N(0,I), so we *know* the statistical structure. Gaussian noise keeps imagined embeddings *within the trained distribution* instead of drifting away. It's like reverse diffusion guidance.

**More interesting variant:** Make noise *depend on prediction loss*:

```python
# Per-sample noise based on how far prediction is from SIGReg prior
emb_norm_deviation = (next_emb.norm(dim=-1, keepdim=True) - expected_norm).abs()
adaptive_noise = noise_scale * emb_norm_deviation.clamp(max=1.0)
next_emb = next_emb + adaptive_noise * torch.randn_like(next_emb)
```

High deviation → more noise → more exploration in imagination. Low deviation → precise exploitation.

**Novelty:** Medium-high. **Effort:** Very low. **Direct help:** Yes — breaks systematic compounding error.

---

### 5. Adversarial Latent Dynamics

**Idea:** Train a small discriminator distinguishing:
- **Real** embedding sequences (z_t → z_{t+1} from encoder)
- **Imagined** sequences (z_t → pred(z_t, a_t) from predictor)

Use discriminator score as additional predictor training signal (GAN loss in latent space).

```python
real_pairs = torch.cat([z_t, z_next_real], dim=-1)         # from encoder
fake_pairs = torch.cat([z_t, predictor(z_t, a_t)], dim=-1) # from predictor
disc_real = discriminator(real_pairs)
disc_fake = discriminator(fake_pairs)
# Standard GAN losses for discriminator and predictor
```

**Why it matters:** MSE doesn't capture whether predicted embeddings have the right *distributional properties*. Embedding can have correct mean but wrong variance/correlation. A discriminator catches this.

**Paper angle:** Connects JEPA world models with GANs — two major paradigms never crossed in MBRL. SIGReg enforces *single-timestep* N(0,I), but says nothing about *sequences*. The discriminator captures temporal structure.

**Novelty:** Very high. **Effort:** Medium-high. **Direct help:** Indirect — improves predictor fidelity over time.

---

### 6. Reward-Conditioned Representation Learning

**Idea:** SIGReg treats all states equally. But for RL, reward-relevant states are far more important. Weight SIGReg/prediction loss by proximity to reward events:

```python
# reward_proximity: 1.0 near rewards, decay to 0.1 for "dead time"
reward_proximity = compute_reward_proximity(rewards, decay=0.95)  # backward exponential
weighted_pred_loss = (per_sample_pred_loss * reward_proximity).mean()
```

**Alternative:** Reward-conditioned contrastive loss — embeddings near reward events should be easier to distinguish from each other than embeddings during "dead time." This allocates more representation capacity to decision-relevant states.

**Why it's new:** ALL JEPA literature (LeWM, I-JEPA, V-JEPA) treats all timesteps uniformly. RL is fundamentally different because not all information is equally valuable. Nobody has made JEPA reward-aware.

**Novelty:** Very high. **Effort:** Low-medium. **Direct help:** Potentially large for sparse-reward environments like Pong.

---

### 7. Active Imagination (Dreaming from Critic's Perspective)

**Idea:** Instead of starting imagination rollouts from random replay states, start from states where the **critic is most uncertain**:
- High TD-error
- Large value change between consecutive updates
- States near reward boundaries

```python
# Score each replay state by critic uncertainty
td_errors = (lambda_returns - critic(embeddings)).abs()
sample_weights = td_errors / td_errors.sum()
start_indices = torch.multinomial(sample_weights, num_starts)
```

**Why it helps us:** v6 collapsed in a narrow region of state space (the transition from losing to winning). If imagination had focused there, critic would have gotten much more training on that exact transition *before* it happened in reality.

**This is active learning for the critic.** Extensive literature in supervised ML, never applied to imagination-based MBRL.

**Novelty:** High. **Effort:** Low. **Direct help:** Yes — pre-trains critic on regime boundaries.

---

### 8. Norm-Preserving Prediction (Symplectic-inspired)

**Idea:** If SIGReg keeps embeddings on N(0,I), the predictor should respect this geometry. Standard transformers don't — predictions can drift off-distribution over many steps.

Options:
- **SimNorm** after each prediction step (from TD-MPC2, but never combined with SIGReg)
- **Project-back:** After prediction, re-normalize to target embedding statistics
- **Orthogonal predictor parameterization:** Constrain predictor weights to preserve norms

```python
next_emb = predictor(z_t, a_t)
# Re-normalize to match SIGReg target distribution
next_emb = (next_emb - next_emb.mean(0)) / (next_emb.std(0) + 1e-6)  # batch-level
# Or softer: exponential moving average normalization
```

**Why it matters:** SIGReg gives a *known target distribution*. This is a stronger guarantee than just normalization — you know *exactly* what the distribution should look like. Nobody has exploited this connection.

**Novelty:** Medium-high. **Effort:** Low. **Direct help:** Yes — prevents embedding drift in long rollouts.

---

### 9. Bi-Level Actor-Critic (Short + Long Horizon)

**Idea:** Train two critics simultaneously:
- **Critic-short** (horizon=3): Low compounding error, accurate but myopic
- **Critic-long** (horizon=15): Higher error, sees further ahead

Actor loss uses a blend:

```python
advantage = alpha * short_advantage + (1 - alpha) * long_advantage
# alpha anneals from 1.0 → 0.3 over training
```

**Why it works:** Early in training, short rollouts are reliable and long ones are garbage. Late in training, long rollouts are better but can collapse (our v6). Two critics provide a natural fallback — if long-critic collapses, short-critic still dominates.

**Nobody does this.** All MBRL papers use a single horizon. Some adjust the horizon, but none run *parallel critics with different horizons*.

**Novelty:** High. **Effort:** Medium (duplicate critic + separate imagination). **Direct help:** Yes — built-in collapse resistance.

---

### 10. Latent Replay with Temporal Stitching

**Idea:** Find states in the replay buffer where embeddings are close (||z_a - z_b|| < epsilon) but from different episodes. Use these as "wormholes" for imagination — start in one episode, jump to a similar state from another episode, continue imagination from there.

```python
# Build approximate nearest-neighbor index over embeddings in buffer
# During imagination start selection:
query_emb = encoder(random_replay_state)
neighbors = ann_index.search(query_emb, k=5)  # from different episodes
# Stitch: start imagination from neighbor embedding
```

**What this gives:** Much more diverse imagination experience. Actor sees transitions it would never see in a single episode. Especially useful for our problem — critic could experience "go from negative to positive reward" *before* it happens in reality, by stitching fragments from different episodes.

**Novelty:** Very high. **Effort:** Medium-high (needs ANN index). **Direct help:** Yes — diverse experience prevents narrow critic training.

---

## Priority Ranking for Implementation

### Tier 1: Test immediately (low effort, high potential)
1. **Imagination horizon curriculum (#3)** — 5 lines of code. Can test in v11.
2. **Embedding augmentation (#4)** — 3 lines of code. Can A/B test as a flag.
3. **Norm-preserving prediction (#8)** — Simple post-prediction normalization step.

### Tier 2: Next experiment batch (medium effort, strong paper contribution)
4. **Predictor ensemble (#2)** — Needs extra heads but architecture is modular.
5. **Active imagination (#7)** — TD-error weighted sampling, straightforward.
6. **Bi-level critic (#9)** — Duplicate critic with short horizon.

### Tier 3: Major paper contribution (higher effort, highest novelty)
7. **Contrastive dynamics (#1)** — Fundamental change to prediction loss. High novelty.
8. **Reward-conditioned representations (#6)** — Challenges a JEPA-wide assumption.
9. **Adversarial latent dynamics (#5)** — GAN + JEPA, novel but tricky to stabilize.
10. **Temporal stitching (#10)** — Needs ANN infrastructure but very creative.

---

## Paper Positioning

Our unique contribution in context of the literature:

1. **JEPA + AC imagination for discrete Atari** — LeWM does planning, MuDreamer uses RSSM. Nobody does our combination.
2. **SIGReg + categorical critic** — Original combination. TD-MPC2 uses categorical values with a different world model.
3. **Regime-shift analysis** — Detailed documentation of v6 collapse and fix attempts (v7-v10) is itself a contribution.
4. **Any of the Tier 3 techniques** — If one works, it's a clear novel contribution on top of the system.

Best paper structure: "We built a JEPA-RL system, identified the critic collapse problem unique to deterministic latent models, and solved it with [technique X]."
