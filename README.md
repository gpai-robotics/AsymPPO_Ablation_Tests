# Ablations & Comparison SOP

For ablation task configurations and standard run commands, refet to [Ablation Task SOP](docs/policy_ablation_sop.md).

## Core Metrics (used across all ablations unless noted)

| Metric | What it captures |
|---|---|
| Episode length | Survival / failure rate proxy |
| Distance from origin | Net forward progress |
| Stable progress | Progress without falls/resets |
| Base stability | Body orientation/velocity variance |
| Joint symmetries | Left-right gait consistency |
| Success rate | Task completion (terrain-specific) |

Report all ablations using this metric set for comparability, adding task-specific metrics only where noted.

---

## Checkpoint Validation Script

The checkpoint validation entrypoint can be found at [Validation Tools](scripts/eval/ablation_eval.py)

The script can be run using the following command :

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/ablation_eval.py \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --checkpoint $ASYMPPO_CKPT \
  --num_envs 16 \
  --teleop-keyboard \
```

---

## 1. Reward Design Ablation (Priority — highest novelty)

**Goal:** Isolate the contribution of each new reward term.

**New reward terms**
- Hip joint deviation - 
- Stable progress
- Adaptive swing recovery
- (Optional) Gait rewards, e.g. air time variance — to test redundancy of gait-phase generators

**Procedure**
1. Baseline: standard locomotion rewards only (no new terms) vs. full new reward set
2. Leave-one-out: remove each new reward individually, compare against full set
3. (Optional) Add/remove gait-phase reward to test whether it's redundant given the other new terms

**Metrics:** full core metric set, per terrain/scenario

---

## 2. Terrain Curriculum Consolidation

**Goal:** Test combining terrain curriculums into a single training run vs. separate runs, at reduced difficulty.

**Setup**
- Lower domain randomization at start (friction, COM shift)
- Lower initial difficulty (step height, slope angle, etc.)

**Comparisons**
1. All terrains combined in one run — vs. separate per-terrain runs
2. Stairs + rough + slopes combined — vs. separate/full-combined runs

**Metrics:** episode length, stable progress, distance from origin

---

## 3. Domain Randomization Range

**Goal:** Validate DR ranges against literature and measure OOD robustness.

**Steps**
1. Survey DR ranges used in comparable papers (friction, payload, COM shift, others as relevant)
2. Benchmark current ranges against those
3. Run OOD tests sweeping each parameter beyond trained range

**Primary parameters:** friction (esp. low-friction regime), payload, COM shift

**Metrics:** success rate, episode length, distance covered (+ core metrics above)

---

## 4. AsymPPO vs. Student-Teacher

**Goal:** Compare AsymPPO (asymmetric actor-critic) against a Student-Teacher architecture.

**Setup**
- Teacher observations = critic observations
- Student observations = actor observations

**Comparisons**
- AsymPPO vs. Student-Teacher, each run:
  - Without new reward terms
  - With new reward terms

(4 conditions total: {AsymPPO, Student-Teacher} × {baseline rewards, new rewards})

**Metrics:** core metric set

---

## 5. History Encoder Length

**Goal:** Determine effect of observation history length on performance.

**Configs tested:** 20, 50, 100, 150 steps (100 steps ≈ 2 seconds — reference point)

**Procedure**
- Fix scenario across all configs (e.g., 5 runs of stair climbing) for controlled comparison
- Run each history length under identical conditions

**Metrics:** core metric set, scenario-specific (e.g. stair success rate)

---

## 6. Velocity Range

**Goal:** Determine performance of the pipeline in a range of velocities in all axes  

**Configs Tested:** +-0.2 m/s from the baseline velocity range

**Procedure**
- Fix scenario across all velocity ranges for controlled comparison
- Run each training velocity range under identical conditions

**Metrics:** velocity tracking, success rate

## Reporting Template (per ablation)

- Config diff (what changed)
- Core metrics table (mean ± std across seeds/runs)
- Scenario-specific metrics if applicable
- Plots: episode length over training, distance-from-origin over training
- Notes / anomalies

