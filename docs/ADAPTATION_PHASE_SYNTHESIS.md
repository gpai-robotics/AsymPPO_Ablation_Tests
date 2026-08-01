# Adaptation Phase Synthesis

This note freezes the first completed adaptation-phase comparison after the
post-fix restart of the switched-environment regime.

## Frozen Checkpoints

### studentNA

- task: `RMA-Go2-Adaptation-Student-Rough-NoAdapt`
- experiment: `go2_adaptation_student_no_adapt_v0`
- checkpoint:
  `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt`

### studentAdapt-V0

- task: `RMA-Go2-Adaptation-Student-Rough-History`
- experiment: `go2_adaptation_student_history_v0`
- checkpoint:
  `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt`

### studentAdapt-V1

- task: `RMA-Go2-Adaptation-Student-Rough-History-V1`
- experiment: `go2_adaptation_student_history_v1`
- checkpoint:
  `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v1/2026-04-23_12-31-29/model_1999.pt`
- status:
  - frozen as a canonical checkpoint
  - canonical post-fix eval matrix complete

### studentAdapt-V2

- task: `RMA-Go2-Adaptation-Student-Rough-History-V2`
- experiment: `go2_adaptation_student_history_v2`
- checkpoint:
  `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v2/2026-04-23_16-00-02/model_1999.pt`
- status:
  - frozen as a canonical checkpoint
  - canonical post-fix eval matrix complete
  - architecturally distinct from `V1`
  - not empirically distinct from `V1` in canonical evaluation outputs

## Final Training Comparison

Final training snapshot at iteration `1999/2000`:

| Metric | studentNA | studentAdapt-V0 |
|---|---:|---:|
| reward | 30.85 | 31.72 |
| episode length | 843.98 | 872.87 |
| timeout | 0.8000 | 0.8081 |
| terrain levels | 3.6164 | 3.5847 |
| error_vel_xy | 0.2424 | 0.1940 |
| error_vel_yaw | 0.1833 | 0.1778 |
| base_height termination | 0.1778 | 0.1679 |
| base_orientation termination | 0.0016 | 0.0020 |
| feet_slide | -0.0518 | -0.0485 |
| switch_reached_frac | 0.8816 | 0.8316 |

## Conclusion

The first imitation-based adaptation route (`Adapt-V0`) is a real positive
result.

Honest summary:

- `studentNA` proved that a deployable proprio-only student can become strong
  even under hidden mid-episode dynamics switches.
- `studentAdapt-V0` finished slightly stronger overall than `studentNA`.
- The final gain is not a blowout, but it is broad enough to count as a real
  adaptation win:
  - higher total reward
  - longer episodes
  - better timeout rate
  - lower translational tracking error
  - slightly lower body-height failure rate
  - slightly lower foot slide

That means the adaptation branch is now justified by a completed result, not
just by scaffolding or design intent.

## V1 And V2 Closeout

`studentAdapt-V1` and `studentAdapt-V2` are now both frozen and canonically
evaluated.

What the repo truth says:

- `V1` is a real completed explicit-latent result
- `V2` is a real completed modular architectural result
- `V2` does not introduce a new empirical result over `V1`

Canonical evidence:

- the full post-fix eval outputs for `V1` and `V2` are identical
- the archived checkpoints are different files, but their shared actor/critic
  weights are identical
- the only parameter-name difference is the architectural rename from
  `history_encoder.*` to `adaptation_module.*`

So the correct interpretation is:

- `V1` remains the canonical explicit-latent adaptation result
- `V2` remains the canonical modularization milestone
- neither `V1` nor `V2` displaced `V0` as a clearly stronger empirical
  adaptation winner in the current repo

## Interpretation

The most important project-level takeaway is:

- `V3 final` established a meaningful privileged expert upper bound
- `studentNA` established a serious no-adaptation deployable baseline
- `studentAdapt-V0` showed that history-based adaptation plus frozen-teacher
  guidance can beat that no-adaptation baseline on the same switched task
- `studentAdapt-V1` showed that the explicit-latent path can train to a mature
  completed result
- `studentAdapt-V2` showed that the explicit modular split can also train to
  completion, but did not yield a new empirical outcome beyond `V1`

So the adaptation chapter now has a clean first positive result.

## What This Does Not Yet Prove

This result does **not** by itself prove that:

- `Adapt-V0` is the final deployable architecture
- explicit latent prediction is unnecessary
- the project has solved sim-to-real deployment

It only proves that the first adaptation mechanism is useful and worth keeping
in the ladder.

## Evaluation Method Reference

Use [EVALUATION_METHODS.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/EVALUATION_METHODS.md)
as the canonical description of how the repo currently:

- measures per-scenario metrics
- computes the scalar isolated/OOD ranking score
- handles one-shot switch scenarios
- verifies that requested friction / mass / motor overrides were actually
  applied
- separates ranking metrics from gait diagnostics

That methods note also records the explicit evaluation-audit validation runs and
the remaining limitations of the current scoring rule.

## Eval Commands

Run these from `/home/bhuvan/tools/IsaacLab`.

### studentNA gait

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --num_envs 16 --steps 200 --command-profile standstill --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_standstill.json --headless
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --num_envs 16 --steps 200 --command-profile forward --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_forward.json --headless
```

### studentAdapt-V0 gait

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --num_envs 16 --steps 200 --command-profile standstill --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_standstill.json --headless
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --num_envs 16 --steps 200 --command-profile forward --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_forward.json --headless
```

### Canonical isolated suite

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/run_isolated_suite.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --suite blind_baseline_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_na
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/run_isolated_suite.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --suite blind_baseline_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/adaptation_student_v0
```

### OOD suites

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --suite ood_geometry_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_na
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --suite ood_geometry_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_v0
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --suite ood_dynamics_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_na
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --suite ood_dynamics_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_v0
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --suite ood_push_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_na
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --suite ood_push_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_v0
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-NoAdapt --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt --suite ood_switch_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_na
```

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Adaptation-Student-Rough-History --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt --suite ood_switch_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/adaptation_student_v0
```
