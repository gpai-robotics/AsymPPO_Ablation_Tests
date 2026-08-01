# Teacher Design V2

Teacher V2 is the compressed-encoder follow-up to V1.

## Motivation

Teacher V0 showed that privilege could be used in a crouched, conservative way.
Teacher V1 keeps the same privilege path and adds a targeted anti-crouch term.

Teacher V2 asks a different question:

> is the current terrain encoder too wide, allowing the policy to exploit a
> loose high-capacity terrain code rather than a compact task-relevant one?

This is motivated partly by `rl_locomotion`, which uses much smaller privileged
latents than the current `32`-dim terrain latent used in our V0/V1 stack.

## What Stays Fixed

Relative to Teacher V1, keep fixed:

- terrain-privileged observation definition
- B2-aware warm-start
- actor/critic MLP sizes
- PPO settings
- anti-crouch environment intervention
- task, terrain family, and evaluation stack

## One Deliberate Change

Teacher V2 compresses the terrain branch more aggressively:

- V1 terrain latent: `32`
- V2 terrain latent: `8`

and reduces the encoder hidden widths accordingly.

Current V2 terrain branch:

- `187 -> 64 -> 32 -> 8`

## Why This Is Useful

If V2 performs similarly or better than V1, it suggests the larger terrain
latent was unnecessary and possibly too permissive.

If V2 performs clearly worse, it suggests the current task really does need a
richer terrain code.

Either result is informative, and the comparison is much cleaner than changing
reward, privilege content, and encoder size all at once.

## Registered Task

Historical note:
- `RMA-Go2-Privileged-Teacher-Rough-V2` is archived and no longer kept in the
  active task registry.

- `RMA-Go2-Privileged-Teacher-Rough-V2`

Environment config:

- `rma_go2_lab/envs/teacher/rough_v1_cfg.py`

Runner config:

- `rma_go2_lab/models/teacher/ppo_v2_cfg.py`

## Naming Note

Evaluation artifact names may still include suffixes like:

- `blind_baseline_v1`
- `ood_geometry_v1`
- `ood_push_v1`

These refer to the version of the evaluation protocol, not the teacher
variant. So a file under `artifacts/.../teacher_v2/` with
`...ood_geometry_v1...` in its name means:

- teacher variant = `V2`
- evaluation suite protocol version = `v1`
