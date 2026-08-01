# RMA Reference Map

This note maps the most useful papers from the original RMA reference list to
the concrete design problems in this repo.

The goal is not to summarize every citation. The goal is to answer:

> which papers are most useful for the current project, and what part of the
> repo or pipeline should each one inform?

## Current Repo Questions

The current project is effectively organized around four questions:

1. How do we build a strong deployable blind locomotion prior?
2. How do we define a meaningful privileged upper bound?
3. How do we make within-episode hidden changes actually matter?
4. How do we train a deployable adaptation student that closes the gap?

The papers below are grouped around those questions.

## Highest-Priority References

### [55] Yu et al. — Universal policy with online system identification

Why it matters:

- very close to our adaptation question
- policy conditioning on hidden factors
- online inference from recent experience
- useful contrast to explicit system-ID style adaptation

Maps to:

- `rma_go2_lab/envs/adaptation/rough_cfg.py`
- `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
- `rma_go2_lab/models/adaptation/actor_critic.py`
- `rma_go2_lab/models/adaptation/adapt_ppo_cfg.py`

Use it for:

- deciding what hidden factors should vary within an episode
- thinking about whether latent inference should target system parameters
  explicitly or only behaviorally useful compressed factors
- framing comparisons against a no-adaptation baseline

### [22] Hwangbo et al. — Learning agile and dynamic motor skills for legged robots

Why it matters:

- strongest direct influence on blind baseline design
- reward/curriculum philosophy is highly relevant
- helps calibrate what a strong locomotion prior should look like before
  adaptation enters the picture

Maps to:

- `rma_go2_lab/envs/blind/rough_cfg.py`
- frozen blind baseline checkpoints
- current deployable student observation interface

Use it for:

- reward balance
- curriculum design
- rough locomotion priors
- sim-to-real locomotion setup discipline

### [31] Lee et al. — Learning quadrupedal locomotion over challenging terrain

Why it matters:

- especially relevant to future terrain expansion
- useful once we deliberately reintroduce discrete obstacle competence
- good reference for terrain-diverse training beyond plain rough random terrain

Maps to:

- future terrain extension beyond the current rough-only adaptation phase
- any later stair / boxes / debris / more complex terrain branch

Use it for:

- terrain curriculum design
- how to mix diverse terrain families without collapsing training
- what stronger geometric generalization could look like

### [49] Tan et al. — Sim-to-real agile locomotion for quadruped robots

Why it matters:

- strong deployment reference
- useful for calibration of actuator realism and sim-to-real hardening
- good contrast to our more adaptation-centric route

Maps to:

- later deployment hardening phase
- actuator realism and control-stack integration

Use it for:

- sim-to-real pitfalls
- actuator/control realism
- what still needs to happen after the adaptation story is solved in simulation

### [40] Peng et al. — Learning agile robotic locomotion skills by imitating animals

Why it matters:

- useful for latent-conditioned locomotion thinking
- useful comparison point for adaptation via latent optimization / imitation
- helps clarify how our route differs from offline latent search methods

Maps to:

- imitation-informed student design
- future decisions about latent supervision versus pure action imitation

Use it for:

- latent-conditioned policy framing
- imitation perspectives
- understanding what slower test-time adaptation baselines look like

## Important Supporting References

### [39] Peng et al. — Sim-to-real transfer with dynamics randomization

Why it matters:

- core baseline philosophy for blind robust policies
- helps calibrate what domain randomization alone can and cannot solve

Maps to:

- `B1/B2/B3`
- `studentNA`

Use it for:

- defining the “robust without adaptation” side of the comparison

### [52] Xie et al. — Dynamics randomization revisited

Why it matters:

- useful sanity-check paper for how far dynamics randomization can go on its own

Maps to:

- blind baseline interpretation
- no-adaptation baseline interpretation

Use it for:

- keeping us honest when evaluating whether adaptation is actually load-bearing

### [58] Yu et al. — Learning fast adaptation with meta strategy optimization

Why it matters:

- good alternate adaptation formulation if the current history-student route
  underperforms

Maps to:

- future adaptation design alternatives

Use it for:

- alternative fast adaptation ideas beyond the first imitation-based route

### [47] Song et al. — Rapidly adaptable legged robots via evolutionary meta-learning

Why it matters:

- useful contrast case for a different adaptation strategy family

Maps to:

- contingency planning if the current adaptation route stalls

Use it for:

- broadening the adaptation design space later, not for immediate copying

## Reward / Task Design References

### [41] Polet and Bertram — Energetically optimal quadruped gaits

Why it matters:

- directly relevant to the “natural constraints / bioenergetics” reasoning in
  original RMA

Maps to:

- blind / teacher reward design philosophy

Use it for:

- deciding whether a reward term supports realistic gait quality or is just
  engineering convenience

### [34] Matthis et al. — Gaze and foot placement in natural terrain

Why it matters:

- not central to the current proprio-only phase
- highly relevant if we later move toward vision/exteroception

Maps to:

- future exteroceptive extension after adaptation

Use it for:

- long-term thinking about stair/downhill/debris competence

## Lower-Priority But Useful Context

### [10] Clavera et al. / [13] Finn et al.

Why they matter:

- strong meta-learning background
- useful if we later want to reinterpret adaptation as fast online inference or
  fast task adaptation

Use them for:

- conceptual expansion, not immediate implementation guidance

### [24] Iscen et al. — Policies modulating trajectory generators

Why it matters:

- mostly useful as a contrast

Use it for:

- clarifying that our current route is not based on hand-designed gait
  generators

### [7] Bongard and Lipson

Why it matters:

- classical system identification background

Use it for:

- perspective on the system-ID side of the literature, not as the main
  implementation guide

## Recommended Reading Order For This Repo

If you only read a few, read these in this order:

1. `[55]`
2. `[22]`
3. `[31]`
4. `[49]`
5. `[40]`
6. `[52]`
7. `[58]`
8. `[47]`

## Repo Mapping Summary

### Blind baselines

Use:

- `[22]`
- `[39]`
- `[49]`
- `[52]`

Relevant files:

- `rma_go2_lab/envs/blind/rough_cfg.py`

### Privileged teacher / expert

Use:

- `[55]`
- `[40]`

Relevant files:

- `rma_go2_lab/envs/teacher/rough_v3_cfg.py`
- `docs/TEACHER_PHASE_SYNTHESIS.md`

### No-adaptation student

Use:

- `[39]`
- `[52]`
- `[55]`

Relevant files:

- `rma_go2_lab/envs/adaptation/rough_cfg.py`

### Adaptation student

Use:

- `[55]`
- `[40]`
- `[58]`
- `[47]`

Relevant files:

- `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
- `rma_go2_lab/models/adaptation/actor_critic.py`
- `rma_go2_lab/models/adaptation/ppo_with_v3_expert.py`
- `docs/ADAPTATION_IMPLEMENTATION_V0.md`

### Future deployment / sim2real hardening

Use:

- `[49]`
- `[22]`
- `[31]`

Relevant future area:

- deployment stack and hardware validation

## One-Line Summary

For the current project, the most useful papers are the ones that help us do
one of three things well:

- build a strong blind locomotion prior
- make the latent/adaptation mechanism truly load-bearing
- harden the eventual deployable student for sim-to-real
