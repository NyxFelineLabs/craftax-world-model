# Experiment Plan

## Research question

How can the sample efficiency of a Transformer World Model be improved in Craftax's open-ended game environments within at most 1M environment steps?

## Baseline

- ICML 2025 Transformer World Model (TWM)
- Craftax-classic: 69.66% reward at 1M environment steps
- Reference package: `improving-transformers-world-model==0.0.59` from PyPI

## Candidate improvement directions

The following are candidate directions only; none is implemented in the current setup.

1. **Tokenizer structural prior:** incorporate spatial or object-aware structure into the image tokenizer rather than relying only on nearest-neighbor patch codes.
2. **Adaptive block teacher-forcing schedule:** vary the block-level teacher-forcing horizon during training instead of keeping a fixed schedule.
3. **Dyna warmup annealing and PPO coupling:** anneal imagined-rollout warmup and adjust the coupling between world-model imagination and PPO updates.

## Evaluation protocol

- Environment: Craftax-classic
- Final metric: reward after 1M environment steps
- Curves: reward versus environment steps
- Replication: at least 3 seeds, reported as mean plus/minus standard deviation
- Compare against the TWM baseline under the same environment-step budget and evaluation protocol

## Compute budget

Run a 100k-step pilot on an RTX 3090 first. Measure step time and peak GPU memory, then extrapolate the cost of a 1M-step run and the full multi-seed experiment. Do not start a run longer than 10k steps during environment setup.
