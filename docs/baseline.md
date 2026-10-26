# Baseline Anchor: Improving Transformer World Models

## Paper

- **Title:** Improving Transformer World Models for Data-Efficient RL
- **Venue:** ICML 2025
- **arXiv:** 2502.01591

## Reported results

The following values are taken from the reference material supplied for this project. They have not been independently reproduced here.

| Method | Craftax-classic reward at 1M environment steps |
|---|---:|
| Transformer World Model (TWM) | 69.66% |
| DreamerV3 | 53.2% |
| Human | 65.0% |

The primary reproduction anchor is **69.66% reward at 1M environment steps on Craftax-classic**.

## Reference implementation

- **Package source:** PyPI
- **Package name:** `improving-transformers-world-model`
- **Installed version:** `0.0.59`
- **Install location:** `/root/miniconda3/envs/craftax-wm/lib/python3.10/site-packages` (environment rebuilt on Python 3.10; see `docs/setup-log.md` section 10)
- **Import check:** `from improving_transformers_world_model import WorldModel` succeeded.
- **Source checkout:** none required. The former `~/craftax-external/improving-twm` plan was replaced by the PyPI package.

The direct PyPI install initially failed while building the environment-related `pufferlib` dependency: `pufferlib` attempted a CUDA extension build with a CUDA 11.5/13.0 mismatch. The TWM wheel and its runtime dependencies were installed instead, without modifying package source. `pufferlib` is not imported by `WorldModel`; it also requires `numpy<2.0`, which conflicts with the verified JAX/Craftax environment using NumPy 2.2.6.

- **Runtime torch requirement:** TWM 0.0.59 instantiates `torch.nn.RMSNorm` (`world_model.py` lines 297, 411, 475), which exists only from PyTorch 2.4. Constructing `WorldModel` on the installed torch 2.1.2+cu121 raises `AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`. Every published TWM release (0.0.1-0.0.59) uses `nn.RMSNorm`, so the model can currently be imported but not constructed in this environment. Recorded in `docs/setup-log.md` section 10.5.

## Starting configuration

The official TWM README example is:

```python
WorldModel(
    image_size=63,
    patch_size=7,
    channels=3,
    transformer=dict(
        dim=512,
        depth=4,
        block_size=81,
    ),
    tokenizer=dict(
        dim=7 * 7 * 3,
        distance_threshold=0.5,
    ),
)
```

Source-signature verification against installed version `0.0.59`:

- `WorldModel.__init__` requires `image_size`, `patch_size`, and `channels`; they have no defaults.
- `dim`, `depth`, and `block_size` are not top-level `WorldModel` arguments. They are passed through the `transformer` dictionary to `BlockCausalTransformer`.
- `BlockCausalTransformer.__init__` requires `dim`, `depth`, and `block_size`; they have no defaults.
- `NearestNeighborTokenizer` requires `dim` and `distance_threshold`; the example uses `dim=49` and `distance_threshold=0.5`.
- With `image_size=63` and `patch_size=7`, one 63 x 63 frame produces 9 x 9 = 81 patch tokens, consistent with the example `block_size=81`.

The project starting configuration therefore remains `image_size=63`, `patch_size=7`, `channels=3`, `dim=512`, `depth=4`, and `block_size=81`.
