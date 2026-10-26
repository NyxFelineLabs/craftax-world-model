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

## Implementation provenance

- **Package source:** PyPI
- **Package name:** `improving-transformers-world-model`
- **Installed version:** `0.0.59`
- **Install location:** `/root/miniconda3/envs/craftax-wm/lib/python3.10/site-packages` (environment rebuilt on Python 3.10; see `docs/setup-log.md` section 10)
- **Import check:** `from improving_transformers_world_model import WorldModel` succeeded.
- **Source checkout:** none required. The former `~/craftax-external/improving-twm` plan was replaced by the PyPI package.

The direct PyPI install initially failed while building the environment-related `pufferlib` dependency: `pufferlib` attempted a CUDA extension build with a CUDA 11.5/13.0 mismatch. The TWM wheel and its runtime dependencies were installed instead, without modifying package source. `pufferlib` is not imported by `WorldModel`; it also requires `numpy<2.0`, which conflicts with the verified JAX/Craftax environment using NumPy 2.2.6.

- **Runtime torch requirement:** TWM 0.0.59 instantiates `torch.nn.RMSNorm` (`world_model.py` lines 297, 411, 475), which exists only from PyTorch 2.4, and every published TWM release (0.0.1-0.0.59) uses `nn.RMSNorm`. torch was therefore upgraded, with explicit approval, from 2.1.2+cu121 to **2.4.1+cu121**, after which the GPU forward and backward pass runs (see `docs/setup-log.md` sections 10.5 and 11).

- **Current environment (2026-10-27):** Python 3.10.22; torch `2.4.1+cu121` (CUDA 12.1, `torch.nn.RMSNorm` available, torch NumPy bridge operational with NumPy 2.2.6); JAX `0.6.2` with `jax.devices() == [CudaDevice(id=0)]`; `nvidia-cudnn-cu12` `9.27.0.42` (required by `jax-cuda12-plugin`, while torch pins 9.1.0.70 - see `docs/setup-log.md` section 11).
- **WorldModel construction (official README, confirmed with `inspect.signature` on 0.0.59):**

```text
(self, image_size, patch_size, channels, num_actions=0, reward_min_value=0.0, reward_max_value=1.0,
 reward_num_bins=0.0, transformer_use_token_embed=True,
 tokenizer: 'NearestNeighborTokenizer | Module | dict' = {},
 transformer: 'BlockCausalTransformer | dict' = {},
 hl_gauss_loss_kwargs: 'dict' = {'sigma_to_bin_ratio': 1.0},
 is_terminal_loss_weight=1.0, reward_loss_weight=1.0, dropout_embed=0.1)
```

- **Verified GPU step:** the README example input `state = torch.randn(2, 3, 20, 63, 63, device='cuda')` (batch, channels, time, height, width) produced `loss = 11.682136535644531`, with `loss.backward()` and an `AdamW` step on torch 2.4.1+cu121 (peak allocated 7541.18 MB).

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
