# Setup Log

Date: 2026-10-26  
Host: WSL2 Ubuntu-22.04 on Windows  
Project: `/mnt/e/Project/NyxFelineLabs/craftax-world-model`

This log records command-verified results. Literature metrics are user-supplied reference values, not reproduced here.

## 0. Preconditions

| Check | Command | Exit | Key result |
|---|---|---:|---|
| WSL kernel | `uname -a` | 0 | Linux 6.18.33.2-microsoft-standard-WSL2 x86_64 |
| Distro | `printenv WSL_DISTRO_NAME` | 0 | `Ubuntu-22.04` |
| Project path | `ls -la /mnt/e/Project/NyxFelineLabs/craftax-world-model` | 0 | Existing Windows-side directory, initially empty |
| Working directory | `cd ... && pwd` | 0 | `/mnt/e/Project/NyxFelineLabs/craftax-world-model` |
| GitHub CLI | `gh auth status` | 127 initially | `gh` was not installed in WSL |

GitHub CLI `2.103.2` was installed from the official release `.deb`. WSL authorization reused the already authenticated local Windows GitHub account `SEMHAQ`; the token was never printed. `gh api user/orgs` showed `NyxFelineLabs` with exit 0.

## 1. Repository skeleton

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Initialize Git | `git init` (after checking `.git`) | 0 | Repository initialized |
| Create directories | `mkdir -p src scripts configs docs` | 0 | Directories preserved if already present |
| Create ignore file | append-if-missing into `.gitignore` | 0 | Python, environment, data, log, external, and egg-info patterns added |
| Create README | create only if absent | 0 | Title, scope, anchor, and storage layout recorded |

## 2. GitHub repository

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Rename branch | `git branch -M main` | 0 | Branch `main` |
| First create/push attempt | `gh repo create ... --source=. --remote=origin --push` | 1 | Expected failure: local repository had no commit yet |
| Initial commit | `git commit -m "chore: initialize craftax world model repository"` | 0 | Commit `4c74aa6` |
| Create and push | same `gh repo create` command after initial commit | 0 | Public repository created and `main` pushed |

Remote URL: `https://github.com/NyxFelineLabs/craftax-world-model.git`

## 3. Conda and GPU JAX

| Check | Command | Exit | Key result |
|---|---|---:|---|
| Create environment | `conda create -n craftax-wm python=3.11 -y` | 0 | `/root/miniconda3/envs/craftax-wm` |
| Python | `python --version` | 0 | Python 3.11.17 |
| pip | `pip --version` | 0 | pip 26.2.2 |
| JAX install | `pip install -U "jax[cuda12]"` with proxy variables unset | 0 | JAX 0.10.2 and CUDA 12 wheels installed |
| GPU visibility | `python -c "import jax; print(jax.__version__); print(jax.devices())"` | 0 | `0.10.2`; `[CudaDevice(id=0)]` |

## 4. Craftax and baseline dependencies

### Craftax

| Check | Command | Exit | Key result |
|---|---|---:|---|
| Install | `pip install craftax` with proxy variables unset | 0 | Craftax 1.6.1 installed |
| Import | `python -c "import craftax; print('craftax OK')"` | 0 | `craftax OK` |

### Craftax_Baselines installation-method correction

This was an installation-method correction, not a source-code modification.

The initial command `pip install git+https://github.com/MichaelTMatthews/Craftax_Baselines.git` exited 1 because the upstream project is a script-and-requirements repository, not an installable Python package:

```text
ERROR: ... does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
```

The official method was then used:

```bash
git clone https://github.com/MichaelTMatthews/Craftax_Baselines.git ~/craftax-external/Craftax_Baselines
cd ~/craftax-external/Craftax_Baselines
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

| Check | Exit | Key result |
|---|---:|---|
| Clone | 0 | Commit `7ce36fa05b84a2c9e758012f1e6da402e1e3a891` |
| Requirements install | 0 | All requirements resolved |
| `pre-commit install` | 0 | Hook installed |
| `pip check` | 0 | No broken requirements at that point |

JAX remained 0.10.2; it was not upgraded or downgraded. `orbax-checkpoint` was set to the repository-pinned `0.5.0`. Key versions: JAX 0.10.2, flax 0.12.8, gymnax 1.0.1, optax 0.2.8, orbax-checkpoint 0.5.0.

## 5. Craftax smoke test

The required 100-step `Craftax-Symbolic-v1` script ran with exit 0 and printed:

```text
step OK, reward: 0.0 obs shape: (8268,)
```

At execution time `nvidia-smi` reported an RTX 3090 with 24576 MiB total and 12035 MiB already used. JAX emitted transient allocation-retry messages while selecting a memory fraction, then initialized successfully on the GPU.

## 6. TWM installation-method correction

This was an installation-method correction, not a source-code modification. TWM was sourced from PyPI rather than cloned from the former `lucidrains` checkout path.

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Direct install attempt | `pip install improving-transformers-world-model` | 1 | `pufferlib` wheel build failed: detected CUDA 11.5 versus PyTorch CUDA 13.0 |
| CPU PyTorch | `pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu` | 0 | `2.14.1+cpu`; no CUDA dependency |
| TWM runtime dependencies | install TWM dependencies except `pufferlib` | 0 | Core PyTorch dependencies installed |
| TWM wheel | `pip install --no-deps improving-transformers-world-model` | 0 | Version 0.0.59 installed from PyPI |
| Import | `python -c "from improving_transformers_world_model import WorldModel; print('TWM import OK')"` | 0 | `TWM import OK` |

`pufferlib` was not installed: its requirements specify `numpy<2.0`, which conflicts with the working JAX/Craftax stack using NumPy 2.4.6, and `WorldModel` does not import it. `pip check` therefore reports only that `pufferlib` is absent. This does not affect the requested `WorldModel` import.

## 7. WSL-native data and external links

| Target | Readlink result |
|---|---|
| `logs` | `/root/craftax-workspace/logs` |
| `data` | `/root/craftax-workspace/data` |
| `external` | `/root/craftax-external` |

The original slash-suffixed ignore rules did not match Git symlinks. Exact non-slash entries `data`, `logs`, and `external` were appended to `.gitignore`; `git check-ignore -v` then confirmed all three. `git status --short` shows no symlink entries.

## 8. Documentation

Created:

- `docs/baseline.md`
- `docs/experiment-plan.md`
- `docs/setup-log.md`

## 9. Commit and push

Primary commit:

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Stage | `git add -A` | 0 | `.gitignore` and three docs staged; symlinks remained ignored |
| Status | `git status --short` | 0 | `M .gitignore`, `A docs/baseline.md`, `A docs/experiment-plan.md`, `A docs/setup-log.md` |
| Commit | `git commit -m "feat: craftax env setup, baseline anchor, experiment plan draft"` | 0 | Commit `aebb0e4`; 4 files changed, 226 insertions |
| First push attempt | `git push` | interrupted | Stalled while WSL Git waited for usable HTTPS credentials |
| Configure helper | `gh auth setup-git` | 0 | Reused the existing authenticated GitHub CLI session |
| Push retry | `GIT_TERMINAL_PROMPT=0 git push` | 0 | `4c74aa6..aebb0e4 main -> main` |

This setup-log update is committed and pushed separately from the primary environment/documentation commit.

## 10. Environment rebuild: Python 3.10, torch 2.1.2+cu121 from local wheel

Reason for the rebuild: the user-supplied wheel `/mnt/e/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl` is a **cp310** build, so it cannot be installed into the previous Python 3.11.17 environment. The environment was therefore recreated on Python 3.10 and every dependency was reinstalled. No training was started.

### 10.1 Rebuild steps

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Remove old environment | `conda remove -n craftax-wm --all -y` | 0 | Previous Python 3.11 environment removed |
| Create environment | `conda create -n craftax-wm python=3.10 -y` | 0 | With proxy variables unset; an earlier attempt without them hit the stale `192.168.16.1:7890` proxy |
| Python | `python --version` | 0 | Python 3.10.22 |
| Local wheel present | `ls -l /mnt/e/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl` | 0 | 2200673027 bytes |
| torch install | `pip install "/mnt/e/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl"` | 0 | torch 2.1.2+cu121 |
| torch CUDA check | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | 0 | `2.1.2+cu121 12.1 True` |
| CUDA matmul | 1024x1024 `x @ x` on `cuda` | 0 | `matmul OK` |
| JAX (GPU) | `pip install -U "jax[cuda12]"` | 0 | JAX 0.6.2; `jax.devices()` = `[CudaDevice(id=0)]` |
| Craftax | `pip install craftax` | 0 | Craftax 1.6.1; `craftax OK` |
| Craftax_Baselines | `pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` | 0 | Repo requirements applied; the repo pins `orbax-checkpoint` 0.5.0; warning `jax 0.6.2 does not provide the extra 'cuda12-pip'` |

Key versions after the reinstall: JAX 0.6.2, flax 0.10.7, optax 0.2.8, gymnax 1.0.1, distrax 0.1.5, chex 0.1.90, orbax-checkpoint 0.5.0, NumPy 2.2.6.

### 10.2 TWM reinstall

Installation method: PyPI package `improving-transformers-world-model` (no source checkout, no source modification).

A plain dependency-resolving install cannot be used in this environment: TWM 0.0.59 declares `torch>=2.2`, and every PyTorch dependency of TWM declares an even newer torch (`hl-gauss-pytorch` >=2.3, `rotary-embedding-torch` >=2.4, `vector-quantize-pytorch` >=2.4, `x-mlps-pytorch` >=2.4, `hyper-connections` >=2.5, `torch-einops-utils` >=2.5, `x-ppo` >=2.5, `mean-conc-beta` >=2.5, `accelerated-scan` >=2.9). A constrained resolve against the local torch failed:

```text
ERROR: ResolutionImpossible ... hl-gauss-pytorch 0.0.1 depends on torch>=2.2 ...
       The user requested (constraint) torch==2.1.2+cu121
```

The dependency stack was therefore installed with `--no-deps`, which preserves the locally installed torch:

| Action | Command | Exit | Key result |
|---|---|---:|---|
| Constrained resolve | `pip install -c /tmp/twm-constraints.txt <TWM deps>` | 1 | ResolutionImpossible: the TWM dependency stack requires torch>=2.3 |
| Dependency stack (torch-preserving) | `pip install --no-deps <21 packages>` | 0 | einx 0.4.3, einops 0.8.2, hl-gauss-pytorch 0.3.0, hyper-connections 0.4.11, jaxtyping 0.3.7, rotary-embedding-torch 0.9.1, vector-quantize-pytorch 1.32.0, torch-einops-utils 0.1.45, x-mlps-pytorch 0.6.9, x-ppo 0.0.26, mean-conc-beta 0.2.9, assoc-scan 0.0.6, accelerated-scan 0.3.1, adam-atan2-pytorch 0.4.1, ema-pytorch 0.8.3, evolutionary-policy-optimization 0.5.0, loguru 0.7.3, frozendict 2.4.7, ninja 1.13.2, tqdm 4.70.1, wadler-lindig 0.1.8 |
| TWM package | `pip install --no-deps improving-transformers-world-model` | 0 | 0.0.59 from PyPI; `torch` remained `2.1.2+cu121` |
| Import | `python -c "from improving_transformers_world_model import WorldModel; print('TWM import OK')"` | 0 | `TWM import OK` |
| `pip check` | `pip check` | 1 | `pufferlib` and `accelerate` not installed; ten `requires torch>=2.x, but you have torch 2.1.2+cu121` reports |

`pufferlib` was not installed: `pip install --dry-run improving-transformers-world-model` shows pip would build the `pufferlib` 3.0.0 sdist (60.7 MB) from source, and its requirement `numpy<2.0` conflicts with the JAX/Craftax stack. `WorldModel` does not import `pufferlib`.

### 10.3 JAX and torch coexistence

| Command | Exit | Result |
|---|---:|---|
| `import torch, jax`; `torch.randn(64,64,device="cuda") @ itself`; `jax.devices()` | 0 | `torch cuda: True`, `jax devices: [CudaDevice(id=0)]`, `torch after jax import: OK` |

No CUDA initialization conflict occurs. One caveat is recorded: NumPy 2.2.6 is installed while the cp310 torch 2.1.2 wheel was built against NumPy 1.x, so importing torch emits `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`. torch works on CUDA, but the torch-to-NumPy conversion bridge is disabled in this combination. NumPy was deliberately **not** downgraded to `numpy<2`, because the JAX 0.6.2 stack in this environment (jaxlib/ml_dtypes, `dm-tree`, `gymnasium`) declares `numpy>=2.x`.

### 10.4 Craftax smoke test (100 steps)

| Command | Exit | Result |
|---|---:|---|
| 100-step `Craftax-Symbolic-v1` rollout (the script supplied for this step) | 0 | `craftax step OK, reward 0.0 obs shape (8268,)` |

### 10.5 TWM GPU forward and backward: BLOCKED

The supplied script cannot run against the installed TWM API. It first fails at construction because `dim`, `depth` and `block_size` are not top-level `WorldModel` arguments (they belong to the nested `transformer` dict; see `docs/baseline.md`):

```text
TypeError: WorldModel.__init__() got an unexpected keyword argument 'dim'
```

Re-running with the README-faithful construction (`transformer=dict(dim=512, depth=4, block_size=81)`, `tokenizer=dict(dim=7 * 7 * 3, distance_threshold=0.5)`) fails against the installed torch:

```text
File ".../improving_transformers_world_model/world_model.py", line 297, in __init__
    self.norm = nn.RMSNorm(dim)
AttributeError: module 'torch.nn' has no attribute 'RMSNorm'
```

`torch.nn.RMSNorm` was added in PyTorch 2.4. TWM 0.0.59 declares `torch>=2.2` but in practice requires torch>=2.4 (`nn.RMSNorm` at `world_model.py` lines 297, 411 and 475). Its optional flex-attention path imports `torch.nn.attention.flex_attention` (2.5+) and degrades to `None` when unavailable. Inspection of every published TWM release (0.0.1 through 0.0.59) shows that all of them use `nn.RMSNorm`, so no PyPI release of TWM is compatible with torch 2.1.2.

Consequently no loss value and no peak-VRAM number can be reported for this step: the model cannot be constructed on torch 2.1.2+cu121, so no forward/backward pass is possible. Neither TWM nor torch was modified, per the stated constraints.

Not executed, recorded for the follow-up decision: torch>=2.4 with CUDA 12.1 keeps the same CUDA major/minor that this wheel uses, for example `pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121` (a cp310 wheel exists). This would replace the user-supplied local 2.1.2 wheel and therefore needs explicit approval.

GPU state during the tests: RTX 3090, 24576 MiB total, 754 MiB in use before the runs. No training was started.

### 10.6 Status of this rebuild

Steps 1-5 are verified: environment rebuild, local-wheel torch, GPU JAX, Craftax, Craftax_Baselines, TWM install and import, JAX/torch coexistence, and the Craftax smoke test. Step 6 is blocked at the TWM forward/backward test because the requested torch 2.1.2+cu121 lacks `torch.nn.RMSNorm`, which TWM 0.0.59 requires.
