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
