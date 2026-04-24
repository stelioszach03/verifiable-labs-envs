# sparse-fourier-recovery-tools

Tool-use sparse Fourier recovery (fft, ifft, ista, check_residual).

Verifiable Labs Scientific-RL environment. Published as a thin wrapper around the monorepo at https://github.com/stelioszach03/verifiable-labs-envs — the wrapper pulls the monorepo as a Git dependency so the full source of truth (rewards, forward operators, LLM adapter, conformal calibration) stays in one place.

## Install

```bash
prime env install verifiable-labs/sparse-fourier-recovery-tools
```

## Use

```python
from verifiers import load_environment    # or Prime SDK equivalent
env = load_environment("sparse-fourier-recovery-tools")
out = env.run_baseline(seed=0)
print(out["reward"])
```

See the monorepo README + docs for the reward spec, contamination story, and benchmark data.
