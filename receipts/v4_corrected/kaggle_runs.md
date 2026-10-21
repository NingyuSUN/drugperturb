# Corrected OP3 analysis: Kaggle runs

The kernel and the code dataset are private, under the account `ningyusunny`. The code dataset `drugperturb-v4c-code` is new and separate from `drugperturb-v4-code`, so every v4 kernel keeps its code version. The kernel was launched on 2026-10-21 after the user approved it.

| Kernel | Version | Hardware | Code archive SHA256 (dataset version) | Repository commit | Inputs | Outcome |
|---|---|---|---|---|---|---|
| dp-v4c-fit-op3 | 1 | T4 ×2 | bb9eb191… (v1) | a9950ee | dp-v4-op3-prepare | complete; 12 fits, 1 h 52 min; no non-finite evaluation |

## Commands

The commands ran in WSL from the `dp-op3c` worktree, with Kaggle CLI 2.2.4. The archive holds the 419 files that git tracks at a9950ee; no data, models or outputs are included.

```
KAGGLE_BUILD_ROOT=$HOME/kb python kaggle/push.py code --dataset drugperturb-v4c-code
KAGGLE_BUILD_ROOT=$HOME/kb python kaggle/push.py kernel v4c-fit-op3
```

The first `datasets create`, run from the default build directory under the worktree, failed before any upload. The Kaggle CLI rejected its upload cache name with "File name too long". The short build root `$HOME/kb` avoids this; the v4 pushes used the same root.

## Run

- **Timing.** The kernel was pushed at 01:06 UTC and wrote its receipt at 02:58 UTC. It was complete by 03:02 UTC.
- **Code and tests.** The receipt ([k3_v4c_op3.json](op3/k3_v4c_op3.json)) records the code-archive SHA256 bb9eb191…, which equals the archive that was pushed. The in-kernel tests (`tests/test_v4_cellmodels.py`, `test_v4_core.py`, `test_v3_core.py` and `test_descriptor_scaling.py`) passed: 31 tests.
- **Fits.** All 12 fits stopped by patience, between 10 and 24 minutes each and 219 fit-minutes in total. None had a non-finite validation evaluation. Every fit record has `descriptor_columns: 168`, `analysis: v4_corrected_posthoc`, `test_outcomes_loaded: false` and `engineering_smoke: false`.
- **Committed outputs.** The fit records and histories are in [op3/fits](op3/fits), and the seals and the validation-only selection in [op3/sealed](op3/sealed).
- **Not committed.** The prediction archives and logs stay out of Git, under the repository's rule on large outputs. The local copy downloaded with `kaggle kernels output ningyusunny/dp-v4c-fit-op3` is at `~/kout/v4cfit` in WSL; the seals hash the archives.
