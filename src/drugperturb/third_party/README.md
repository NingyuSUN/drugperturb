# Vendored third-party code

| File | Upstream | Commit | SHA256 | License |
|---|---|---|---|---|
| `_chemcpa_model_upstream.py` | [theislab/chemCPA](https://github.com/theislab/chemCPA) `chemCPA/model.py` | `43e830eb0958c54e4aa64442c17ec0fed19b3f15` | `481fe1f447f1318e0446764d4dcfc8b7e6d6c21efe21cd156fceb8e693f75ae6` | MIT, see `LICENSE.chemCPA` |

The file is unmodified. `drugperturb.v3.chemcpa` imports `ComPert` from it and
re-implements the published training step (`chemCPA/lightning_module.py`) without
Lightning, Hydra or Weights & Biases. `tests/test_v3_vendored.py` checks the hash.
