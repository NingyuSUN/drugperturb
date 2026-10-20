# v4 Kaggle runs

All kernels are private, under the account `ningyusunny`, and use the private code dataset `drugperturb-v4-code`. Each kernel mounts the latest dataset version that exists at run time.

| Kernel | Version | Hardware | Code archive SHA256 (dataset version) | Repository commit | Inputs | Outcome |
|---|---|---|---|---|---|---|
| dp-v4-op3-extract | 1 | CPU | 8eb216d8… (v1) | 046d97c | GEO GSE279945 h5ad | complete |
| dp-v4-fit-sciplex | 1 | T4 ×2 | 8eb216d8… (v1) | 046d97c | dp-v3-prepare | complete; 16 fits, 2 h 6 min |
| dp-v4-op3-prepare | 2 | CPU | c9b6fa41… (v2) | 1d5f840 | dp-v4-op3-extract | complete |
| dp-v4-fit-op3 | 1 | T4 ×2 | c9b6fa41… (v2) | 1d5f840 | dp-v4-op3-prepare | error: 2 of 12 fits overflowed (amendment A2); outputs archived unread |
| dp-v4-anchored | 1 | CPU | 281266f3… (v3) | 0f357ee | dp-v4-op3-prepare, dp-v3-prepare | complete |
| dp-v4-fit-op3 | 2 | T4 ×2 | 8d405d3b… (v4) | aed9701 | dp-v4-op3-prepare | running |

Version 1 of dp-v4-op3-prepare was pushed a few seconds before version 2. A shell retry pushed version 2, which differs only in its push time. Downstream kernels read the latest version's output.

Code changes between archives 8eb216d8 and c9b6fa41:

- the non-collapse flags were renamed `pass_*`, and the diagnostics report accepts either spelling;
- receipts now record the code-archive SHA256.

No model, loss, data or evaluation code changed between the two archives.
