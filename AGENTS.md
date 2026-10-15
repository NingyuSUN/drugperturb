# Research correctness and contribution rules

- Use public biological data and annotations with source URLs, versions and hashes. Synthetic fixtures establish software behavior only.
- Keep final-test responses out of gene selection, transforms, model/hyperparameter choice, split selection and checkpoint selection. Freeze predictions before final evaluation; a new scientific question requires a new protocol and untouched evaluation data.
- Treat compounds as prediction units, cells as measurement units and biological replicates separately. Report denominators, exclusions, missing values and failed checks.
- Preserve original structures and source bytes. Reject undefined atoms and unresolved active mixtures; group canonical duplicates and scaffolds before splitting.
- Fit gene selection, scaling and PCA on training compounds only. Score measured deltas, never PCA-reconstructed truth. Preserve zero-change and strong simple baselines.
- Keep raw/large data, environments and model binaries out of Git. Record runtime versions, actual commands, input/output hashes and the limits of each execution claim.
- Preserve established results and reproducibility failures. Tests, simulations, workflow replay and code review do not constitute independent biological or clinical validation.
