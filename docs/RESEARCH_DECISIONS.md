# Research decisions and evidence boundaries

| Decision | Rationale | Evidence / trade-off |
|---|---|---|
| Predict compound-level treated-minus-vehicle change | Measures response instead of recovering baseline expression | Each compound is a prediction unit; cells and shared controls do not create independent drugs |
| Hold out chemical structure groups | Tests compounds outside training parent/scaffold identities | Related chemistry can remain similar; all maximum training similarities are reported |
| Fit genes, standardization and PCA on training only | Prevents evaluation responses from changing the target representation | Two thousand variable genes omit much of the prespecified biology; variance alone does not establish signal |
| Include zero, mean and nearest-neighbor references | Establishes whether learning chemical structure adds value | The mean reference is strong and the GNN advantage remains inconclusive |
| Limit neural capacity and tuning | 119 training compounds cannot justify an unconstrained deep-model search | Four candidates and three seeds per neural family; no pretrained model or GPU-scale claim |
| Choose checkpoints by measured gene-space error | PCA-coordinate loss need not rank checkpoints by the actual endpoint | Optimization remains in PC space, but selection and scoring invert predictions and retain actual truth |
| Freeze predictions before final scoring | Keeps final-test outcomes out of candidate selection | Reproducibility replays reuse the opened test set; they are not new validation studies |
| Prespecify biological sets and coverage gates | Prevents selecting attractive mechanisms after observing results | All five sets fail the output-coverage gate; no pathway-recovery claim is made |
| Retain alternative-control and replicate diagnostics | Tests how experimental definitions affect interpretation | Rankings change with the vehicle definition; shared control/plate effects remain a limitation |

## Attribution and responsibility

The source measurements and original preprocessing come from the cited sci-Plex study. This repository contains a new implementation of the stated benchmark. Automated tests and code reviews address implementation behavior. Biological claims are restricted to the measured experiment and stated evaluation design.

