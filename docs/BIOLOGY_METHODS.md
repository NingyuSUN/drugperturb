# Frozen biological descriptions and response evaluation

This protocol uses public annotations fixed before evaluating held-out treatment responses. It describes transcriptional measurements; it does not establish cell-cycle phase, enzyme activity, treatment efficacy, or independent biological validation.

## Public sources and auditable membership

The author cell-cycle lists come from `bin/cc.genes.RDS` at sci-Plex commit `079639c50811dd43a206a779ab2f0199a147c98f`: [download](https://raw.githubusercontent.com/cole-trapnell-lab/sci-plex/079639c50811dd43a206a779ab2f0199a147c98f/bin/cc.genes.RDS). This 524-byte file contains 43 S-phase markers (`s.genes`) and 54 G2/M markers (`g2m.genes`). Its SHA256 is `c7e83853b825d59e77055bcae140815a5b7161f47be1f3fed2e5407efe688640`. The extraction script validates the complete pinned R XDR-v2 named-list structure; it supplies no hand-entered gene symbols.

Three pathways are fixed from [Reactome version 97 GMT](https://download.reactome.org/97/ReactomePathways.gmt.zip): HDACs deacetylate histones (`R-HSA-3214815`, 60 source members), HATs acetylate histones (`R-HSA-3214847`, 108), and Cellular responses to stress (`R-HSA-2262752`, 791). The archive has 298,479 bytes and SHA256 `8c1dbc8578431da5d2d5118262718c60b553a9be3398e93658daa069e4a9afd4`. The broad stress set includes rRNA entries; these are retained in the original membership denominator even if no unique gene mapping exists. The stress set is also available for per-cell descriptive scoring.

Run from the project root:

```bash
.venv/bin/python scripts/freeze_biology.py
.venv/bin/python scripts/freeze_biology.py --offline
```

The script archives exact source bytes, per-source SHA256 and retrieval receipts under ignored `data/raw/biology/`. It writes versioned source definitions, original member lists and mapping records to `configs/biology_v1.json`. It reads only `data/processed/gene_metadata.tsv`, never counts, response rankings, train/test responses or model predictions. Cached sources are verified before use; changed pinned sources cause failure. Missing network responses cause an incomplete status and nonzero exit, not a silently reduced collection. Offline replay verifies that all required cached sources exist and match their provenance receipts.

## Mapping to the actual gene annotation

First match each source symbol exactly to `gene_symbol` in the extracted author gene metadata. A single original gene ID is accepted; multiple IDs are ambiguous and excluded from quantitative scoring. If the symbol is absent, query `https://rest.genenames.org/fetch/symbol/{URL-encoded-symbol}` with JSON acceptance. Accept only one HGNC document whose approved symbol exactly equals the queried symbol. If the approved-symbol query returns zero documents, query the official `fetch/prev_symbol/{symbol}` endpoint and require a unique document whose `prev_symbol` list explicitly contains the original symbol. This is an exact official historical-name mapping, not a guessed synonym. Join its `ensembl_gene_id` to the original Ensembl ID after removing the version suffix. The original versioned ID is preserved. Multiple original IDs for the same stable ID remain ambiguous.

The script archives every actual HGNC JSON response and its URL, retrieval time and hash. It records matched, unmatched, ambiguous and failed-retrieval states. It does not guess aliases or select a mapping by its expression. Both the empty approved-symbol query and the subsequent previous-symbol query are archived when the fallback is used. For example, current histone symbol H2AC1 maps via HGNC to ENSG00000164508, whose previous symbol is HIST1H2AA; direct modern-symbol matching would otherwise omit many histones. No missing marker is replaced with a convenient correlated gene.

Original source-member count, successfully mapped source-member count and distinct mapped gene count are separate quantities. If multiple members resolve to one gene, the gene is scored once. Coverage uses the original source-member denominator and distinct evaluable genes in the numerator, retaining losses and many-to-one mapping effects. Mapping coverage is not selected-output coverage.

## Single-cell and well descriptions

Recompute all scores from raw counts. For each nonzero-library cell, divide each human-gene UMI count by that cell's total human-gene UMI count and multiply by 10,000. For each fixed set, score `log1p(10000 * sum(set UMIs) / sum(all human-gene UMIs))`, using each unique mapped gene once. A zero library yields NA and is counted explicitly. Do not reuse the author's stored Size_Factor, PCA, globally standardized values or scores.

This score adapts the fixed [author marker-sum method](https://raw.githubusercontent.com/cole-trapnell-lab/sci-plex/079639c50811dd43a206a779ab2f0199a147c98f/bin/cell_cycle.R) to fixed library-total normalization. It is not claimed to reproduce the author's numerical scores or Monocle size factors. It is also not a Seurat matched-control-gene score. Marker expression is not a validated cell-phase classification. The broad stress score measures the relative abundance of its covered transcripts and can be dominated by abundant genes; it does not diagnose a particular stress mechanism.

Report per-well cell count, score median and interquartile range, retaining biological replicate, plate, dose pattern and matched vehicle. Keep replicate-level differences rather than pooling all cells across replicates. These outputs describe heterogeneous measurements and sampling; cells do not become independent drug experiments. Do not use these scores to remove drug-induced states or tune models against held-out outcomes.

## Predeclared response evaluation

Select model-output genes using training compounds only. For each pathway, intersect unique mapped genes with this frozen selected-gene list. Quantitative evaluation requires **at least 10 unique selected genes and coverage of at least 25% of the full original source set**. Otherwise label the pathway NA and retain the reason and denominator. Do not change the gene selection to rescue a pathway after looking at responses. The initial configuration leaves selected-gene coverage and evaluability null until the model's training-only gene set exists.

On the same covered genes, compare predicted and measured treatment-minus-matched-vehicle expression changes using mean delta, gene-level RMSE and direction agreement. Truth must remain the measured delta, never a PCA-reconstructed target. Report the covered-gene denominator, and retain zero/undefined directional cases explicitly. Positive and negative changes can cancel in the mean, so a single mean direction is insufficient evidence. Per-compound summaries remain the prediction unit; replicate summaries describe uncertainty and sensitivity.

HDAC and HAT membership overlaps, particularly in histones. Their results are therefore not independent biological confirmations. Pathway members include enzymes, partners and substrates; their transcript levels do not directly report enzyme activity or histone acetylation. No result here establishes pathway activation, treatment mechanism, or efficacy without additional evidence.

## Verified metadata-only freeze, 2026-10-14

The actual extracted gene annotation contains 58,347 rows and has SHA256 `a7f2df38a65846b28195883950aa09fdff7aa795a0f16c34c56e82091309e27c`. The successful run queried HGNC for 58 unique symbols absent from exact annotation-symbol matching; historical-symbol fallbacks generated additional archived requests. No requests remained failed. An offline replay reproduced the configuration byte for byte: SHA256 `92799ec33e9f816c6b354c96d5546a2cd767a48537ec979c407eea36dbfc6451`.

| Set | Original members | Matched members | Unique mapped genes |
|---|---:|---:|---:|
| Author S-phase | 43 | 43 | 43 |
| Author G2/M | 54 | 54 | 54 |
| HDAC | 60 | 59 | 59 |
| HAT | 108 | 107 | 107 |
| Cellular responses to stress | 791 | 784 | 783 |

H2AC18 has a current HGNC Ensembl ID absent from this older gene annotation and remains unmatched in all three pathways. The stress set additionally retains unmapped `18S rRNA`, `28S rRNA`, `5.8S rRNA`, `5S rRNA`, `HSBP2` and `X`. Stress members IL8 and CXCL8 resolve to the same original gene, `ENSG00000169429.10`, and are counted only once in unique-gene scoring. The original author symbols MLF1IP, FAM64A and HN1 were resolved through official exact previous-symbol records. These are mapping results, not evidence that a model's selected genes satisfy pathway coverage or that any biological response has been validated.
