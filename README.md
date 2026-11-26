# Aquaculture Pond Detection from Satellite Imagery

Binary classification of 10 m × 10 m ground patches as **aquaculture pond** or **other**, from Sentinel-1 (SAR) and Sentinel-2 (optical) monthly composites. Built for the FAO × AI for Good (ITU) challenge on Zindi.

The main difficulty is not the model but **distribution shift**: the model is trained on one set of conditions and scored on another. Most of this repository documents how that shift was diagnosed, which fixes were tested, and which ones failed.

> **Status:** best public leaderboard score **0.829** (AUC 0.851, F1 0.814). This project is a documented investigation of a hard generalization problem, not a top-of-leaderboard solution.

---

## Problem

| | |
|---|---|
| Task | Pond (1) vs other (0) for each location |
| Metric | 0.6 × F1 + 0.4 × ROC-AUC |
| Constraint | Decision threshold fixed at 0.5 (probabilities are scored separately for AUC) |
| Data | 12 monthly composites per location: Sentinel-1 (VH, VV) and 10 Sentinel-2 bands |
| Rules | Only supplied data allowed; lon/lat are not to be used as features |

**Train:** 1,821 locations, 12 full months, about 40% ponds.
**Test:** 1,030 locations, each with only a 4–6 month consecutive window (the rest masked as `-9999`), with optical gaps from cloud cover. The pond rate of the test set is about 54%, measured with an "all ones" leaderboard probe (score 0.6226 corresponds to F1 0.704 at AUC 0.5).

---

## Approach

### Features (119)
Each location is reduced to 17 signals: 12 raw bands plus 5 indices (`NDWI`, `MNDWI`, `NDVI`, `NDMI`, `VH−VV`). Each signal gets 7 NaN-aware summaries: mean, std, min, max, range, median, and number of valid months. Summaries were chosen over the raw 12-month sequence because they do not depend on which months a location happens to cover (raw sequences scored 0.015 worse in CV).

`-9999` must be converted to `NaN` first; otherwise the model treats the mask as a value and predicts almost everything as a pond.

### Model
50/50 ensemble of `HistGradientBoostingClassifier` (native NaN handling, shallow trees, L2) and a median-imputed, standardized `LogisticRegression`. No class weights, no calibration, threshold 0.5.

### Augmentation
The training set is complemented with corrupted copies that imitate the test structure: a random consecutive 4–6 month window, radar preserved, about 30% of optical observations dropped as clouds. This gave roughly +0.09 in CV.

### Validation
`KMeans(10)` on features, then `GroupKFold(5)` so whole clusters are held out. Validation folds are corrupted the same way as the test, and F1 is estimated in a resampled "54% ponds" world. Measured noise is about ±0.01.

---

## Key findings

1. **Train and test are trivially separable.** An adversarial classifier reaches AUC 1.0 (0.998 even after removing the valid-month counts), so the shift is in the spectral and radar values themselves, not only in the masking pattern.
2. **No validation built from the training data predicts the leaderboard.** Grouped CV scores 0.94–0.97 while the leaderboard sits near 0.82. Splitting the training data by geography (train on one half, predict the other) also scores about 0.97, so the training data is internally homogeneous and the shift only appears between train and test.
3. **Window-aware training helps on what CV can see.** With the validation set corrupted to look like the test, training on 4–6 month windows scored 0.899 against 0.789 for training on full 12 months. Training windows alone reduced train/test separability only from 1.00 to 0.986.
4. **Relative features transfer better than absolute ones.** Measured as train/test separability (lower means more similar):

   | Feature family | Features | Separability |
   |---|---|---|
   | Absolute levels (baseline 119) | 119 | 0.989 |
   | Intra-pixel amplitudes (p75−p25, p90−p10) | 36 | 0.896 |
   | Radar percentiles + SDWI | 14 | 0.849 |
   | Water-occurrence frequencies | 4 | 0.748 |

   Leaderboard result for the invariant-feature model: `TODO: fill in after submission`.
5. **Self-training does not move the ranking.** Pseudo-labeling converged to the same ranking as the baseline (Spearman 0.96): the model's confidence rests on features that do not transfer.

---

## Experiment log

Public leaderboard scores for the main submissions (noise about ±0.01, so differences below 0.01 are not meaningful):

| Experiment | Score | Note |
|---|---|---|
| Baseline (119 features, ensemble) | 0.819 | AUC 0.856, F1 0.795 |
| Class-weighted (fewer false positives) | 0.798 | Worse |
| Calibration (isotonic / sigmoid) | worse | Not submitted after CV |
| 3-model ensemble | 0.813 | No gain over 2 models |
| Temporal interpolation of gaps | 0.826 | Within noise |
| Enriched radar features | 0.813 | No gain |
| Per-point z-score normalization | 0.824 | Separability stays 1.0 |
| Stable ("robust") feature subset | **0.829** | Best, AUC unchanged |
| "All ones" probe | 0.6226 | Used to measure the test pond rate |

The full list of 20 hypotheses, with verdicts, is in [`docs/JOURNAL_EXPERIENCES.md`](docs/JOURNAL_EXPERIENCES.md).

---

## Limitations

- The score gap to the top of the leaderboard (about 0.90+) was not closed. The shift between train and test could not be reproduced inside the training data, so improvements could only be judged with a limited number of leaderboard submissions.
- The public leaderboard covers about 30% of the test set; the private score may differ.
- The cause of the shift (different region, different period, or both) was not identified with certainty.
- For diagnostics only, coordinates of the training points were recovered by matching them to an earlier release of the dataset. They were used to analyze geographic structure and **never as model features**.

---

## Reproducibility

All notebooks are seeded (`SEED = 42`) and each one regenerates its own submission file when run end to end.

```
.
├── data/                      # not included (competition terms); see below
├── notebooks/                 # stage-by-stage experiments, ending with Stage18_TestFenetres.ipynb
├── docs/
│   └── JOURNAL_EXPERIENCES.md # all hypotheses and verdicts
├── requirements.txt
└── README.md
```

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, scipy, shap, jupyter
# Download Train.csv, Test.csv and features_test.csv from the Zindi challenge page into ./data
jupyter nbconvert --to notebook --execute notebooks/Stage18_TestFenetres.ipynb
```

The data is not redistributed here; download it from the competition page after accepting its terms.

---

## What this project taught me

- A clean validation protects against fooling yourself, but it cannot fix a shift that is absent from the training data.
- Probing the leaderboard for a fact (the test pond rate) is useful; probing it to gain a few thousandths of score is overfitting.
- When several modifications of one design all plateau at the same level, the useful move is to leave that design: diagnose the shift, then look outside it.
