# 6 · Evaluation — the official GlaS metrics & results

← [`05_inference_postprocessing.md`](05_inference_postprocessing.md) · code: `src/dcan/metrics.py`, `scripts/`

We score predicted instance maps against the GT with the three **official GlaS** metrics
(`metrics.py` → `compute_all_metrics`). All are **object-level** (per gland), not per pixel.

## 6.1 The three metrics

- **Detection F1** (`detection_f1`) — did we find the right *objects*? A predicted gland is a true
  positive if it overlaps **≥ 50%** of a GT gland, matched **one-to-one** (`_match_instances`): a
  single merged blob can be the TP of at most one GT gland, so it cannot be rewarded for covering
  several. This one-to-one rule is what correctly penalises the FCN baseline's merges.
- **Object-Dice** (`object_dice`) — overlap quality, area-weighted over matched glands (GlaS Eq. 5).
- **Object-Hausdorff** (`object_hausdorff`) — boundary/shape error: the worst-case contour distance,
  area-weighted (GlaS Eq. 7). Lower is better. This is where the contour branch helps most.

`rank_sum` ranks methods across the three metrics (lower total rank = better overall).

## 6.2 Where the results & figures live

| Artifact | Path |
|----------|------|
| Per-image + aggregate metrics | `runs/<model>/results_test_a.json` (and `*_tta.json`) |
| Comparison tables | `runs/comparison_*.csv` / `.md` (`scripts/compare.py`) |
| Per-metric DCAN-vs-FCN figures | `runs/figures_eval/` ; with TTA `runs/figures_eval_tta/` (`scripts/plot_metrics.py`) |
| Qualitative panels | `runs/figures_partA/` (`scripts/visualize.py`) |

## 6.3 Headline & next steps

DCAN beats the FCN baseline on all three metrics, with the biggest margin on Hausdorff — the
contour branch working as intended. With test-time augmentation (no retraining) DCAN reaches
~0.835 F1 / 0.864 Dice / 50.8 Hausdorff, roughly **halving** the gap to the paper.

Why we still trail the paper, and the recipe-fixes-first roadmap to close the rest (fix the `w_a`
schedule, train to convergence, loss redesign, stronger backbone), are in
[`../ACCURACY_ANALYSIS.md`](../ACCURACY_ANALYSIS.md).

Reproduce: `scripts/evaluate.py` (score predictions), `scripts/compare.py` (tables),
`scripts/plot_metrics.py` (figures), `scripts/tune_postprocess.py` (threshold/TTA sweep).

← Back to [`00_README.md`](00_README.md)
