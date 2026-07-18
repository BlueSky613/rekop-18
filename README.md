# Solution A Aggressive Variant (uid 232) - Final v2 - 2026-07-18

## Strategy

Live-shape priority. Current live queries are usually about 90 batches x 100 hands.
Solution A emphasizes merged100 batches with weight 1.3 and recency half-life 3.0.

## Configuration

| Item | Value |
|---|---|
| Data | 2026-07-06 through 2026-07-18, mirrored payload view, original 1906 plus merged100 1560 for 3466 rows |
| Main ensemble | LGBM x5 with leaves63, RF500, ExtraTrees500, HistGB500, LogReg |
| Diversity members | Two LGBM models trained with autocorr, rand, and state axes excluded, 291 features, weight 0.7 |
| Weights | recency half-life 3.0 x merged weight 1.3 |
| Safety cap | top-K 10 percent, minimum 2, above 0.5 only |
| Defenses | per-chunk try/except, NaN sanitation, deterministic fallback |

## Verification

```text
Forward test: train through 2026-07-17, evaluate on 2026-07-18
Original34: AP=0.9854 recall=0.9189 -> comp=0.9706
Merged100:  AP=0.9958 recall=0.9667 -> comp=0.9885
Speed: 90 batches x 100 hands = 3.1 seconds
Cap: exactly 9/90 flagged, adversarial input checks passed, joblib 49.4 MB
```

## Key History

| Change | Reason |
|---|---|
| Inference speed fix | Avoids recomputing features 343 times per chunk. |
| Mixed-shape training | Original-only training was weaker on live-shape proxy data. |
| Per-model feature subsets | Adds two diversity models for humanized bot patterns. |

## Deploy

```bash
pip install -r requirements.txt
python train.py --all
python verify.py
```
