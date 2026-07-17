"""풀이 A (공격형, uid232) — 최종 레시피 v4 (2026-07-17).

핵심 (전부 실측 근거):
  · payload 미러링 — 안 하면 서빙에서 0.92→0.00
  · ★혼합모양 학습: 원판(30~40핸드) + 병합100핸드 — 라이브 질의가 100핸드 배치라
    원판만 학습하면 comp 0.78, 100핸드 학습하면 0.95 (+0.1675)
  · 병합배치 가중 1.3 — 라이브 모양 우선 (A의 공격 포인트)
  · recency 반감기 3 — 그날 집중
  · 창의피처 포함 343개 (combined.py)
사용: python train.py --all   (전체 데이터로 최종 학습)
"""
import copy
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_ml.combined import chunk_features


def _root(p):
    for _ in range(7):
        if (p / "02_benchmark_data").exists():
            return p
        p = p.parent
    return p


ROOT = _root(HERE)
CHUNK_DIR = ROOT / "02_benchmark_data" / "chunks"
OUT = HERE / "model"; OUT.mkdir(exist_ok=True)

MIN_DATE = "2026-07-06"        # 현재 생성기 시대만
HALFLIFE = 3.0                 # A: 그날 집중
MERGED_BOOST = 1.3             # A: 라이브(100핸드) 모양 가중
MAX_MERGED_PER_DAY_LABEL = 60


def _mirror():
    try:
        from poker44.validator.payload_view import prepare_hand_for_miner
        return prepare_hand_for_miner
    except Exception:
        for c in [ROOT / "10_relearn_2026-07-15" / "Poker44-subnet-fresh",
                  ROOT / "01_subnet_code" / "Poker44-subnet"]:
            if (c / "poker44" / "validator" / "payload_view.py").exists():
                sys.path.insert(0, str(c))
                from poker44.validator.payload_view import prepare_hand_for_miner
                return prepare_hand_for_miner
    raise RuntimeError("prepare_hand_for_miner 없음")


def load(mirror):
    """원판 배치 + 병합100 배치(같은 날·같은 라벨 3개 이어붙임, 겹침, 상한)."""
    feats, ys, dates, shapes = [], [], [], []   # shape 0=원판, 1=병합100
    for p in sorted(CHUNK_DIR.glob("*.json")):
        if p.stem < MIN_DATE:
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        d = b["release"]["sourceDate"]
        per_label = {0: [], 1: []}
        for rec in b["chunks"]:
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # ★서빙 동일 변환
                feats.append(chunk_features(hands)); ys.append(int(y))
                dates.append(d); shapes.append(0)
                per_label[int(y)].append(hands)
        for lab in (0, 1):
            same = per_label[lab]
            cnt = 0
            for i in range(len(same)):
                if cnt >= MAX_MERGED_PER_DAY_LABEL or len(same) < 3:
                    break
                merged = (same[i] + same[(i + 1) % len(same)] + same[(i + 2) % len(same)])[:100]
                if len(merged) < 80:
                    continue
                feats.append(chunk_features(merged)); ys.append(lab)
                dates.append(d); shapes.append(1); cnt += 1
    return feats, np.array(ys, np.int32), np.array(dates), np.array(shapes)


def main(train_on_all=True):
    mirror = _mirror()
    print("풀이 A v4: 미러링 + 혼합모양(원판+병합100) + 병합가중 1.3 + 반감기 3")
    feats, y, dates, shapes = load(mirror)
    cols = sorted({k for f in feats for k in f})
    X = np.array([[f.get(n, 0.0) for n in cols] for f in feats], np.float64)
    print(f"rows={len(y)} (원판 {(shapes==0).sum()}, 병합100 {(shapes==1).sum()}) features={len(cols)} bot_rate={y.mean():.3f}")

    held = [] if train_on_all else sorted(set(dates.tolist()))[-1:]
    mask = np.ones(len(y), bool) if train_on_all else np.array([d not in set(held) for d in dates])
    uniq = sorted(set(dates.tolist())); dpos = {d: i for i, d in enumerate(uniq)}
    parr = np.array([dpos[d] for d in dates], float)
    sw = (np.power(0.5, (parr.max() - parr) / HALFLIFE) *
          np.where(shapes == 1, MERGED_BOOST, 1.0))[mask]
    print(f"train rows={mask.sum()} halflife={HALFLIFE} merged_boost={MERGED_BOOST}")

    models, weights = [], []
    for seed in range(5):
        m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=20, random_state=seed, verbose=-1)
        m.fit(X[mask], y[mask], sample_weight=sw); models.append(m); weights.append(1.0)
    for M, w in ((RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0), 1.2),
                 (LogisticRegression(max_iter=3000, C=0.5), 0.6)):
        M.fit(X[mask], y[mask], sample_weight=sw); models.append(M); weights.append(w)

    art = {"models": models, "model_weights": weights, "feature_names": cols, "calibrator": None,
           "metadata": {"name": "poker44-A-aggressive", "version": "4", "blend": "mean_proba",
                        "payload_mirrored": True, "recency_weighted_halflife_days": HALFLIFE,
                        "merged_boost": MERGED_BOOST, "mixed_shapes": True,
                        "safety": "topk_cap", "n_models": len(models),
                        "strategy": "live-shape first: native+merged100 mixed, merged-boosted",
                        "trained_on_all": train_on_all, "holdout_dates": held}}
    path = OUT / "poker44_model.joblib"
    joblib.dump(art, path, compress=3)
    print(f"saved {path.name} ({path.stat().st_size/1e6:.1f}MB, {len(models)} models)")


if __name__ == "__main__":
    main(train_on_all="--all" in sys.argv)
