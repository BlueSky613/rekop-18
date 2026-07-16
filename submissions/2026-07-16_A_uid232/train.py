"""풀이 A (공격형) — 벤치 최대치 목표.

전량 데이터 + 강한 앙상블(용량↑·규제↓) + payload 미러링 + mild recency.
공통 원칙(미러링·top-K캡·recency·매니페스트없음·비복제) 전부 적용. 캘리브레이션은 지렛대 아니므로
차별점은 '모델 용량/데이터'. A는 벤치 적합 최대치에 베팅.
"""
import json, os, sys
from pathlib import Path
import joblib, lightgbm as lgb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_ml.combined import chunk_features


def _root(p):
    configured = os.getenv("POKER44_BENCHMARK_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "02_benchmark_data").exists():
            return root
        if (root / "chunks").exists():
            return root.parent
        raise RuntimeError(f"POKER44_BENCHMARK_ROOT does not contain benchmark chunks: {root}")

    for _ in range(7):
        if (p / "02_benchmark_data").exists():
            return p
        p = p.parent
    return p


ROOT = _root(HERE)
CHUNK_DIR = ROOT / "02_benchmark_data" / "chunks"
OUT = HERE / "model"; OUT.mkdir(exist_ok=True)

# A(공격형=그날 최대점수): 실측 최적 레시피로 교정.
#   · 현재 생성기(07-06+)만 — 옛 데이터는 도움 안 됨(daily_max 실측)
#   · 중간 recency 반감기 3 — 극단(1~2)은 손해, 4가 최적이나 A는 그날 집중 위해 3
#   · 창의 피처는 combined.py에 통합됨, 고용량 앙상블 유지
MIN_DATE = "2026-07-06"
HALFLIFE = 3.0


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
    import copy
    bags = []
    for p in sorted(CHUNK_DIR.glob("*.json")):
        if p.stem < MIN_DATE:
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        d = b["release"]["sourceDate"]
        for rec in b["chunks"]:
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # ★서빙 동일 변환
                bags.append((chunk_features(hands), int(y), d))
    return bags


def main(train_on_all=True):
    mirror = _mirror()
    print("풀이 A: payload 미러링 + 전량 데이터 + 강한 앙상블")
    bags = load(mirror)
    cols = sorted({k for f, _, _ in bags for k in f})
    X = np.array([[f.get(n, 0.0) for n in cols] for f, _, _ in bags], np.float64)
    y = np.array([b[1] for b in bags], np.int32)
    dates = np.array([b[2] for b in bags])
    print(f"bags={len(bags)} features={len(cols)} bot_rate={y.mean():.3f}")

    held = [] if train_on_all else sorted(set(dates.tolist()))[-3:]
    mask = np.ones(len(bags), bool) if train_on_all else np.array([d not in set(held) for d in dates])
    uniq = sorted(set(dates.tolist())); dpos = {d: i for i, d in enumerate(uniq)}
    parr = np.array([dpos[d] for d in dates], float)
    sw = np.power(0.5, (parr.max() - parr) / HALFLIFE)[mask]
    print(f"train rows={mask.sum()} recency halflife={HALFLIFE}")

    models, weights = [], []
    # 강한 LightGBM ×5 (용량↑·규제↓)
    for seed in range(5):
        m = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.025, num_leaves=63,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=15, reg_lambda=0.5, random_state=seed, verbose=-1)
        m.fit(X[mask], y[mask], sample_weight=sw); models.append(m); weights.append(1.0)
    # 트리 다양성: RF + ExtraTrees + HistGBM + LogReg
    for M in (RandomForestClassifier(n_estimators=700, min_samples_leaf=2, random_state=0, n_jobs=-1),
              ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, random_state=0, n_jobs=-1),
              HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, random_state=0),
              LogisticRegression(max_iter=3000, C=0.5)):
        M.fit(X[mask], y[mask], sample_weight=sw); models.append(M); weights.append(5.0 / 4.0)

    art = {"models": models, "model_weights": weights, "feature_names": cols, "calibrator": None,
           "metadata": {"name": "poker44-A-aggressive", "version": "3", "blend": "mean_proba",
                        "payload_mirrored": True, "recency_weighted_halflife_days": HALFLIFE,
                        "safety": "topk_cap", "n_models": len(models),
                        "strategy": "max-benchmark: all-data + high-capacity ensemble",
                        "trained_on_all": train_on_all, "holdout_dates": held}}
    path = OUT / "poker44_model.joblib"
    joblib.dump(art, path, compress=3)
    print(f"saved {path.name} ({path.stat().st_size/1e6:.1f}MB, {len(models)} models)")


if __name__ == "__main__":
    main(train_on_all="--all" in sys.argv)
