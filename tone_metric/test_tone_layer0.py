# test_tone_layer0.py — unit tests for the Layer-0 instrument-validation gate. Pure python, CPU.
# Run:  /home/moses/audio_env/bin/python approachB/test_tone_layer0.py
#   (or)  python -m pytest approachB/test_tone_layer0.py -q
#
# We synthesize populations of per-clip score dicts (pred/target lists) with KNOWN per-class behaviour and
# assert the corrected statistics: balanced accuracy penalizes flatten-to-Mid, the shuffled-target control
# sits at prior-free chance (NOT the Sum p^2 raw-acc trap), min-support gates to INDETERMINATE, bootstrap
# lower-bounds sit below the point estimate, and the permutation trend test separates rising from flat.

import math
import random

import tone_layer0 as L0

# fast thresholds for tests (small bootstrap / few perms) — overrides merged into LAYER0_THRESHOLDS
FAST = {"bootstrap_n": 200, "trend_perm": 400}


# ----------------------------- synthetic population builder -----------------------------
def make_scores(n_clips=120, tbus=10, recall=None, coverage=1.0, mix=("H", "M", "L"),
                weights=(0.2, 0.6, 0.2), seed=0):
    """Build n_clips score dicts. Each TBU's target is drawn from `mix` with `weights`; the prediction is
    correct with prob recall[class], else a uniformly-wrong other class; with prob (1-coverage) the TBU is
    abstained (pred None). This lets us dial per-class recall and coverage independently."""
    recall = recall or {"H": 0.8, "M": 0.9, "L": 0.8}
    rng = random.Random(seed)
    others = {c: [d for d in ("H", "M", "L") if d != c] for c in ("H", "M", "L")}
    out = []
    for _ in range(n_clips):
        target, pred = [], []
        for _ in range(tbus):
            t = rng.choices(mix, weights=weights)[0]
            target.append(t)
            if rng.random() > coverage:
                pred.append(None)
            elif rng.random() < recall[t]:
                pred.append(t)
            else:
                pred.append(rng.choice(others[t]))
        out.append({"pred": pred, "target": target})
    return out


# ----------------------------- balanced accuracy vs raw -----------------------------
def test_balanced_accuracy_penalizes_flatten_to_mid():
    # A flatten-to-Mid instrument: Mid recall ~1.0, H/L recall ~0.1. RAW micro-acc is high (Mid is 60%),
    # but balanced accuracy must collapse toward chance.
    flat = make_scores(recall={"H": 0.1, "M": 0.98, "L": 0.1}, seed=1)
    agg = L0.aggregate(flat)
    bal = L0.balanced_accuracy(agg)
    assert agg["micro_acc"] > 0.60, agg["micro_acc"]      # raw acc is fooled
    assert bal < 0.45, bal                                # balanced acc is not
    good = make_scores(recall={"H": 0.85, "M": 0.9, "L": 0.85}, seed=2)
    assert L0.balanced_accuracy(L0.aggregate(good)) > 0.80


def test_weighted_f1_matches_recall_when_balanced():
    good = make_scores(recall={"H": 0.9, "M": 0.9, "L": 0.9}, weights=(1/3, 1/3, 1/3), seed=3)
    wf1 = L0.weighted_f1(L0.aggregate(good))
    assert 0.82 <= wf1 <= 0.95, wf1


# ----------------------------- bootstrap lower bound -----------------------------
def test_bootstrap_lcb_below_point_and_stable():
    good = make_scores(recall={"H": 0.8, "M": 0.9, "L": 0.8}, seed=4)
    point, lcb = L0.bootstrap_lcb(good, L0.balanced_accuracy, n_boot=300, q=0.10, seed=0)
    assert lcb <= point + 1e-9
    assert point - lcb < 0.08            # a 1200-TBU sample has a tight CI
    assert lcb > 0.70


def test_bootstrap_lcb_wider_on_small_sample():
    big = make_scores(n_clips=120, seed=5)
    small = make_scores(n_clips=12, seed=5)
    _, lcb_big = L0.bootstrap_lcb(big, L0.balanced_accuracy, n_boot=300, seed=0)
    _, lcb_small = L0.bootstrap_lcb(small, L0.balanced_accuracy, n_boot=300, seed=0)
    pt_big = L0.balanced_accuracy(L0.aggregate(big))
    pt_small = L0.balanced_accuracy(L0.aggregate(small))
    assert (pt_big - lcb_big) < (pt_small - lcb_small)    # smaller sample -> wider interval


# ----------------------------- the shuffled-target trap (the worst draft bug) -----------------------------
def test_shuffled_target_control_is_prior_free_chance():
    # With 60% Mid, a label-PERMUTED control scored in RAW accuracy lands ~Sum p^2 = 0.44 (ABOVE a naive
    # 0.40 ceiling -> would false-fail a perfect meter). In BALANCED accuracy it lands ~1/3, safely under
    # the 0.40 ceiling. This is the whole reason the gate uses balanced accuracy for controls.
    good = make_scores(recall={"H": 0.9, "M": 0.9, "L": 0.9}, seed=6)
    shuffled = L0.permute_targets(good, seed=0)
    agg_sh = L0.aggregate(shuffled)
    raw = agg_sh["micro_acc"]
    bal = L0.balanced_accuracy(agg_sh)
    assert raw > 0.40, raw                # the trap: raw chance is ABOVE 0.40 for Mid-heavy Yoruba
    assert bal <= 0.40, bal               # balanced chance is below -> control passes for a real meter


def test_all_mid_baseline_is_one_third():
    good = make_scores(seed=7)
    bal = L0.balanced_accuracy(L0.aggregate(L0.force_all_mid(good)))
    assert abs(bal - 1/3) < 0.02, bal


# ----------------------------- trend test -----------------------------
def test_trend_increasing_detects_monotone():
    # raw (k, detected) observations: detection rises 0.2 -> 0.6 -> 0.95 across k in {1,2,3}, 60 obs each
    rng = random.Random(0)
    xs, ys = [], []
    for k, rate in ((1, 0.2), (2, 0.6), (3, 0.95)):
        for _ in range(60):
            xs.append(k); ys.append(1 if rng.random() < rate else 0)
    rho, p = L0.trend_increasing(xs, ys, n_perm=400, seed=0)
    assert rho > 0.3 and p < 0.05, (rho, p)


def test_trend_flat_is_not_significant():
    rng = random.Random(1)
    xs, ys = [], []
    for k in (1, 2, 3):
        for _ in range(60):
            xs.append(k); ys.append(1 if rng.random() < 0.5 else 0)   # no trend
    rho, p = L0.trend_increasing(xs, ys, n_perm=400, seed=0)
    assert p > 0.05, (rho, p)


# ----------------------------- role-aware gate: PASS / FAIL / INDETERMINATE -----------------------------
ORACLE = {"detect_by_k": {0.5: 0.0, 1.0: 0.0, 1.5: 0.0}, "strong_detect": 0.0,
          "false_flip_rate": 0.08, "trend_p": 1.0, "n_flips": 400}   # reported only — never gates


def _tone_controls(real, seed=0):
    """Controls a GENUINE F0 tone meter (I2) produces: collapses to chance on flat F0 (monotone), drops
    on tone-scramble. The monotone-collapse + margin are what CERTIFY it reads pitch."""
    return {"shuffled_target": L0.permute_targets(real, seed=seed),
            "monotone": L0.force_all_mid(real),                                          # flat F0 -> 0.33
            "all_mid": L0.force_all_mid(real),
            "psola_shuffle": make_scores(recall={"H": 0.25, "M": 0.5, "L": 0.25}, seed=seed + 9)}


def _quality_controls(real, seed=0):
    """Controls a PITCH-BLIND quality meter (I1-like) produces: monotone does NOT collapse (it reads
    segments, not pitch) and tone-scramble barely drops it (no margin). It can only pass as 'quality'."""
    return {"shuffled_target": L0.permute_targets(real, seed=seed),
            "monotone": make_scores(recall={"H": 0.5, "M": 0.6, "L": 0.5}, seed=seed + 5),    # ~0.53, NOT collapsed
            "all_mid": L0.force_all_mid(real),
            "psola_shuffle": make_scores(recall={"H": 0.82, "M": 0.86, "L": 0.82}, seed=seed + 9)}  # ~real -> no margin


def test_gate_pass_quality_and_tone():
    real = {"I1": make_scores(recall={"H": 0.84, "M": 0.9, "L": 0.84}, coverage=0.95, seed=11),
            "I2": make_scores(recall={"H": 0.68, "M": 0.74, "L": 0.68}, coverage=0.9, seed=21)}
    controls = {"I1": _quality_controls(real["I1"]), "I2": _tone_controls(real["I2"])}
    oracle = {"I1": ORACLE, "I2": ORACLE}
    v = L0.layer0_gate(real, controls, oracle, roles={"I1": "quality", "I2": "tone"},
                       thresholds=FAST)
    assert v["layer0_verdict"] == "PASS", L0.explain(v)


def test_quality_passes_despite_pitch_blindness():
    # the KEY reframe: I1 is pitch-blind (monotone stays high, no margin) yet PASSES as 'quality'
    real = {"I1": make_scores(recall={"H": 0.84, "M": 0.9, "L": 0.84}, coverage=0.95, seed=12)}
    v = L0.layer0_gate(real, {"I1": _quality_controls(real["I1"])}, {"I1": ORACLE},
                       roles={"I1": "quality"}, thresholds=FAST)
    assert v["layer0_verdict"] == "PASS", L0.explain(v)


def test_tone_role_fails_a_pitch_blind_meter():
    # the SAME I1-like pitch-blind meter, if you tried to certify it as a TONE meter, FAILS — its monotone
    # control does NOT collapse and it has no scramble margin. This is what guards against I1-as-tone.
    real = {"I2": make_scores(recall={"H": 0.84, "M": 0.9, "L": 0.84}, coverage=0.95, seed=13)}
    v = L0.layer0_gate(real, {"I2": _quality_controls(real["I2"])}, {"I2": ORACLE},
                       roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "FAIL", L0.explain(v)
    checks = v["per_instrument"]["I2"]["checks"]
    assert checks["monotone_collapse"] is False and checks["margin"] is False


def test_tone_role_fails_flatten_to_mid():
    real = {"I2": make_scores(recall={"H": 0.15, "M": 0.98, "L": 0.15}, coverage=0.9, seed=14)}
    v = L0.layer0_gate(real, {"I2": _tone_controls(real["I2"])}, {"I2": ORACLE},
                       roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "FAIL", L0.explain(v)
    c = v["per_instrument"]["I2"]["checks"]
    assert c["balanced_acc"] is False and (c["H_recall"] is False or c["L_recall"] is False)


def test_tone_role_fails_no_margin():
    # high real accuracy but tone-scramble doesn't drop it -> not actually reading tone -> FAIL
    real = {"I2": make_scores(recall={"H": 0.7, "M": 0.78, "L": 0.7}, coverage=0.9, seed=15)}
    controls = _tone_controls(real["I2"])
    controls["psola_shuffle"] = make_scores(recall={"H": 0.69, "M": 0.77, "L": 0.69}, seed=99)  # no drop
    v = L0.layer0_gate(real, {"I2": controls}, {"I2": ORACLE}, roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "FAIL", L0.explain(v)
    assert v["per_instrument"]["I2"]["checks"]["margin"] is False


def test_gate_indeterminate_on_thin_support():
    real = {"I2": make_scores(n_clips=8, recall={"H": 0.7, "M": 0.78, "L": 0.7}, seed=16)}
    v = L0.layer0_gate(real, {"I2": _tone_controls(real["I2"])}, {"I2": ORACLE},
                       roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "INDETERMINATE", L0.explain(v)
    assert v["per_instrument"]["I2"]["support_ok"] is False


def test_tone_role_indeterminate_without_monotone_control():
    # the tone role REQUIRES the monotone control as evidence it reads pitch; absent -> INDETERMINATE
    real = {"I2": make_scores(recall={"H": 0.7, "M": 0.78, "L": 0.7}, coverage=0.9, seed=17)}
    controls = _tone_controls(real["I2"]); controls.pop("monotone")
    v = L0.layer0_gate(real, {"I2": controls}, {"I2": ORACLE}, roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "INDETERMINATE", L0.explain(v)
    assert v["per_instrument"]["I2"]["evidence_ok"] is False


def test_oracle_is_reported_not_gated():
    # a terrible oracle (detect 0, trend_p 1.0) must NOT fail a tone meter that passes the real checks —
    # the oracle is diagnostic only in this reframe.
    real = {"I2": make_scores(recall={"H": 0.68, "M": 0.74, "L": 0.68}, coverage=0.9, seed=18)}
    v = L0.layer0_gate(real, {"I2": _tone_controls(real["I2"])},
                       {"I2": {"detect_by_k": {1.5: 0.0}, "strong_detect": 0.0, "trend_p": 1.0,
                               "false_flip_rate": 0.5, "n_flips": 400}},
                       roles={"I2": "tone"}, thresholds=FAST)
    assert v["layer0_verdict"] == "PASS", L0.explain(v)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
