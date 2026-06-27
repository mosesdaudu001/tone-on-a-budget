# test_tone_gate.py — unit tests for the Layer-B tone gate. Pure python, no GPU/network.
# Run:  python -m pytest approachB/test_tone_gate.py -q   (or)   python approachB/test_tone_gate.py

import math
import tone_gate as tg


# ----------------------------- helpers to fabricate per-clip score dicts -----------------------------
def clip(pred, target):
    """Build a minimal probe_score-shaped dict from per-TBU pred/target lists."""
    return {"pred": list(pred), "target": list(target)}


def perfect_clips(n=40):
    """A balanced population the instrument labels perfectly (H/M/L each present)."""
    out = []
    for i in range(n):
        tgt = ["H", "M", "L", "M", "H", "L"]
        out.append(clip(tgt, tgt))  # pred == target
    return out


def flatten_to_mid_clips(n=40):
    """Same targets, but the model predicts MID everywhere — high micro-acc (Mid is plurality) yet
    H and L recall are ZERO. This is the failure the per-class recall floor must catch."""
    out = []
    for i in range(n):
        tgt = ["H", "M", "L", "M", "H", "L"]
        out.append(clip(["M"] * len(tgt), tgt))
    return out


def abstain_hard_clips(n=40):
    """Predicts MID where target is MID (correct) but ABSTAINS (None) on every H and L. micro-acc on
    covered TBUs is a perfect 1.0, but H/L coverage is 0 — the coverage floor must catch this."""
    out = []
    for i in range(n):
        tgt = ["H", "M", "L", "M", "H", "L"]
        pred = [None if t in ("H", "L") else "M" for t in tgt]
        out.append(clip(pred, tgt))
    return out


def good_gen_clips(n=40, err_every=7):
    """Mostly-correct generation with a few errors sprinkled in — should pass against a weak floor."""
    out = []
    k = 0
    for i in range(n):
        tgt = ["H", "M", "L", "M", "H", "L"]
        pred = []
        for t in tgt:
            k += 1
            if k % err_every == 0:               # occasional wrong-but-not-Mid-collapse error
                pred.append({"H": "L", "L": "H", "M": "H"}[t])
            else:
                pred.append(t)
        out.append(clip(pred, tgt))
    return out


def shuf_floor_clips(n=40):
    """Matched-timbre tone-shuffled floor: tones permuted, so the instrument (scoring vs CORRECT
    orthography) gets ~chance. Low acc -> big margin under a good gen."""
    out = []
    for i in range(n):
        tgt = ["H", "M", "L", "M", "H", "L"]
        pred = ["M", "H", "M", "L", "M", "H"]   # deliberately wrong-ish vs tgt
        out.append(clip(pred, tgt))
    return out


# ----------------------------- aggregate() -----------------------------
def test_aggregate_perfect():
    a = tg.aggregate(perfect_clips())
    assert abs(a["micro_acc"] - 1.0) < 1e-9
    assert abs(a["coverage"] - 1.0) < 1e-9
    for c in ("H", "M", "L"):
        assert abs(a["per_class"][c]["recall"] - 1.0) < 1e-9
        assert abs(a["per_class"][c]["coverage"] - 1.0) < 1e-9


def test_aggregate_flatten_zero_hl_recall():
    a = tg.aggregate(flatten_to_mid_clips())
    # Mid is 2/6 of every clip -> micro-acc ~0.33, but H and L recall are exactly 0.
    assert abs(a["per_class"]["H"]["recall"] - 0.0) < 1e-9
    assert abs(a["per_class"]["L"]["recall"] - 0.0) < 1e-9
    assert a["per_class"]["H"]["coverage"] == 1.0  # it DID label them (just wrong)


def test_aggregate_abstain_zero_hl_coverage():
    a = tg.aggregate(abstain_hard_clips())
    assert a["per_class"]["H"]["coverage"] == 0.0
    assert a["per_class"]["L"]["coverage"] == 0.0
    assert math.isnan(a["per_class"]["H"]["recall"])  # nothing covered -> recall undefined
    # acc-on-covered is a perfect 1.0 (only Mids were scored, all correct) — the trap this guards against
    assert abs(a["micro_acc"] - 1.0) < 1e-9


def test_aggregate_empty_is_nan():
    a = tg.aggregate([])
    assert math.isnan(a["micro_acc"])
    assert a["n_scored"] == 0


# ----------------------------- the gate -----------------------------
def test_gate_passes_good_gen_over_weak_floor():
    v = tg.tone_gate(gen={"I1": good_gen_clips()},
                     shuf={"I1": shuf_floor_clips()},
                     required=("I1",))
    assert v["pass"] is True, tg.explain(v)


def test_gate_fails_flatten_to_mid():
    v = tg.tone_gate(gen={"I1": flatten_to_mid_clips()},
                     shuf={"I1": shuf_floor_clips()},
                     required=("I1",))
    assert v["pass"] is False
    assert v["per_instrument"]["I1"]["checks"]["H_recall"] is False
    assert v["per_instrument"]["I1"]["checks"]["L_recall"] is False


def test_gate_fails_abstain_hard():
    v = tg.tone_gate(gen={"I1": abstain_hard_clips()},
                     shuf={"I1": shuf_floor_clips()},
                     required=("I1",))
    assert v["pass"] is False
    assert v["per_instrument"]["I1"]["checks"]["H_coverage"] is False
    assert v["per_instrument"]["I1"]["checks"]["L_coverage"] is False


def test_gate_fails_when_floor_too_close():
    # Good gen but the "floor" is ALSO good (margin ~0) -> tone signal not above the matched-timbre floor.
    v = tg.tone_gate(gen={"I1": good_gen_clips()},
                     shuf={"I1": good_gen_clips()},   # identical => margin 0
                     required=("I1",))
    assert v["pass"] is False
    assert v["per_instrument"]["I1"]["checks"]["floor_margin"] is False


def test_dual_instrument_requires_kappa():
    gen = {"I1": good_gen_clips(), "I2": good_gen_clips()}
    shuf = {"I1": shuf_floor_clips(), "I2": shuf_floor_clips()}
    # No kappa_pairs supplied -> a dual-instrument gate cannot pass.
    v_missing = tg.tone_gate(gen=gen, shuf=shuf, kappa_pairs=None, required=("I1", "I2"))
    assert v_missing["pass"] is False and v_missing["kappa_pass"] is False
    # Agreeing instruments -> high kappa -> pass.
    pairs, skipped = tg.cocovered_pairs(gen["I1"], gen["I2"])
    assert skipped == 0 and len(pairs) > 0
    v_ok = tg.tone_gate(gen=gen, shuf=shuf, kappa_pairs=pairs, required=("I1", "I2"))
    assert v_ok["pass"] is True, tg.explain(v_ok)
    assert v_ok["kappa"] is not None and v_ok["kappa"] > 0.45


def test_cohen_kappa_bounds():
    assert tg.cohen_kappa([("H", "H"), ("M", "M"), ("L", "L")]) == 1.0
    assert math.isnan(tg.cohen_kappa([]))
    # systematic disagreement -> low/negative kappa
    k = tg.cohen_kappa([("H", "L"), ("L", "H"), ("H", "L"), ("L", "H")])
    assert k < 0.45


def test_cocovered_skips_misaligned_clips():
    a = [clip(["H", "M"], ["H", "M"])]
    b = [clip(["H"], ["H"])]            # different length target -> skipped, not zipped
    pairs, skipped = tg.cocovered_pairs(a, b)
    assert skipped == 1 and pairs == []


def test_ge_rejects_nan_and_none():
    assert tg._ge(0.8, 0.7) is True
    assert tg._ge(0.6, 0.7) is False
    assert tg._ge(float("nan"), 0.7) is False
    assert tg._ge(None, 0.7) is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
