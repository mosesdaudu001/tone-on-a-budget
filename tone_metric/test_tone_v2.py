# Unit tests for tone_eval_v2 (run: cd approachB && python test_tone_v2.py — needs numpy, swift-f0; no GPU)
import math, os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tone_eval_v2 as v2


def _mk_pre(tones, st_levels, slope=0.0, dur=0.2, hop=0.016, f0_ref=100.0, drop=()):
    """Build a synthetic precompute dict: one window per TBU, constant per-TBU semitone
    st_levels[tone] + slope*t, NaN-voided windows listed in `drop`."""
    n = len(tones)
    wins = [(i * dur, (i + 1) * dur) for i in range(n)]
    T = n * dur
    times = np.arange(0, T, hop)
    f0 = np.full_like(times, np.nan, dtype="float64")
    for i, tn in enumerate(tones):
        m = (times >= wins[i][0]) & (times < wins[i][1])
        if i in drop:
            continue
        st = st_levels[tn] + slope * ((wins[i][0] + wins[i][1]) / 2)
        f0[m] = f0_ref * 2 ** (st / 12.0)
    return dict(f0=f0, times=times, wins=wins, tones=list(tones), method="synthetic", backend="injected")


def test_tbu_seq():
    assert v2.tbu_seq("Báwo ni") == ["H", "M", "M"]
    assert v2.tbu_seq("ọkọ̀") == ["M", "L"]
    assert v2.tbu_seq("ńlá") == ["H", "H"]          # toned syllabic nasal IS a TBU
    print("PASS tbu_seq")


def test_flat_for_align_nasal_fix():
    flat, pos = v2._flat_for_align("ọkọ̀ ńlá")
    assert flat == "ọkọ nla", repr(flat)             # dot-below KEPT, tone marks stripped
    tones = v2.tbu_seq("ọkọ̀ ńlá")
    assert len(pos) == len(tones) == 4               # v1 dropped the ń window -> len mismatch
    assert [flat[p] for p in pos] == ["ọ", "ọ", "n", "a"]
    print("PASS flat_for_align (nasal fix, dot kept)")


LEVELS = {"H": 4.0, "M": 0.0, "L": -4.0}


def test_perfect_downdrift():
    tones = list("HMLHMLHMLHML")
    for slope in (0.0, -1.0, -1.7):
        pre = _mk_pre(tones, LEVELS, slope=slope)
        r = v2.score_from_precomputed(pre)
        assert r["coverage"] == 1.0
        assert r["accuracy"] == 1.0, (slope, r["accuracy"], r["pred"], r["target"])
    print("PASS perfect speech scores 1.0 under downdrift up to -1.7 st/s")


def test_detrend_helps_long():
    # 20 TBUs over 4 s at -1.5 st/s: level transitions accumulate -0.3 st/gap; with adjacent windows
    # that's inside the fall threshold, so build LONG gaps by spacing windows out (1 TBU per 0.8 s)
    tones = list("HHMMLLHHMM")
    pre = _mk_pre(tones, LEVELS, slope=-1.5, dur=0.8)
    on = v2.score_from_precomputed(pre, detrend=True)
    off = v2.score_from_precomputed(pre, detrend=False)
    assert on["accuracy"] >= off["accuracy"]
    assert on["accuracy"] == 1.0, (on["accuracy"], on["pred"], on["target"])
    print(f"PASS detrend (on {on['accuracy']:.2f} >= off {off['accuracy']:.2f}; slope fit {on['slope']:.2f})")


def test_trended_tone_patterns():
    # ADVERSARIAL (caught by review 2026-06-10): tone patterns that TREND with time. A raw OLS detrend
    # absorbs the pattern itself (HHHMMMLLL fits a steep "declination"; LLLMMMHHH clamps to 0 and leaves
    # real downdrift in). Tone-centered (Frisch-Waugh) fitting must score these at 1.0.
    for tones, slope in (("HHHMMMLLL", 0.0), ("HHHMMMLLL", -1.0),
                         ("LLLMMMHHH", -1.5), ("LLLMMMHHH", 0.0),
                         ("HLHLHLHL", -1.0), ("MMHHLLMM", -0.8)):
        pre = _mk_pre(list(tones), LEVELS, slope=slope, dur=0.8)   # long gaps = worst case
        r = v2.score_from_precomputed(pre)
        assert r["accuracy"] == 1.0, (tones, slope, r["accuracy"], round(r["slope"], 2), r["pred"], r["target"])
        # the fitted slope must recover the TRUE declination, not the tone pattern
        assert abs(r["slope"] - slope) < 0.35, (tones, slope, r["slope"])
    print("PASS trended-tone patterns (Frisch-Waugh detrend recovers true slope, acc 1.0)")


def test_abstention():
    tones = list("HMLHML")
    pre = _mk_pre(tones, LEVELS, drop=(2,))          # TBU 2 unvoiced
    r = v2.score_from_precomputed(pre)
    assert r["n_trans"] == 5 and r["n_scored"] == 3  # transitions 1->2 and 2->3 abstain
    assert abs(r["coverage"] - 3 / 5) < 1e-9
    assert r["accuracy"] == 1.0
    assert r["pred"][1] is None and r["pred"][2] is None
    print("PASS abstention (unvoiced TBU drops its 2 transitions, no 'M' freebie)")


def test_monotone_not_rewarded():
    tones = list("HMLHML")                            # varied targets
    pre = _mk_pre(tones, {"H": 0.0, "M": 0.0, "L": 0.0})   # flat monotone audio
    r = v2.score_from_precomputed(pre)
    level_frac = sum(t == "level" for t in r["target"]) / r["n_trans"]
    assert r["accuracy"] == level_frac == 0.0        # this pattern has NO level transitions
    pre2 = _mk_pre(list("HHMMLL"), {"H": 0.0, "M": 0.0, "L": 0.0})
    r2 = v2.score_from_precomputed(pre2)
    assert abs(r2["accuracy"] - 3 / 5) < 1e-9        # only its 3 level transitions score
    print("PASS monotone audio cannot game varied-tone text")


def test_shuffled_control():
    truth = list("HMLHMLHMLH")
    pre = _mk_pre(truth, LEVELS)
    pre["tones"] = list("LLHHMMLLHH")                 # mismatched target text
    r = v2.score_from_precomputed(pre)
    assert r["accuracy"] <= 0.6, r["accuracy"]
    print(f"PASS shuffled-target control ({r['accuracy']:.2f} <= 0.6)")


def test_degenerate():
    pre = _mk_pre(["H"], LEVELS)
    r = v2.score_from_precomputed(pre)
    assert r["n_trans"] == 0 and math.isnan(r["accuracy"]) and r["coverage"] == 0.0
    print("PASS single-TBU degenerate (NaN accuracy, no crash)")


def test_end_to_end_swiftf0():
    # 6 harmonic-rich 'syllables', tones H M L H M L with -1 st/s downdrift, scored via the REAL
    # SwiftF0 backend + proportional windows (continuous voicing -> equal split is exact).
    sr = 16000
    tones = list("HML" * 2)
    dur, segs = 0.25, []
    for i, tn in enumerate(tones):
        st = LEVELS[tn] + -1.0 * (i + 0.5) * dur
        f0 = 130.0 * 2 ** (st / 12.0)
        t = np.arange(int(sr * dur)) / sr
        segs.append(sum((0.5 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6)))
    wav = np.concatenate(segs).astype("float32")
    text = "bá ba bà bá ba bà"
    assert v2.tbu_seq(text) == tones
    r = v2.tone_transition_score(wav, sr, text)
    assert r["backend"] == "swift-f0", r["backend"]
    assert r["method"] == "proportional"
    assert r["coverage"] >= 0.8, r
    assert r["accuracy"] == 1.0, (r["accuracy"], r["pred"], r["target"])
    print(f"PASS end-to-end SwiftF0 (acc {r['accuracy']:.2f}, cov {r['coverage']:.2f}, slope {r['slope']:.2f})")


if __name__ == "__main__":
    test_tbu_seq()
    test_flat_for_align_nasal_fix()
    test_perfect_downdrift()
    test_detrend_helps_long()
    test_trended_tone_patterns()
    test_abstention()
    test_monotone_not_rewarded()
    test_shuffled_control()
    test_degenerate()
    test_end_to_end_swiftf0()
    print("\nALL TESTS PASS")
