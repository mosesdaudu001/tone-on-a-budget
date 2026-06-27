# test_tone_oracle.py — tests for the PSOLA tone-flip oracle.
# Run:  /home/moses/audio_env/bin/python approachB/test_tone_oracle.py
#
# Two layers:
#   (A) PURE tests (no parselmouth/models): summarize_oracle math, and run_oracle_clip's ORCHESTRATION
#       (candidate selection, detected flag, false-flip bookkeeping) with the PSOLA + scorers faked.
#   (B) A parselmouth-GATED test that psola_shift_window applies the exact requested in-window semitone
#       shift and ~0 out-of-window — codifying the manual verification from the recon workflow.

import math

import tone_oracle as orc


# ============================== (A1) summarize_oracle math ==============================
def _flip_row(meter, k, st, detected, ff=0, fft=4):
    return dict(kind="flip", tbu=0, src="L", k=k, semitones=st, meter=meter,
                pred_at_tbu=("H" if detected else "M"), expect="H",
                detected=detected, false_flips=ff, ff_total=fft)


def test_summarize_detect_by_k_and_false_flip():
    rows = []
    # k=0.5 -> 1/4 detect ; k=1.0 -> 4/4 detect ; false flips 1 per row at k=1.0
    for d in (True, False, False, False):
        rows.append(_flip_row("I1", 0.5, 3.0, d, ff=0))
    for d in (True, True, True, True):
        rows.append(_flip_row("I1", 1.0, 6.0, d, ff=1))
    b = orc.summarize_oracle(rows, "I1", trend_perm=400, seed=0)
    assert abs(b["detect_by_k"][0.5] - 0.25) < 1e-9
    assert abs(b["detect_by_k"][1.0] - 1.0) < 1e-9
    assert abs(b["strong_detect"] - 1.0) < 1e-9         # strong = detection at the largest k (1.0)
    # false-flip rate = total changed / total untouched = (0*4 + 1*4) / (4*4 + 4*4) = 4/32
    assert abs(b["false_flip_rate"] - (4 / 32)) < 1e-9
    assert b["n_flips"] == 8


def test_summarize_trend_significant_when_monotone():
    rows = []
    for st, rate in ((2.0, 0.2), (4.0, 0.6), (6.0, 0.95)):
        for j in range(40):
            rows.append(_flip_row("I1", st / 6.0, st, detected=(j / 40.0) < rate))
    b = orc.summarize_oracle(rows, "I1", trend_perm=500, seed=0)
    assert b["trend_rho"] > 0.3 and b["trend_p"] < 0.05, (b["trend_rho"], b["trend_p"])


def test_measure_delta_HL_detrended_recovers_through_downdrift():
    # build a clip with strong downdrift: tones alternate H/L, base residuals +-3 st, slope -3 st/s baked
    # in. RAW H-L medians are compressed/confused by declination; the DETRENDED measure must recover ~6 st.
    import numpy as np

    def st_to_hz(st):
        return 100.0 * (2.0 ** (st / 12.0))

    tones = ["H", "L", "H", "L", "H", "L"]
    base = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]
    dur, fr, slope = 0.2, 0.01, -3.0
    times = np.arange(0.0, len(tones) * dur - 1e-9, fr)
    f0 = np.full(times.shape, np.nan, dtype="float64")
    wins = []
    for i in range(len(tones)):
        t0, t1 = i * dur, (i + 1) * dur
        wins.append((t0, t1))
        hz = st_to_hz(base[i] + slope * 0.5 * (t0 + t1))
        f0[(times >= t0) & (times < t1)] = hz
    pre = dict(f0=f0, times=times, wins=wins, tones=tones, method="forced-align", backend="synthetic")
    d = orc.measure_delta_HL(pre)
    assert d is not None and abs(d - 6.0) < 1.2, d        # true tonal H-L gap recovered despite downdrift


def test_summarize_control_trip():
    rows = [dict(kind="control", tbu=0, control="roundtrip", meter="I1", pred_at_tbu="L", src="L",
                 flipped=False),
            dict(kind="control", tbu=1, control="tiny+0.2st", meter="I1", pred_at_tbu="H", src="L",
                 flipped=True)]
    b = orc.summarize_oracle(rows, "I1", trend_perm=50)
    assert abs(b["control_trip"] - 0.5) < 1e-9


# ============================== (A2) run_oracle_clip orchestration (faked) ==============================
def test_run_oracle_clip_candidate_and_bookkeeping(monkeypatch=None):
    # tones: TBU0=H(both right), TBU1=M(skip), TBU2=L(I2 WRONG on clean -> excluded), TBU3=H(both right)
    pre = {"tones": ["H", "M", "L", "H"], "wins": [(0., 1.), (1., 2.), (2., 3.), (3., 4.)]}
    base = {"I1": {"pred": ["H", "M", "L", "H"]},
            "I2": {"pred": ["H", "M", "M", "H"]}}      # I2 wrong on the L -> TBU2 not a candidate

    _saved = (orc.measure_delta_HL, orc.psola_shift_window, orc.psola_roundtrip)
    orc.measure_delta_HL = lambda pre, **kw: 6.0       # bypass F0 synthesis (tested in test_tone_f0_abs)
    # fake PSOLA: encode the window into a marker the fake scorers can read
    orc.psola_shift_window = lambda wav, t0, t1, st, **kw: ("shift", t0, t1, st)
    orc.psola_roundtrip = lambda wav, **kw: ("roundtrip",)

    wins = pre["wins"]; tones = pre["tones"]

    def fake_scorer(name):
        def score(y):
            pred = list(base[name]["pred"])
            if isinstance(y, tuple) and y[0] == "shift":
                _, t0, t1, st = y
                i = wins.index((t0, t1))
                if abs(st) >= 4.0:                       # strong flip detected; weak (k=0.5 -> 3st) not
                    pred[i] = "H" if tones[i] == "L" else "L"
                if name == "I1" and i == 0 and abs(st) >= 4.0:
                    pred[1] = "L"                        # inject ONE false-flip on a neighbour for I1
            return {"pred": pred}
        return score

    rows = orc.run_oracle_clip(None, orc.SR, "txt", pre, base,
                               {"I1": fake_scorer("I1"), "I2": fake_scorer("I2")},
                               ks=(0.5, 1.0), target_classes=("H", "L"))
    flips = [r for r in rows if r["kind"] == "flip"]
    flipped_tbus = {r["tbu"] for r in flips}
    assert flipped_tbus == {0, 3}, flipped_tbus           # TBU2 excluded (I2 wrong on clean), M skipped
    # strong flips (k=1.0, st=6) detected; weak (k=0.5, st=3) not
    strong = [r for r in flips if r["k"] == 1.0]
    weak = [r for r in flips if r["k"] == 0.5]
    assert all(r["detected"] for r in strong), strong
    assert all(not r["detected"] for r in weak), weak
    # I1 false-flip on neighbour at TBU0 strong; I2 has none
    i1_strong_t0 = [r for r in strong if r["meter"] == "I1" and r["tbu"] == 0]
    assert all(r["false_flips"] == 1 for r in i1_strong_t0), i1_strong_t0
    i2_strong = [r for r in strong if r["meter"] == "I2"]
    assert all(r["false_flips"] == 0 for r in i2_strong), i2_strong
    # negative controls present and not flipping class
    ctrls = [r for r in rows if r["kind"] == "control"]
    assert ctrls and all(r["flipped"] is False for r in ctrls)
    orc.measure_delta_HL, orc.psola_shift_window, orc.psola_roundtrip = _saved   # restore reals


# ============================== (B) parselmouth-gated: real semitone shift ==============================
def test_psola_shift_window_real_semitones():
    try:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
    except Exception as e:
        print("  SKIP test_psola_shift_window_real_semitones (parselmouth missing):", e)
        return
    sr = orc.SR
    t = np.arange(0, 1.0, 1.0 / sr)
    wav = (0.3 * np.sin(2 * np.pi * 140.0 * t)).astype("float32")   # steady 140 Hz
    y = orc.psola_shift_window(wav, 0.40, 0.60, 4.0, sr=sr)         # +4 st inside [0.40,0.60)
    assert abs(len(y) - len(wav)) <= 2, (len(y), len(wav))

    def mean_hz(sig, a, b):
        snd = parselmouth.Sound(np.asarray(sig, "float64"), sampling_frequency=sr)
        pitch = call(snd, "To Pitch", 0.01, 65.0, 400.0)
        return call(pitch, "Get mean", a, b, "Hertz")

    inside = mean_hz(y, 0.43, 0.57)
    outside = mean_hz(y, 0.05, 0.30)
    assert abs(12 * math.log2(inside / 140.0) - 4.0) < 0.5, inside     # ~+4 st in-window
    assert abs(12 * math.log2(outside / 140.0)) < 0.5, outside          # ~0 st out-of-window


def test_psola_shift_windows_multi_disjoint():
    try:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
    except Exception as e:
        print("  SKIP test_psola_shift_windows_multi_disjoint (parselmouth missing):", e)
        return
    sr = orc.SR
    t = np.arange(0, 1.0, 1.0 / sr)
    wav = (0.3 * np.sin(2 * np.pi * 140.0 * t)).astype("float32")
    y = orc.psola_shift_windows(wav, [(0.10, 0.30, +5.0), (0.60, 0.80, -5.0)], sr=sr)  # one pass

    def mean_hz(sig, a, b):
        snd = parselmouth.Sound(np.asarray(sig, "float64"), sampling_frequency=sr)
        return call(call(snd, "To Pitch", 0.01, 65.0, 400.0), "Get mean", a, b, "Hertz")

    up = mean_hz(y, 0.13, 0.27)
    down = mean_hz(y, 0.63, 0.77)
    assert abs(12 * math.log2(up / 140.0) - 5.0) < 0.6, up        # window 1 raised +5 st
    assert abs(12 * math.log2(down / 140.0) + 5.0) < 0.6, down     # window 2 lowered -5 st


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
