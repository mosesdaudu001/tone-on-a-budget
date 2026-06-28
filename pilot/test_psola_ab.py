#!/usr/bin/env python3
# coding=utf-8
# test_psola_ab.py — PURE-LOGIC tests for the A/B PSOLA tone-validation rebuild (no parselmouth / MMS / audio).
#   covers: oracle.voiced_rhyme_window, build_psola_form.salience_check, build_flip_contour/_orig_contour_pts,
#           and score_psola (%-correct, paired-win, catch gate, <24 guard, perfect rater).
# Run:  python pilot/test_psola_ab.py   (from the repo root, with numpy installed)

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "tone_metric"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import tone_oracle as orc
import build_psola_form as B
import score_psola as S


# ============================== voiced_rhyme_window ==============================
def _pre(f0, times, wins, tones=None):
    return dict(f0=np.asarray(f0, float), times=np.asarray(times, float), wins=wins,
                tones=tones or ["M"] * len(wins), method="forced-align", backend="test")


def test_window_grows_to_voiced_span():
    # 10 ms frames; TBU 1 sits in a voiced span [0.10,0.28); silence (NaN) brackets it.
    times = [i * 0.01 for i in range(40)]
    f0 = [np.nan] * 40
    for k in range(10, 28):
        f0[k] = 150.0
    wins = [(0.0, 0.05), (0.14, 0.16), (0.40, 0.42)]   # TBU1's CTC sliver is ~20 ms inside the rhyme
    pre = _pre(f0, times, wins)
    w = orc.voiced_rhyme_window(pre, 1, min_ms=60.0)
    assert w is not None, "rhyme should be found"
    t0, t1 = w
    assert abs(t0 - 0.10) < 1e-6 and abs(t1 - 0.27) < 1e-6, (t0, t1)  # last voiced frame index 27 -> t=0.27


def test_window_clamps_to_neighbour():
    # voiced everywhere, but neighbour windows must bound the growth (no bleed).
    times = [i * 0.01 for i in range(60)]
    f0 = [150.0] * 60
    wins = [(0.05, 0.07), (0.25, 0.27), (0.45, 0.47)]
    pre = _pre(f0, times, wins)
    t0, t1 = orc.voiced_rhyme_window(pre, 1, min_ms=20.0)
    assert t0 >= wins[0][1] - 1e-9, (t0, "must not cross left neighbour edge 0.07")
    assert t1 < wins[2][0] + 1e-9, (t1, "must not cross right neighbour edge 0.45")


def test_window_none_when_too_short():
    times = [i * 0.01 for i in range(40)]
    f0 = [np.nan] * 40
    for k in (15, 16, 17):           # only 30 ms voiced -> below the 60 ms floor
        f0[k] = 150.0
    wins = [(0.0, 0.05), (0.15, 0.17), (0.30, 0.32)]
    pre = _pre(f0, times, wins)
    assert orc.voiced_rhyme_window(pre, 1, min_ms=60.0) is None


def test_window_none_when_win_is_none():
    pre = _pre([150.0] * 10, [i * 0.01 for i in range(10)], [None, None])
    assert orc.voiced_rhyme_window(pre, 0) is None


# ============================== contour builders (pure) ==============================
def test_build_flip_contour_lands_at_target():
    # st_abs(t) = r_target + slope*t ; hz = 100*2^(st_abs/12). Median residual must reconstruct r_target.
    t0, t1, r_target, slope = 1.00, 1.20, 3.0, -1.0
    pts = orc.build_flip_contour(t0, t1, r_target, slope, step=0.01)
    assert pts and abs(pts[0][0] - t0) < 1e-9 and pts[-1][0] <= t1 + 1e-9
    for (t, hz) in pts:
        st = 12.0 * np.log2(hz / 100.0)
        assert abs((st - slope * t) - r_target) < 1e-6   # residual = st - slope*t == r_target


def test_orig_contour_pts_only_inside_window():
    times = [i * 0.01 for i in range(20)]
    f0 = [np.nan if k % 2 else 120.0 + k for k in range(20)]
    pre = _pre(f0, times, [(0.0, 0.2)])
    pts = orc._orig_contour_pts(pre, 0.05, 0.15)
    assert all(0.05 <= t < 0.15 for (t, _) in pts)
    assert all(hz == hz for (_, hz) in pts)              # no NaN carried through


# ============================== salience_check predicate ==============================
def _sal(**kw):
    base = dict(med_st_flipped=5.0, med_st_correct=0.0, r_flipped=2.0, mid_ref=0.0,
                theta_h=1.0, theta_l=1.0, pred_flipped="H", expect="H")
    base.update(kw)
    return B.salience_check(**base)


def test_salience_all_pass():
    r = _sal()
    assert r["passed"] and r["delta_ok"] and r["band_ok"] and r["pred_flip_ok"]
    assert abs(r["realized_st"] - 5.0) < 1e-9


def test_salience_fail_realized_delta():
    r = _sal(med_st_flipped=2.0)                # |2-0| = 2 st < 3 st floor
    assert not r["passed"] and not r["delta_ok"] and r["band_ok"] and r["pred_flip_ok"]


def test_salience_fail_band_H():
    r = _sal(r_flipped=0.5)                     # 0.5 < mid_ref(0)+theta_h(1) -> not in H band
    assert not r["passed"] and not r["band_ok"]


def test_salience_fail_band_L():
    r = _sal(expect="L", pred_flipped="L", r_flipped=-0.5)   # -0.5 > mid_ref-theta_l(-1) -> not in L band
    assert not r["passed"] and not r["band_ok"]


def test_salience_band_L_pass():
    r = _sal(expect="L", pred_flipped="L", med_st_flipped=-5.0, med_st_correct=0.0, r_flipped=-2.0)
    assert r["passed"]


def test_salience_fail_pred_not_flipped():
    r = _sal(pred_flipped="M")                  # classifier did NOT call it H
    assert not r["passed"] and not r["pred_flip_ok"]


def test_salience_none_inputs():
    assert not _sal(med_st_flipped=None)["passed"]
    assert not _sal(r_flipped=None)["passed"]


# ============================== score_psola (A/B) ==============================
def _km_pair(pid, side_map, fc, ff, dc=None, df=None):
    return dict(pair_id=pid, intended_text="ọkọ̀", side_map=side_map,
                tone_i2_frozen_correct=fc, tone_i2_frozen_flipped=ff,
                tone_i2_deployed_correct=(dc if dc is not None else fc),
                tone_i2_deployed_flipped=(df if df is not None else ff),
                condition="pair", is_catch=False, expect_pick=None, flip_dir="L->H", tbu_index=2)


def _km_catch(pid, side_map):
    return dict(pair_id=pid, intended_text="catch", side_map=side_map,
                tone_i2_frozen_correct=None, tone_i2_frozen_flipped=None,
                tone_i2_deployed_correct=None, tone_i2_deployed_flipped=None,
                condition="catch", is_catch=True, expect_pick=side_map, flip_dir="flatten", tbu_index=None)


def _build_keymap(n_pairs=26, n_catch=5, frozen_correct_higher=True, seed=0):
    """n_pairs A/B pairs (correct side alternates A/B) + n_catch catch trials. Frozen metric ranks correct >
    flipped on every pair when frozen_correct_higher."""
    rng = np.random.default_rng(seed)
    km, idx = {}, 1
    for j in range(n_pairs):
        side = "A" if j % 2 == 0 else "B"
        fc = 0.6 + 0.1 * rng.random()
        ff = fc - (0.1 if frozen_correct_higher else -0.1)
        km[f"item{idx:02d}"] = _km_pair(f"pair_{j}", side, round(fc, 4), round(ff, 4))
        idx += 1
    for j in range(n_catch):
        km[f"item{idx:02d}"] = _km_catch(f"catch_{j}", "A" if j % 2 == 0 else "B")
        idx += 1
    return km


def _answer_code(km, pair_acc=1.0, catch_acc=1.0, seed=1):
    """Build a PILOT2 answer string: pick the correct side for pair_acc fraction of pairs, catch_acc of catch."""
    rng = np.random.default_rng(seed)
    parts = ["PILOT2"]
    for iid, v in km.items():
        correct_side = v["expect_pick"] if v["is_catch"] else v["side_map"]
        acc = catch_acc if v["is_catch"] else pair_acc
        if rng.random() < acc:
            pick = correct_side
        else:
            pick = "B" if correct_side == "A" else "A"
        parts.append(f"{iid}={pick}")
    return ";".join(parts)


def test_metric_block_paired_win():
    km = _build_keymap(frozen_correct_higher=True)
    mb = S.metric_block(km)
    assert mb["frozen"]["win_rate"] == 1.0 and mb["frozen"]["wins"] == 26 and mb["frozen"]["losses"] == 0
    assert mb["frozen"]["sign_p"] < 0.001
    assert len(mb["margins"]) == 26 and all(m > 0 for m in mb["margins"].values())


def test_metric_block_paired_win_reversed():
    km = _build_keymap(frozen_correct_higher=False)
    mb = S.metric_block(km)
    assert mb["frozen"]["win_rate"] == 0.0 and mb["frozen"]["losses"] == 26


def test_score_rater_maps_sides_and_catch():
    km = _build_keymap(n_pairs=4, n_catch=4)
    # perfect rater: picks correct side everywhere
    code = _answer_code(km, pair_acc=1.0, catch_acc=1.0)
    r = S.score_rater("perfect", S.parse_compact(code), km)
    assert r["catch_ok"] == 4 and r["catch_tot"] == 4
    assert sum(r["pair_correct"].values()) == 4 and len(r["pair_correct"]) == 4
    # a rater who fails catch
    code2 = _answer_code(km, pair_acc=1.0, catch_acc=0.0)
    r2 = S.score_rater("badcatch", S.parse_compact(code2), km)
    assert r2["catch_ok"] == 0


def test_unsure_dropped():
    km = _build_keymap(n_pairs=3, n_catch=3)
    iids = [k for k, v in km.items() if not v["is_catch"]]
    code = "PILOT2;" + ";".join(f"{i}=unsure" for i in iids)
    r = S.score_rater("u", S.parse_compact(code), km)
    assert r["n_unsure"] == 3 and len(r["pair_correct"]) == 0


def test_binom_and_sign():
    # 26/26 correct is overwhelmingly significant; 13/26 is not.
    assert S.binom_two_sided_p(26, 26) < 1e-6
    assert S.binom_two_sided_p(13, 26) > 0.5
    assert S.sign_test_p(26, 0) < 1e-6


def test_point_biserial_positive():
    # margin strongly predicts human-correct -> positive r
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0, 0, 0, 1, 1, 1]
    assert S.point_biserial(x, y) > 0.7
    assert S.point_biserial([1, 1, 1], [0, 1, 0]) != S.point_biserial([1, 1, 1], [0, 1, 0]) or True  # const x -> nan
    assert np.isnan(S.point_biserial([1.0, 1.0, 1.0], [0, 1, 0]))


def _run_main(km, code=None, min_scoreable=24):
    """Run score_psola.main() in-process against a temp keymap, capturing stdout."""
    import io
    import contextlib
    d = tempfile.mkdtemp()
    kp = os.path.join(d, "keymap_psola.json")
    json.dump(km, open(kp, "w"), ensure_ascii=False)
    argv = ["score_psola.py", "--keymap", kp, "--min-scoreable", str(min_scoreable)]
    if code is not None:
        argv += ["--answers", code]
    old = sys.argv
    sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            S.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def test_main_perfect_rater():
    km = _build_keymap(n_pairs=26, n_catch=5, frozen_correct_higher=True)
    code = _answer_code(km, pair_acc=1.0, catch_acc=1.0)
    out = _run_main(km, code)
    assert "human accuracy = 100.0%" in out, out
    assert "paired-win FROZEN" in out and "win-rate = 100.0%" in out
    assert "AUROC = n/a" in out          # human got all trials right -> one class -> AUROC undefined
    assert "NO SCOREABLE PILOT" not in out


def test_main_catch_gate_discards():
    km = _build_keymap(n_pairs=26, n_catch=5)
    code = _answer_code(km, pair_acc=1.0, catch_acc=0.0)   # fails every catch
    out = _run_main(km, code)
    assert "CATCH SCREEN FAILED" in out and "NO SCOREABLE PILOT" in out


def test_main_too_few_scoreable_guard():
    km = _build_keymap(n_pairs=10, n_catch=5)              # only 10 pairs < 24
    code = _answer_code(km, pair_acc=1.0, catch_acc=1.0)
    out = _run_main(km, code)
    assert "too few scoreable trials" in out and "NO SCOREABLE PILOT" in out


def test_main_partial_rater_computes_accuracy():
    km = _build_keymap(n_pairs=26, n_catch=5, frozen_correct_higher=True)
    # human correct on exactly the pairs where the metric margin is positive -> should agree (AUROC high-ish)
    rng = np.random.default_rng(7)
    parts = ["PILOT2"]
    for iid, v in km.items():
        if v["is_catch"]:
            parts.append(f"{iid}={v['expect_pick']}")
            continue
        # 70% correct
        cs = v["side_map"]
        pick = cs if rng.random() < 0.7 else ("B" if cs == "A" else "A")
        parts.append(f"{iid}={pick}")
    out = _run_main(km, ";".join(parts))
    assert "KEYSTONE" in out and "human accuracy =" in out
    assert "NO SCOREABLE PILOT" not in out


def test_metric_only_no_raters():
    km = _build_keymap()
    out = _run_main(km, code=None)
    assert "metric-only read above" in out and "paired-win FROZEN" in out


# ============================== runner ==============================
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        npass += 1
    print(f"\nALL {npass} PURE-LOGIC TESTS PASSED")
