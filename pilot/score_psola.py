#!/usr/bin/env python3
# coding=utf-8
# score_psola.py — score the PSOLA-isolated tone test against KNOWN ground truth.
#
# WHY THIS IS A CLEANER READ THAN score_pilot.py
#   The pilot scorer validated tone_i2 against a human ✓/✗ on TTS clips with NO ground truth — so a chance AUROC
#   could not be blamed on the metric vs. the stimulus (tone was confounded with synthetic naturalness). Here the
#   ground truth is KNOWN BY CONSTRUCTION: every non-catch item is one twin of an artifact-matched pair whose only
#   difference is the F0 of one syllable. condition='correct' -> the tone is right (expect ✓); condition='flipped'
#   -> exactly one syllable's tone was pushed past the opposite band (expect ✗). So we can measure three things
#   that the pilot could not:
#     1. HUMAN PERCEPTION vs KNOWN GT  — can a native actually HEAR the F0 flip? (the stimulus-validity question)
#     2. METRIC SENSITIVITY (human-free) — the PAIRED oracle: is tone_i2(correct) > tone_i2(flipped) per pair?
#     3. HUMAN<->METRIC AGREEMENT — AUROC(tone_i2, human ✓/✗) over non-catch items [PRIMARY], kappa [secondary].
#
# GATE vs SUCCESS (the mistake the pilot scorer fixed, kept fixed here):
#   GATE (rater + STIMULUS validity): catch screen passes AND the human can hear the flips at all
#       (flipped-item accuracy clearly > chance). If the human can't hear the flips, the STIMULI failed
#       (flips inaudible / artifact) — which also invalidates the metric test. Raise --k and rebuild.
#   SUCCESS (metric validity, reported separately): AUROC >= 0.70 AND paired metric win-rate >= ~0.75.
#
# Reuses ALL stats from score_pilot.py (auroc, wilson_ci, cohen_kappa, parse_compact, parse_csv, JUDGE) — no
# re-implementation. Adds only an exact two-sided sign test for the paired oracle.
#
#   python score_psola.py --keymap keymap_psola.json                                   # metric-only (human-free)
#   python score_psola.py --keymap keymap_psola.json --answers "PILOT1;item01=ok;..."  # one rater
#   python score_psola.py --keymap keymap_psola.json --answers-file r1.txt r2.txt       # several raters

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from score_pilot import auroc, wilson_ci, cohen_kappa, parse_compact, parse_csv, JUDGE  # shared stats — DO NOT re-impl


# ----------------------------------------------------------------------------- paired sign test (the only new stat)
def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test, H0: P(correct>flipped)=0.5. Ties are dropped before calling.
    Returns the two-sided p (sum of outcomes no more probable than the observed)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) for i in range(n + 1)]
    pk = probs[wins]
    return min(1.0, sum(p for p in probs if p <= pk) / float(2 ** n))


# ----------------------------------------------------------------------------- metric-only block (no humans needed)
def metric_block(keymap):
    """Everything computable from the WITHHELD keymap alone: the paired oracle (tone_i2 correct vs flipped) and a
    threshold metric accuracy vs KNOWN GT. Human-free, so it always runs."""
    # pair the two non-catch twins by pair_id
    pairs = {}
    for v in keymap.values():
        if v["is_catch"]:
            continue
        pairs.setdefault(v["pair_id"], {})[v["condition"]] = v.get("tone_i2")
    wins = losses = ties = usable = 0
    for d in pairs.values():
        c, f = d.get("correct"), d.get("flipped")
        if c is None or f is None:
            continue
        usable += 1
        if c > f:
            wins += 1
        elif c < f:
            losses += 1
        else:
            ties += 1
    win_rate = ((wins + 0.5 * ties) / usable) if usable else float("nan")
    p = sign_test_p(wins, losses)

    # threshold metric accuracy vs GT: predict ✓ iff tone_i2 >= median(tone_i2 over scoreable non-catch items)
    vals = [v["tone_i2"] for v in keymap.values() if not v["is_catch"] and v.get("tone_i2") is not None]
    tau = float(np.median(vals)) if vals else float("nan")
    m_tot = m_cor = 0
    for v in keymap.values():
        if v["is_catch"] or v.get("tone_i2") is None:
            continue
        pred = "ok" if v["tone_i2"] >= tau else "bad"
        m_tot += 1
        m_cor += int(pred == v["expect"])
    return dict(usable_pairs=usable, wins=wins, losses=losses, ties=ties, win_rate=win_rate, sign_p=p,
                tau=tau, metric_acc=(m_cor / m_tot if m_tot else float("nan")), metric_n=m_tot)


# ----------------------------------------------------------------------------- one rater vs KNOWN GT
def score_rater(name, ans, keymap):
    catch_tot = catch_ok = 0
    # human-correct counters split by condition
    corr_tot = corr_ok = 0          # condition=='correct' items (GT expect ✓)
    flip_tot = flip_ok = 0          # condition=='flipped' items (GT expect ✗)
    n_unsure = 0
    pair_h, pair_t = [], []         # human binary (1=✓,0=✗) and continuous tone_i2, for AUROC/kappa (non-catch)
    for item, label in ans.items():
        km = keymap.get(item)
        if km is None:
            continue
        if km["is_catch"]:
            catch_tot += 1
            catch_ok += int(label == km["expect"])
            continue
        j = JUDGE.get(label)
        if j is None:
            n_unsure += 1
            continue
        if km["condition"] == "correct":
            corr_tot += 1
            corr_ok += int(label == km["expect"])    # expect 'ok'
        elif km["condition"] == "flipped":
            flip_tot += 1
            flip_ok += int(label == km["expect"])    # expect 'bad'
        ti2 = km.get("tone_i2")
        if ti2 is not None:
            pair_h.append(j)
            pair_t.append(float(ti2))
    return dict(name=name, catch_tot=catch_tot, catch_ok=catch_ok, corr_tot=corr_tot, corr_ok=corr_ok,
                flip_tot=flip_tot, flip_ok=flip_ok, n_unsure=n_unsure, pair_h=pair_h, pair_t=pair_t)


def load_raters(args):
    raters = []
    if args.answers:
        raters.append(("pasted", parse_compact(args.answers)))
    if args.csv:
        raters.append((args.csv, parse_csv(args.csv)))
    for p in args.answers_file or []:
        raters.append((p, parse_compact(open(p, encoding="utf-8").read())))
    return raters


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Score the PSOLA-isolated Yorùbá tone test against known GT.")
    ap.add_argument("--keymap", required=True)
    ap.add_argument("--answers", help='pasted compact code, e.g. "PILOT1;item01=ok;item02=bad;..."')
    ap.add_argument("--csv", help="responses.csv with columns item_id,answer")
    ap.add_argument("--answers-file", nargs="*", help="one or more files, each a pasted compact code")
    ap.add_argument("--catch-min", type=int, default=5, help="min correct catch items (of n_catch) to keep a rater")
    ap.add_argument("--min-scoreable", type=int, default=24, help="min pooled non-catch items for a real read")
    args = ap.parse_args()

    try:
        keymap = json.load(open(args.keymap, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"FATAL: cannot read keymap {args.keymap} ({e})")

    n_catch = sum(1 for v in keymap.values() if v["is_catch"])
    n_pairs = len({v["pair_id"] for v in keymap.values() if not v["is_catch"]})
    print(f"keymap: {len(keymap)} items "
          f"({sum(1 for v in keymap.values() if v.get('condition')=='correct')} correct, "
          f"{sum(1 for v in keymap.values() if v.get('condition')=='flipped')} flipped, "
          f"{n_catch} catch; {n_pairs} pairs)")
    print(f"mid_ref=None (deployed-metric register anchor) — same as the shipped tone_i2 / poster.")
    print("-" * 70)

    # ---------- METRIC-ONLY (human-free) — always available ----------
    mb = metric_block(keymap)
    print("METRIC SENSITIVITY (human-free, from the withheld keymap):")
    if mb["usable_pairs"]:
        print(f"  paired oracle  tone_i2(correct) > tone_i2(flipped): "
              f"{mb['wins']}/{mb['usable_pairs']} pairs (ties {mb['ties']})  "
              f"win-rate = {mb['win_rate']*100:.1f}%  sign-test p = {mb['sign_p']:.4f}")
    else:
        print("  paired oracle: no scoreable pairs (tone_i2 uncovered on too many twins).")
    if mb["metric_n"]:
        print(f"  threshold metric acc vs GT (✓ iff tone_i2>=median {mb['tau']:.3f}): "
              f"{mb['metric_acc']*100:.1f}%  (n={mb['metric_n']})")
    print("-" * 70)

    raters = load_raters(args)
    if not raters:
        print("no rater answers provided (--answers / --csv / --answers-file): metric-only read above.")
        print("provide a reviewer's pasted code to get the HUMAN perception + agreement read.")
        return

    # ---------- per-rater + catch gate ----------
    scored = [score_rater(name, ans, keymap) for name, ans in raters]
    kept = []
    for r in scored:
        passes = (r["catch_ok"] >= args.catch_min) and (r["catch_tot"] >= args.catch_min)
        flag = "KEEP" if passes else "DISCARD"
        print(f"[{r['name']}] catch {r['catch_ok']}/{r['catch_tot']}  "
              f"correct-items ✓ {r['corr_ok']}/{r['corr_tot']}  flipped-items ✗ {r['flip_ok']}/{r['flip_tot']}  "
              f"unsure {r['n_unsure']}  -> {flag}")
        if not passes:
            print(f"  CATCH SCREEN FAILED (<{args.catch_min}/{max(r['catch_tot'], n_catch)}) — discard, not engaged.")
        else:
            kept.append(r)

    if not kept:
        print("-" * 70)
        print("NO SCOREABLE PILOT — no rater passed the catch screen (report kit-released only).")
        return

    # ---------- pool kept raters ----------
    corr_ok = sum(r["corr_ok"] for r in kept); corr_tot = sum(r["corr_tot"] for r in kept)
    flip_ok = sum(r["flip_ok"] for r in kept); flip_tot = sum(r["flip_tot"] for r in kept)
    pair_h = [x for r in kept for x in r["pair_h"]]
    pair_t = [x for r in kept for x in r["pair_t"]]
    n_scored = len(pair_h)

    print("-" * 70)
    if n_scored < args.min_scoreable:
        print(f"pooled scoreable non-catch items = {n_scored} < {args.min_scoreable}")
        print("NO SCOREABLE PILOT — too few scoreable items (report kit-released only).")
        return

    oa_ok, oa_tot = corr_ok + flip_ok, corr_tot + flip_tot
    op, olo, ohi = wilson_ci(oa_ok, oa_tot)
    cp, clo, chi = wilson_ci(corr_ok, corr_tot)
    fp, flo, fhi = wilson_ci(flip_ok, flip_tot)
    print("HUMAN PERCEPTION vs KNOWN GT (chance = 50%):")
    print(f"  overall human accuracy   = {op*100:.1f}%  [Wilson 95% CI {olo*100:.1f}%, {ohi*100:.1f}%]  (n={oa_tot})")
    print(f"  on correct items (✓)     = {cp*100:.1f}%  [{clo*100:.1f}%, {chi*100:.1f}%]  (n={corr_tot})")
    print(f"  on flipped items (✗)     = {fp*100:.1f}%  [{flo*100:.1f}%, {fhi*100:.1f}%]  (n={flip_tot})  "
          f"<- can the native HEAR the flip?")

    # AUROC (PRIMARY) + kappa (secondary)
    hh = np.asarray(pair_h, int)
    tt = np.asarray(pair_t, float)
    auc = auroc(tt, hh)
    rng = np.random.default_rng(4242)
    aboot = []
    for _ in range(2000):
        idx = rng.integers(0, n_scored, n_scored)
        a = auroc(tt[idx], hh[idx])
        if a == a:
            aboot.append(a)
    aboot.sort()
    aulo, auhi = (aboot[int(0.025 * len(aboot))], aboot[int(0.975 * len(aboot))]) if aboot else (float("nan"),) * 2
    tau = float(np.median(tt))
    kappa = cohen_kappa(hh.tolist(), [1 if t >= tau else 0 for t in tt])

    print("-" * 70)
    print("HUMAN <-> METRIC AGREEMENT:")
    if auc == auc:
        print(f"  AUROC(tone_i2, human ✓/✗)  (PRIMARY) = {auc:.3f}  [95% CI {aulo:.3f}, {auhi:.3f}]  (chance=0.50)")
    else:
        print("  AUROC = n/a (human gave only one class).")
    print(f"  Cohen's kappa (secondary, median-binarized) = {kappa:.3f}")

    # ---------- gate vs success ----------
    print("-" * 70)
    flip_heard = (fp > 0.5 and flo > 0.5)               # CI lower bound clearly above chance
    gate_ok = flip_heard                                 # catch already enforced (only catch-passers in 'kept')
    print(f"STIMULUS/RATER GATE (catch passed AND flipped-acc CI > 50%): "
          f"{'PASS — flips are audible' if gate_ok else 'FAIL'}")
    if not flip_heard:
        print("  -> The native could NOT reliably hear the flips (flipped-acc ~ chance). The STIMULI failed "
              "(flip inaudible / artifact-dominated), which INVALIDATES the metric test. Increase --k and rebuild.")
    metric_pass = (auc == auc and auc >= 0.70 and mb["win_rate"] == mb["win_rate"] and mb["win_rate"] >= 0.75)
    if gate_ok:
        if auc != auc:
            verdict = "N/A — human gave only one class"
        elif metric_pass:
            verdict = "PASS — tone_i2 tracks perceived tone AND is paired-sensitive to the flip"
        else:
            verdict = (f"FAIL/INCONCLUSIVE — AUROC {auc:.2f} (need >=0.70) and/or paired win-rate "
                       f"{mb['win_rate']*100:.0f}% (need >=75%)")
        print(f"METRIC-VALIDITY SUCCESS (AUROC>=0.70 AND paired win-rate>=75%, after gate): {verdict}")
    else:
        print("METRIC-VALIDITY SUCCESS: not evaluable — stimulus/rater gate failed.")

    # ---------- poster line ----------
    print("-" * 70)
    print(f"psola-pilot: N={len(kept)}, {n_scored} items, "
          f"human flip-acc={fp*100:.0f}% [{flo*100:.0f}%,{fhi*100:.0f}%], "
          f"metric paired-win={mb['win_rate']*100:.0f}%, "
          f"AUROC(metric,human)={auc:.2f} [{aulo:.2f},{auhi:.2f}]")


if __name__ == "__main__":
    main()
