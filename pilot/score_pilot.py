#!/usr/bin/env python3
# coding=utf-8
# score_pilot.py — score one reviewer's pasted answers against the WITHHELD tone_i2 keymap.
#
# WHAT IT DOES (one rater at a time; run once per pasted answer block):
#   * parses the compact code  "PILOT1;item03=ok;item07=bad;item12=unsure;..."  OR a responses.csv (item_id,answer)
#   * maps ok->human says "tone correct" (1), bad->"tone wrong" (0), unsure-> DROPPED (its own category, see below)
#   * CATCH SCREEN     : catch items carry a known expected answer; <5/6 consistent -> loud "discard this rater"
#   * NATIVE ANCHOR    : % of real native clips the human marked ✓ (should be high — it is the known-correct read)
#   * AUROC  (PRIMARY) : threshold-free. Does tone_i2 RANK the clips a human heard as tone-correct above the
#                        ones heard as wrong? = P(tone_i2[human✓] > tone_i2[human✗]), ties=0.5, chance=0.50.
#                        This is the headline validation statistic: it needs NO per-clip cutoff, so it is immune
#                        to the native-anchor-median artifact that biases kappa (see below). Standard way to
#                        validate a CONTINUOUS metric (tone_i2 is continuous balanced-recall) vs binary labels.
#   * anchor accuracy  : human ✓/✗ accuracy on the known-correct native anchors + Wilson 95% CI (chance ≈ 50%).
#   * kappa (secondary): Cohen's kappa after binarizing tone_i2 at TAU = median native-anchor tone_i2. Reported
#                        for transparency only; because natives are split at their OWN median and stay in the
#                        pool, ~half are forced to disagree -> kappa is biased DOWNWARD (conservative, never
#                        inflated). Do not headline kappa; use AUROC. (unsure dropped from both, documented.)
#   * guard            : <30 scoreable non-catch items OR a failed catch screen -> prints
#                        "NO SCOREABLE PILOT — report kit-released only" instead of a headline number.
#   * one poster line  : pilot: N=<raters>, <m> items, AUROC=<x> [CI], anchor ✓-acc=<y>% [CI]  (kappa secondary)
#
# Pure numpy/stdlib (Wilson CI + Cohen's kappa implemented here). sklearn is NOT required.
#
#   python score_pilot.py --answers "PILOT1;item01=ok;item02=bad;..." --keymap keymap.json
#   python score_pilot.py --csv responses.csv --keymap keymap.json
#   python score_pilot.py --answers-file ans1.txt ans2.txt --keymap keymap.json   # >1 rater -> N raters + IRR

import argparse
import csv
import json
import math
import sys

import numpy as np

# human label -> "tone is correct" judgment (1), "tone is wrong" (0), or None (dropped)
JUDGE = {"ok": 1, "bad": 0, "unsure": None}


# ----------------------------------------------------------------------------- stats (no sklearn)
def wilson_ci(k, n, z=1.95996):
    """Wilson score 95% CI for a binomial proportion k/n. Returns (phat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length binary label lists a,b in {0,1}. Direct (no sklearn)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    # expected agreement from the marginals
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0          # degenerate (one class) -> undefined; report as perfect/none
    return (po - pe) / (1 - pe)


def auroc(scores, labels):
    """Threshold-free AUROC = P(score[label==1] > score[label==0]), ties=0.5 (Mann–Whitney U / n_pos·n_neg).
    scores: continuous metric (tone_i2); labels: 1=human tone-correct, 0=wrong. nan if either class is empty.
    Immune to any TAU cutoff, so unaffected by the native-anchor-median artifact that depresses Cohen's kappa."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average (1-based) ranks, tie-corrected
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), float)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0     # mean rank of the tie block
        i = j + 1
    ranks = np.empty(n, float)
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------------- parsing
def parse_compact(code):
    out = {}
    for tok in code.strip().split(";"):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue           # skips the "PILOT1" header and stray tokens
        item, val = tok.split("=", 1)
        out[item.strip()] = val.strip().lower()
    return out


def parse_csv(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = (row.get("item_id") or "").strip()
            val = (row.get("answer") or "").strip().lower()
            if item and val:
                out[item] = val
    return out


def load_rater(args):
    raters = []
    if args.answers:
        raters.append(("pasted", parse_compact(args.answers)))
    if args.csv:
        raters.append((args.csv, parse_csv(args.csv)))
    for p in args.answers_file or []:
        txt = open(p, encoding="utf-8").read()
        raters.append((p, parse_compact(txt)))
    return raters


# ----------------------------------------------------------------------------- scoring one rater
def tone_threshold(keymap, override):
    if override is not None:
        return override, f"--tone-threshold {override}"
    nat = [v["tone_i2"] for v in keymap.values()
           if v["source"] == "native" and v.get("tone_i2") is not None]
    if not nat:
        return 0.58, ("fallback 0.58 (no native anchor in keymap) — CAUTION: 0.58 is a PASS-POOLED native "
                      "number (nb14), not a per-clip cutoff; per-clip tone_i2 is a noisier estimator on a "
                      "different distribution, so this binarization is only a degraded last resort")
    tau = float(np.median(nat))
    return tau, f"median native-anchor tone_i2 = {tau:.3f} (n_native={len(nat)})"


def score_one(name, ans, keymap, tau):
    """Returns a dict of per-rater stats. ans: item_id -> human label string."""
    catch_total = catch_ok = 0
    nat_total = nat_tick = 0
    pair_h, pair_m, pair_t = [], [], []   # human binary, tone_i2 binary (for kappa), tone_i2 continuous (for AUROC)
    n_unsure = 0

    for item, label in ans.items():
        km = keymap.get(item)
        if km is None:
            continue
        j = JUDGE.get(label)
        if km["is_catch"]:
            catch_total += 1
            exp = km.get("catch_expected")          # 'ok' (native dup) or 'bad' (degraded)
            if exp is not None and label == exp:
                catch_ok += 1
            elif exp is None and label == "ok":      # legacy catch w/o expected -> assume native dup
                catch_ok += 1
            continue
        if j is None:
            n_unsure += 1
            continue                                 # unsure dropped from anchor-acc + kappa (documented)
        if km["source"] == "native":
            nat_total += 1                           # denominator = native items with a definite ✓/✗
            if j == 1:
                nat_tick += 1
        ti2 = km.get("tone_i2")
        if ti2 is None:
            continue                                 # uncovered clip -> no metric label -> not scoreable for kappa
        pair_h.append(j)
        pair_m.append(1 if ti2 >= tau else 0)
        pair_t.append(float(ti2))

    return dict(name=name, catch_total=catch_total, catch_ok=catch_ok,
                nat_total=nat_total, nat_tick=nat_tick, n_unsure=n_unsure,
                pair_h=pair_h, pair_m=pair_m, pair_t=pair_t, n_scored=len(pair_h))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Score a Yorùbá tone pilot rater against the withheld keymap.")
    ap.add_argument("--keymap", required=True)
    ap.add_argument("--answers", help='pasted compact code, e.g. "PILOT1;item01=ok;item02=bad;..."')
    ap.add_argument("--csv", help="responses.csv with columns item_id,answer")
    ap.add_argument("--answers-file", nargs="*", help="one or more files each holding a pasted compact code")
    ap.add_argument("--tone-threshold", type=float, default=None,
                    help="override TAU (tone_i2 >= TAU => tone-correct); default = native-anchor median")
    ap.add_argument("--catch-min", type=int, default=5,
                    help="min consistent catch items (of 6) to keep a rater. MUST be <= the build's --n-catch: "
                         "a rater with fewer than catch-min catch items answered is auto-DISCARDed even if every "
                         "answered catch is correct (conservative by design — discards rather than fabricates).")
    args = ap.parse_args()

    try:
        keymap = json.load(open(args.keymap, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"FATAL: cannot read keymap {args.keymap} ({e})")

    raters = load_rater(args)
    if not raters:
        sys.exit("FATAL: provide --answers, --csv, or --answers-file.")

    tau, tau_doc = tone_threshold(keymap, args.tone_threshold)
    n_catch_total = sum(1 for v in keymap.values() if v["is_catch"])
    print(f"keymap: {len(keymap)} items "
          f"({sum(1 for v in keymap.values() if v['source']=='native')} native, "
          f"{sum(1 for v in keymap.values() if v['source']=='model0h')} model0h, "
          f"{sum(1 for v in keymap.values() if v['source']=='model5h')} model5h, "
          f"{n_catch_total} catch)")
    print(f"tone_i2 -> binary rule: tone-correct iff tone_i2 >= TAU; TAU = {tau:.3f}  [{tau_doc}]")
    print(f"unsure handling: DROPPED from kappa + anchor-accuracy (reported separately).")
    print("note: TAU is the native-anchor MEDIAN, and the native anchors are ALSO in the kappa pool, so ~half of")
    print("      the known-correct natives fall below their own median and are labelled tone-wrong by the metric.")
    print("      This injects forced human-vs-metric disagreement -> kappa is biased CONSERVATIVE (downward, never")
    print("      inflated). Treat the kappa CI as a floor; for a model-only read see per-source counts above.")
    print("-" * 64)

    scored = [score_one(name, ans, keymap, tau) for name, ans in raters]

    # pooled across all kept (catch-passing) raters
    kept = []
    for r in scored:
        catch_pass = (r["catch_ok"] >= args.catch_min) if r["catch_total"] >= args.catch_min else False
        flag = "KEEP" if catch_pass else "DISCARD"
        print(f"[{r['name']}] catch {r['catch_ok']}/{r['catch_total']}  "
              f"native ✓ {r['nat_tick']}/{r['nat_total']}  scoreable(non-catch) {r['n_scored']}  "
              f"unsure {r['n_unsure']}  -> {flag}")
        if not catch_pass:
            print(f"  ⚠️  CATCH SCREEN FAILED ({r['catch_ok']}/{r['catch_total']} < {args.catch_min}/"
                  f"{max(r['catch_total'], n_catch_total)}) — DISCARD this rater as not-engaged. Do not count it.")
        else:
            kept.append(r)

    if not kept:
        print("-" * 64)
        print("NO SCOREABLE PILOT — report kit-released only  (no rater passed the catch screen)")
        return

    pair_h = [x for r in kept for x in r["pair_h"]]
    pair_m = [x for r in kept for x in r["pair_m"]]
    pair_t = [x for r in kept for x in r["pair_t"]]
    nat_tick = sum(r["nat_tick"] for r in kept)
    nat_total = sum(r["nat_total"] for r in kept)
    n_scored = len(pair_h)

    print("-" * 64)
    if n_scored < 30:
        print(f"scoreable non-catch items (pooled) = {n_scored} < 30")
        print("NO SCOREABLE PILOT — report kit-released only  (too few scoreable items)")
        return

    kappa = cohen_kappa(pair_h, pair_m)
    # kappa CI via clip bootstrap over the paired items (percentile)
    rng = np.random.default_rng(4242)
    hh, mm = np.asarray(pair_h), np.asarray(pair_m)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n_scored, n_scored)
        k = cohen_kappa(hh[idx], mm[idx])
        if k == k:
            boots.append(k)
    boots.sort()
    klo, khi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]) if boots else (float("nan"),) * 2

    aphat, alo, ahi = wilson_ci(nat_tick, nat_total)
    agree = float(np.mean(hh == mm))

    # AUROC (PRIMARY) — threshold-free rank concordance of tone_i2 with the human judgment.
    tt = np.asarray(pair_t, float)
    auc = auroc(tt, hh)
    aboot = []
    for _ in range(2000):
        idx = rng.integers(0, n_scored, n_scored)
        a = auroc(tt[idx], hh[idx])
        if a == a:
            aboot.append(a)
    aboot.sort()
    aulo, auhi = (aboot[int(0.025 * len(aboot))], aboot[int(0.975 * len(aboot))]) if aboot else (float("nan"),) * 2

    print(f"raters kept (catch-passing)              = {len(kept)}")
    print(f"pooled scoreable non-catch items         = {n_scored}")
    if auc == auc:
        print(f"AUROC tone_i2 vs human  (PRIMARY)        = {auc:.3f}  [95% CI {aulo:.3f}, {auhi:.3f}]  (chance = 0.50, threshold-free)")
    else:
        print("AUROC tone_i2 vs human  (PRIMARY)        = n/a  (human gave only one class — all ✓ or all ✗)")
    print(f"anchor ✓-accuracy (native known-correct) = {aphat*100:.1f}%  "
          f"[Wilson 95% CI {alo*100:.1f}%, {ahi*100:.1f}%]  (chance ≈ 50%, n={nat_total})")
    print(f"raw human/tone_i2 agreement              = {agree*100:.1f}%")
    print(f"Cohen's kappa (secondary, TAU-binarized) = {kappa:.3f}  [95% CI {klo:.3f}, {khi:.3f}]  (conservative; do not headline)")
    print("-" * 64)
    # preregistered read. CORRECTION (transparent): the original rule was "AUROC >= 0.70 OR anchor-acc >= 75%".
    # That OR was a SPECIFICATION ERROR — anchor-acc and the catch screen measure whether the LISTENER is
    # engaged/competent (a rater-validity GATE), NOT whether the METRIC is valid. Metric validity is AUROC
    # ALONE. Conflating them lets a pilot "pass" on rater engagement while the metric tracks nothing. Corrected:
    #   GATE (rater valid): catch >= catch_min/6  AND  anchor ✓-acc >= 75%   -> only then is the rater usable.
    #   SUCCESS (metric):   AUROC >= 0.70 (rater gate must pass first).
    gate_ok = (aphat >= 0.75)        # catch already enforced upstream (only catch-passing raters reach 'kept')
    print(f"rater-validity GATE (catch>={args.catch_min}/6 AND anchor ✓-acc>=75%): "
          f"{'PASS — listener engaged' if gate_ok else 'FAIL — listener not engaged; do not use this rater'}")
    if auc == auc:
        if not gate_ok:
            verdict = "N/A — rater failed the validity gate"
        elif auc >= 0.70:
            verdict = "PASS — preliminary human evidence the metric tracks perceived tone"
        else:
            verdict = ("INCONCLUSIVE/NEGATIVE — metric did NOT track this listener (AUROC ≈ chance); "
                       "underpowered (N=1) AND the ✓/✗-on-TTS task confounds tone with synthetic naturalness")
        print(f"metric-validity READ (AUROC>=0.70, after gate): {verdict}")
    print(f"pilot: N={len(kept)}, {n_scored} items, "
          f"AUROC={auc:.2f} [{aulo:.2f},{auhi:.2f}], "
          f"anchor ✓-acc={aphat*100:.0f}% [{alo*100:.0f}%,{ahi*100:.0f}%]  (kappa={kappa:.2f} secondary)")


if __name__ == "__main__":
    main()
