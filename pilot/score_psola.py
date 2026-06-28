#!/usr/bin/env python3
# coding=utf-8
# score_psola.py — score the PSOLA-isolated A/B tone test against KNOWN ground truth.
#
# THE A/B REBUILD (PSOLA_RECIPE.md §3/§4): each non-catch TRIAL is an artifact-matched PAIR of the SAME
# sentence — the CORRECT twin and the FLIPPED twin — with the correct side RANDOMIZED per trial (keymap
# side_map). The reviewer taps whichever side sounds like correct Yorùbá (A / B / — not sure). So we measure:
#   1. KEYSTONE — HUMAN %-correct on non-catch trials vs the 50% binomial baseline (Wilson 95% CI + exact
#      binomial p). The decisive "can a native HEAR the tone flip" / "does the metric track that" reference.
#   2. METRIC paired-win — fraction of pairs with tone_i2_frozen(correct) > tone_i2_frozen(flipped) (+ sign-test
#      p). Human-free. FROZEN mid_ref ONLY — the metric-core sensitivity column.
#   3. AGREEMENT — AUROC / point-biserial of the margin Δ = tone_i2_frozen(correct) − tone_i2_frozen(flipped)
#      vs human-correct(0/1), bootstrap CI.
#   + tone_i2_DEPLOYED paired-win, printed ONLY as the localized-blindness caveat (NEVER the agreement claim).
#
# GATE vs SUCCESS:
#   GATE (rater): catch ≥ 80% correct (else DISCARD the rater).
#   HONESTY guard: < --min-scoreable (24) scoreable non-catch trials OR catch < 80% -> "NO SCOREABLE PILOT".
#   SUCCESS (metric validity): paired-win(frozen) high AND AUROC(margin,human) clears chance — reported.
#
# Reuses ALL stats from score_pilot.py (auroc, wilson_ci, parse_compact, parse_csv) — no re-impl.
#
#   python score_psola.py --keymap keymap_psola.json                                   # metric-only (human-free)
#   python score_psola.py --keymap keymap_psola.json --answers "PILOT2;item01=A;..."   # one rater
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
from score_pilot import auroc, wilson_ci, parse_compact, parse_csv  # shared stats — DO NOT re-impl

# A/B answer -> pick string (1=A,2=B kept as the raw side); 'unsure' dropped.
PICKS = {"a": "A", "b": "B", "unsure": None}


# ----------------------------------------------------------------------------- exact two-sided binomial vs 0.5
def binom_two_sided_p(successes, n):
    """Exact two-sided binomial test, H0: p=0.5. Sum of outcome probabilities no more probable than observed."""
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) for i in range(n + 1)]
    pk = probs[successes]
    return min(1.0, sum(p for p in probs if p <= pk) / float(2 ** n))


def sign_test_p(wins, losses):
    """Exact two-sided sign test (ties dropped before calling), H0: P(correct>flipped)=0.5."""
    return binom_two_sided_p(wins, wins + losses)


def point_biserial(x, y01):
    """Pearson correlation between a continuous x (the margin) and a binary y (human-correct 0/1) = the
    point-biserial coefficient. nan if either is constant."""
    x = np.asarray(x, float)
    y = np.asarray(y01, float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------------------------------------------------------- metric-only block (human-free)
def metric_block(keymap):
    """The PAIRED oracle on both anchors, computable from the withheld keymap alone (no humans). FROZEN is the
    agreement column; DEPLOYED is the caveat. Returns per-pair margins keyed by pair_id (FROZEN) for AUROC."""
    def _winrate(field_c, field_f):
        wins = losses = ties = usable = 0
        for v in keymap.values():
            if v["is_catch"]:
                continue
            c, f = v.get(field_c), v.get(field_f)
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
        return dict(wins=wins, losses=losses, ties=ties, usable=usable, win_rate=win_rate,
                    sign_p=sign_test_p(wins, losses))
    frozen = _winrate("tone_i2_frozen_correct", "tone_i2_frozen_flipped")
    deployed = _winrate("tone_i2_deployed_correct", "tone_i2_deployed_flipped")
    margins = {v["pair_id"]: (v["tone_i2_frozen_correct"] - v["tone_i2_frozen_flipped"])
               for v in keymap.values()
               if not v["is_catch"]
               and v.get("tone_i2_frozen_correct") is not None and v.get("tone_i2_frozen_flipped") is not None}
    return dict(frozen=frozen, deployed=deployed, margins=margins)


# ----------------------------------------------------------------------------- one rater vs KNOWN GT
def score_rater(name, ans, keymap):
    """Map each A/B pick through the keymap. correct side = expect_pick (catch) or side_map (pair). Returns the
    catch screen counts + per-pair human-correct(0/1) keyed by pair_id (non-catch only)."""
    catch_tot = catch_ok = 0
    n_unsure = 0
    pair_correct = {}        # pair_id -> 1 if the rater picked the correct twin's side, else 0
    for item, label in ans.items():
        km = keymap.get(item)
        if km is None:
            continue
        pick = PICKS.get(str(label).strip().lower())
        if km["is_catch"]:
            catch_tot += 1
            if pick is not None and pick == km.get("expect_pick"):
                catch_ok += 1
            continue
        if pick is None:
            n_unsure += 1
            continue
        correct_side = km.get("side_map")     # the side holding the CORRECT twin
        pair_correct[km["pair_id"]] = int(pick == correct_side)
    return dict(name=name, catch_tot=catch_tot, catch_ok=catch_ok, n_unsure=n_unsure,
                pair_correct=pair_correct)


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
    ap = argparse.ArgumentParser(description="Score the PSOLA-isolated A/B Yorùbá tone test against known GT.")
    ap.add_argument("--keymap", required=True)
    ap.add_argument("--answers", help='pasted compact code, e.g. "PILOT2;item01=A;item02=B;..."')
    ap.add_argument("--csv", help="responses.csv with columns item_id,answer (answer in {A,B,unsure})")
    ap.add_argument("--answers-file", nargs="*", help="one or more files, each a pasted compact code")
    ap.add_argument("--catch-frac", type=float, default=0.80, help="min fraction of catch trials correct to keep a rater")
    ap.add_argument("--min-scoreable", type=int, default=24, help="min pooled non-catch trials for a real read")
    args = ap.parse_args()

    try:
        keymap = json.load(open(args.keymap, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"FATAL: cannot read keymap {args.keymap} ({e})")

    n_catch = sum(1 for v in keymap.values() if v["is_catch"])
    n_pairs = sum(1 for v in keymap.values() if not v["is_catch"])
    print(f"keymap: {len(keymap)} trials ({n_pairs} A/B pairs, {n_catch} catch)")
    print("metric↔human agreement is on tone_i2_FROZEN (mid_ref frozen from the clean clip); tone_i2_DEPLOYED "
          "(mid_ref=None, as shipped) documents the deployment-time localized-blindness gap only.")
    print("-" * 72)

    # ---------- METRIC-ONLY (human-free) — always available ----------
    mb = metric_block(keymap)
    fz, dp = mb["frozen"], mb["deployed"]
    print("METRIC SENSITIVITY (human-free, from the withheld keymap):")
    if fz["usable"]:
        print(f"  paired-win FROZEN   tone_i2(correct) > tone_i2(flipped): "
              f"{fz['wins']}/{fz['usable']} pairs (ties {fz['ties']})  "
              f"win-rate = {fz['win_rate']*100:.1f}%  sign-test p = {fz['sign_p']:.4f}   <- the agreement column")
    else:
        print("  paired-win FROZEN: no scoreable pairs (tone_i2_frozen uncovered on too many twins).")
    if dp["usable"]:
        print(f"  paired-win DEPLOYED (caveat only, NOT the agreement claim): "
              f"{dp['wins']}/{dp['usable']}  win-rate = {dp['win_rate']*100:.1f}%  sign p = {dp['sign_p']:.4f}")
    print("-" * 72)

    raters = load_raters(args)
    if not raters:
        print("no rater answers provided (--answers / --csv / --answers-file): metric-only read above.")
        print("provide a reviewer's pasted PILOT2 code to get the HUMAN keystone + agreement read.")
        return

    # ---------- per-rater + catch gate (≥80% correct) ----------
    scored = [score_rater(name, ans, keymap) for name, ans in raters]
    kept = []
    for r in scored:
        cfrac = (r["catch_ok"] / r["catch_tot"]) if r["catch_tot"] else 0.0
        passes = r["catch_tot"] > 0 and cfrac >= args.catch_frac
        flag = "KEEP" if passes else "DISCARD"
        print(f"[{r['name']}] catch {r['catch_ok']}/{r['catch_tot']} ({cfrac*100:.0f}%)  "
              f"scoreable pairs {len(r['pair_correct'])}  unsure {r['n_unsure']}  -> {flag}")
        if not passes:
            print(f"  CATCH SCREEN FAILED (<{args.catch_frac*100:.0f}%) — discard, not engaged.")
        else:
            kept.append(r)

    if not kept:
        print("-" * 72)
        print("NO SCOREABLE PILOT — no rater passed the catch screen (report kit-released only).")
        return

    # ---------- pool kept raters (per-trial human-correct + per-trial margin) ----------
    human, margin = [], []
    for r in kept:
        for pid, hc in r["pair_correct"].items():
            if pid in mb["margins"]:
                human.append(hc)
                margin.append(mb["margins"][pid])
    n_scored = len(human)

    print("-" * 72)
    if n_scored < args.min_scoreable:
        print(f"pooled scoreable non-catch trials = {n_scored} < {args.min_scoreable}")
        print("NO SCOREABLE PILOT — too few scoreable trials (report kit-released only).")
        return

    # ---------- KEYSTONE: human %-correct vs 50% ----------
    n_correct = int(sum(human))
    hp, hlo, hhi = wilson_ci(n_correct, n_scored)
    bp_p = binom_two_sided_p(n_correct, n_scored)
    print("KEYSTONE — HUMAN %-CORRECT on non-catch A/B trials (chance = 50%):")
    print(f"  human accuracy = {hp*100:.1f}%  [Wilson 95% CI {hlo*100:.1f}%, {hhi*100:.1f}%]  "
          f"(n={n_scored}, {n_correct} correct)  exact binomial p = {bp_p:.4f}")

    # ---------- AGREEMENT: AUROC + point-biserial of margin vs human-correct ----------
    hh = np.asarray(human, int)
    mm = np.asarray(margin, float)
    auc = auroc(mm, hh)
    pb = point_biserial(mm, hh)
    rng = np.random.default_rng(4242)
    aboot, pboot = [], []
    for _ in range(2000):
        idx = rng.integers(0, n_scored, n_scored)
        a = auroc(mm[idx], hh[idx])
        if a == a:
            aboot.append(a)
        b = point_biserial(mm[idx], hh[idx])
        if b == b:
            pboot.append(b)
    aboot.sort(); pboot.sort()
    aulo, auhi = (aboot[int(0.025 * len(aboot))], aboot[int(0.975 * len(aboot))]) if aboot else (float("nan"),) * 2
    pblo, pbhi = (pboot[int(0.025 * len(pboot))], pboot[int(0.975 * len(pboot))]) if pboot else (float("nan"),) * 2

    print("-" * 72)
    print("METRIC ↔ HUMAN AGREEMENT (FROZEN margin Δ = tone_i2_frozen[correct] − tone_i2_frozen[flipped]):")
    if auc == auc:
        print(f"  AUROC(margin, human-correct)  = {auc:.3f}  [95% CI {aulo:.3f}, {auhi:.3f}]  (chance=0.50)")
    else:
        print("  AUROC = n/a (human got every trial right or every trial wrong — one class only).")
    if pb == pb:
        print(f"  point-biserial r(margin, human-correct) = {pb:.3f}  [95% CI {pblo:.3f}, {pbhi:.3f}]")

    # ---------- poster line ----------
    print("-" * 72)
    auc_s = f"{auc:.2f} [{aulo:.2f},{auhi:.2f}]" if auc == auc else "n/a"
    print(f"psola-AB: N={len(kept)}, {n_scored} trials, "
          f"human %-correct={hp*100:.0f}% [{hlo*100:.0f}%,{hhi*100:.0f}%] vs 50% (p={bp_p:.3f}), "
          f"metric paired-win(frozen)={fz['win_rate']*100:.0f}%, "
          f"AUROC(margin,human)={auc_s}")


if __name__ == "__main__":
    main()
