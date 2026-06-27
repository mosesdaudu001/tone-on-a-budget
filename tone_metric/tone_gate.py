# tone_gate.py — Layer-B per-checkpoint TONE GATE (pure, no I/O, unit-testable).
#
# WHY THIS EXISTS
#   The native-speaker validation gate (old nb24/nb24b absolute per-syllable labeling) is RETIRED:
#   it scored 0.45 vs the same orthography a literate transcriber gets right, because absolute
#   per-syllable H/M/L judgement by ear collapses under downdrift. The AfriHuBERT tone probe is the
#   validated instrument instead (probe-vs-orthography 0.867 / kappa 0.798 on held-out REAL speech,
#   0.40 on tonally-wrong generated speech). This module turns that instrument into a population-level
#   PASS/FAIL gate that a checkpoint must clear before (and after) an A100 bake-off — with NO human in
#   the loop. nb26 calls tone_gate() instead of the dead native-kappa assert.
#
# WHAT IT GUARDS AGAINST (the cheap ways a TTS "passes" without actually getting tone right)
#   1. FLATTEN-TO-MID: Mid is the unmarked majority class, so a model that mumbles everything toward Mid
#      can score high MICRO-accuracy while destroying meaning. -> we gate per-class H AND L recall.
#   2. ABSTAIN-THE-HARD-ONES: the probe abstains on unaligned/short TBUs; a model could "win" by being
#      legible only on easy syllables. -> we gate per-class H AND L COVERAGE, not just recall-on-covered.
#   3. TIMBRE/CODEC CONFOUND: comparing real-gold vs base output confounds voice timbre with tone. The
#      floor here is a MATCHED-TIMBRE tone-shuffled anchor (SHUF): the SAME checkpoint generating the
#      SAME eval texts with the tone diacritics permuted. Same timbre, same vocoder, only tone wrong ->
#      (acc - shuf_acc) isolates the tone axis.
#   4. SINGLE-INSTRUMENT BLIND SPOT: a second, method-independent instrument (I2 = F0-residual) can be
#      added; tone_gate() then also requires inter-instrument agreement (Cohen kappa) as corroboration.
#      I2 is OPTIONAL — pass required=("I1",) to gate on the probe alone (v1), required=("I1","I2") once
#      the F0 instrument and its independent aligner are calibrated.
#
# INPUT CONTRACT
#   Each "score dict" is exactly what tone_probe.probe_score() / tone_eval_v2.score_from_precomputed()
#   return for ONE clip. The fields this module reads:
#       pred   : list[str|None]   per-TBU predicted class in {"H","M","L"} or None (abstained)
#       target : list[str]        per-TBU orthographic (lexical) tone in {"H","M","L"}
#   All other fields (accuracy, coverage, slope, ...) are ignored here — we RE-aggregate from the raw
#   per-TBU pred/target across the whole population, because averaging per-clip accuracies is wrong
#   (clips have different TBU counts) and per-class metrics need the raw confusion, not per-clip means.
#
# This module is PURE PYTHON (no numpy/torch) so it imports and unit-tests anywhere, including the
# CPU-only audio_env. No model loading, no S3, no notebook coupling.

CLASSES = ("H", "M", "L")


# ----------------------------- numeric helpers -----------------------------
def _ge(x, thr):
    """x >= thr, but NaN/None/non-numeric -> False (a metric we couldn't compute never passes)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    if x != x:  # NaN
        return False
    return x >= float(thr)


def _safe_div(num, den):
    return (num / den) if den else float("nan")


# ----------------------------- population aggregation -----------------------------
def aggregate(scores):
    """Aggregate a list of per-clip score dicts into population tone metrics.

    Returns dict:
        micro_acc   : correct / scored  across ALL TBUs in the population (NaN if nothing scored)
        coverage    : scored  / total   TBUs (how many TBUs the instrument was willing to label)
        n_tbu, n_scored, n_correct
        per_class   : {c: {support, covered, correct, recall, coverage}} for c in H/M/L
                        recall   = correct / covered   (accuracy ON the TBUs of class c it labeled)
                        coverage = covered / support   (fraction of class-c TBUs it was willing to label)
        confusion   : {true: {pred: count}}  over labeled TBUs only
    """
    n_tbu = n_scored = n_correct = 0
    cls = {c: {"support": 0, "covered": 0, "correct": 0} for c in CLASSES}
    confusion = {a: {b: 0 for b in CLASSES} for a in CLASSES}

    for s in scores:
        pred = s.get("pred") or []
        target = s.get("target") or []
        for p, t in zip(pred, target):
            if t not in CLASSES:
                continue  # defensive: skip anything not a real tone label
            n_tbu += 1
            cls[t]["support"] += 1
            if p is None:
                continue  # abstained -> counts against coverage, not against recall
            n_scored += 1
            cls[t]["covered"] += 1
            if p in CLASSES:
                confusion[t][p] += 1
            if p == t:
                n_correct += 1
                cls[t]["correct"] += 1

    per_class = {}
    for c in CLASSES:
        sup, cov, cor = cls[c]["support"], cls[c]["covered"], cls[c]["correct"]
        per_class[c] = {
            "support": sup,
            "covered": cov,
            "correct": cor,
            "recall": _safe_div(cor, cov),
            "coverage": _safe_div(cov, sup),
        }

    return {
        "micro_acc": _safe_div(n_correct, n_scored),
        "coverage": _safe_div(n_scored, n_tbu),
        "n_tbu": n_tbu,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "per_class": per_class,
        "confusion": confusion,
    }


# ----------------------------- inter-instrument agreement -----------------------------
def cohen_kappa(pairs, classes=CLASSES):
    """Cohen's kappa over a list of (a, b) class-label pairs. NaN if empty."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(classes)}
    a_count = [0] * len(classes)
    b_count = [0] * len(classes)
    obs = 0
    for a, b in pairs:
        if a not in idx or b not in idx:
            continue
        if a == b:
            obs += 1
        a_count[idx[a]] += 1
        b_count[idx[b]] += 1
    po = obs / n
    pe = sum((a_count[i] / n) * (b_count[i] / n) for i in range(len(classes)))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def cocovered_pairs(scores_a, scores_b, require_same_target=True):
    """Per-TBU (pred_a, pred_b) pairs where BOTH instruments labeled the SAME TBU.

    Assumes scores_a[i] and scores_b[i] are the SAME clip in the SAME order. By default also requires
    the two instruments to share the same orthographic target for that clip (they should, since target
    is derived from the same text) — this catches accidental misalignment between the two score lists.
    Returns (pairs, n_clips_skipped). A clip is skipped (not silently zipped) if lengths mismatch.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"instrument score lists differ in length: {len(scores_a)} vs {len(scores_b)}")
    pairs = []
    skipped = 0
    for sa, sb in zip(scores_a, scores_b):
        pa, ta = sa.get("pred") or [], sa.get("target") or []
        pb, tb = sb.get("pred") or [], sb.get("target") or []
        if require_same_target and ta != tb:
            skipped += 1
            continue
        if len(pa) != len(pb):
            skipped += 1
            continue
        for a, b in zip(pa, pb):
            if a is not None and b is not None:
                pairs.append((a, b))
    return pairs, skipped


# ----------------------------- the gate -----------------------------
# Layer-B thresholds (YORUBA_WAY_FORWARD Phase-0c / nb26 entry). acc_min is per-instrument: the F0
# instrument (I2) is allowed a slightly lower bar than the SSL probe (I1) because raw-pitch labeling is
# noisier than the learned probe. Everything else is shared.
DEFAULT_THRESHOLDS = {
    "acc_min": {"I1": 0.72, "I2": 0.68},  # per-instrument; "_default" used for unlisted names
    "acc_min_default": 0.70,
    "cov_min": 0.70,
    "hl_recall_min": 0.60,   # per-class H and L recall (kills flatten-to-Mid)
    "hl_cov_min": 0.55,      # per-class H and L coverage (kills abstain-the-hard-ones)
    "floor_margin": 0.15,    # (acc - shuf_acc) above the matched-timbre tone-shuffled floor
    "kappa_min": 0.45,       # inter-instrument corroboration (only when >=2 instruments required)
}


def _acc_min_for(th, inst):
    am = th.get("acc_min")
    if isinstance(am, dict):
        return am.get(inst, th.get("acc_min_default", 0.70))
    return am if am is not None else th.get("acc_min_default", 0.70)


def tone_gate(gen, shuf, kappa_pairs=None, thresholds=None, required=("I1",)):
    """Layer-B PASS/FAIL for one checkpoint.

    gen, shuf : dict instrument_name -> list of per-clip score dicts.
                'gen'  = the checkpoint generating the frozen eval texts (correct orthography).
                'shuf' = the SAME checkpoint generating the SAME texts with tone diacritics PERMUTED,
                         scored against the CORRECT orthography (the matched-timbre tone floor).
    kappa_pairs : optional list of (pred_I1, pred_I2) co-covered pairs (see cocovered_pairs). Required
                  only when len(required) >= 2.
    required    : instruments that must independently pass. ("I1",) = probe-only gate (v1);
                  ("I1","I2") = dual-instrument gate once the F0 instrument is calibrated.

    Returns a verdict dict with a top-level boolean "pass", every sub-check, and the aggregates, so the
    caller can log exactly WHY a checkpoint failed.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    overall = True
    per_instrument = {}
    for inst in required:
        if inst not in gen or inst not in shuf:
            per_instrument[inst] = {"error": "missing scores", "pass": False}
            overall = False
            continue
        g = aggregate(gen[inst])
        f = aggregate(shuf[inst])
        margin = (g["micro_acc"] - f["micro_acc"]) if (g["micro_acc"] == g["micro_acc"]
                                                        and f["micro_acc"] == f["micro_acc"]) else float("nan")
        checks = {
            "acc": _ge(g["micro_acc"], _acc_min_for(th, inst)),
            "coverage": _ge(g["coverage"], th["cov_min"]),
            "H_recall": _ge(g["per_class"]["H"]["recall"], th["hl_recall_min"]),
            "L_recall": _ge(g["per_class"]["L"]["recall"], th["hl_recall_min"]),
            "H_coverage": _ge(g["per_class"]["H"]["coverage"], th["hl_cov_min"]),
            "L_coverage": _ge(g["per_class"]["L"]["coverage"], th["hl_cov_min"]),
            "floor_margin": _ge(margin, th["floor_margin"]),
        }
        inst_pass = all(checks.values())
        per_instrument[inst] = {
            "pass": inst_pass,
            "checks": checks,
            "margin": margin,
            "gen": g,
            "shuf": f,
            "acc_min": _acc_min_for(th, inst),
        }
        overall = overall and inst_pass

    kappa = None
    kappa_pass = True
    if len(required) >= 2:
        if kappa_pairs is None:
            kappa_pass = False  # a dual-instrument gate with no agreement evidence cannot pass
        else:
            kappa = cohen_kappa(kappa_pairs)
            kappa_pass = _ge(kappa, th["kappa_min"])
        overall = overall and kappa_pass

    return {
        "pass": bool(overall),
        "required": list(required),
        "per_instrument": per_instrument,
        "kappa": kappa,
        "kappa_pass": kappa_pass,
        "thresholds": th,
    }


def explain(verdict):
    """Human-readable one-block summary of a tone_gate() verdict (for notebook logs)."""
    lines = [f"TONE GATE: {'PASS' if verdict['pass'] else 'FAIL'}  (required={verdict['required']})"]
    for inst, r in verdict["per_instrument"].items():
        if "error" in r:
            lines.append(f"  [{inst}] ERROR: {r['error']}")
            continue
        g = r["gen"]
        pc = g["per_class"]
        lines.append(
            f"  [{inst}] {'pass' if r['pass'] else 'FAIL'}  "
            f"acc={g['micro_acc']:.3f}(>= {r['acc_min']})  cov={g['coverage']:.3f}  "
            f"margin={r['margin']:.3f}")
        lines.append(
            f"        H: rec={pc['H']['recall']:.3f} cov={pc['H']['coverage']:.3f} (n={pc['H']['support']}) | "
            f"L: rec={pc['L']['recall']:.3f} cov={pc['L']['coverage']:.3f} (n={pc['L']['support']})")
        failed = [k for k, v in r["checks"].items() if not v]
        if failed:
            lines.append(f"        failed: {', '.join(failed)}")
    if verdict["kappa"] is not None or len(verdict["required"]) >= 2:
        kv = verdict["kappa"]
        lines.append(f"  kappa(I1,I2)={'n/a' if kv is None else f'{kv:.3f}'}  "
                     f"{'pass' if verdict['kappa_pass'] else 'FAIL'}")
    return "\n".join(lines)
