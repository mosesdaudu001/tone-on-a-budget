# tone_layer0.py — Layer-0 INSTRUMENT-VALIDATION gate (pure python, no I/O, unit-testable).
#
# WHY THIS EXISTS (and why it is NOT tone_gate.py)
#   tone_gate.py is the Layer-B per-CHECKPOINT gate: "is THIS tts checkpoint tonally correct?". It trusts
#   the instruments. Layer-0 is the prior question: "are the instruments themselves trustworthy enough to
#   STAND IN for the native speaker we no longer have?". nb27 validates the two meters (I1 = AfriHuBERT
#   probe, I2 = F0-residual) on REAL held-out diacritized speech and writes the PASS verdict nb26 reads.
#
#   An adversarial review of the first draft found THREE load-bearing statistical defects that this module
#   fixes. They are subtle and they each silently break the gate, so they are spelled out:
#
#   (1) RAW ACCURACY IS THE WRONG STATISTIC. Yoruba is ~65% Mid tone. Raw micro-accuracy is dominated by
#       the easy Mid class: a model that flattens H and L toward Mid (destroying meaning) still scores
#       ~0.87 micro-acc. WORSE, raw acc makes "chance" depend on the tone mix — a label-PERMUTATION
#       control lands at sum_c p_c^2 ~= 0.48, so a "<= 0.40" raw ceiling FALSE-FAILS a perfect meter.
#       FIX: every accuracy here is BALANCED accuracy = mean per-class recall. Chance is ~1/3 regardless
#       of the tone mix, and flatten-to-Mid is exposed (its H/L recall is low so balanced-acc drops).
#
#   (2) MINORITY-CLASS FLOORS NEED SUPPORT + CONFIDENCE INTERVALS. H and L are ~17% each. A point-estimate
#       recall floor on thin support rejects GOOD models by luck (at n_H=40 a true 0.72 recall busts a 0.65
#       floor 16% of the time). FIX: a hard min-support PRECONDITION (verdict = INDETERMINATE, never PASS,
#       below it) plus bootstrap lower-confidence-bound floors (resample CLIPS, not TBUs, to respect within-
#       clip correlation).
#
#   (3) PSOLA MONOTONICITY MUST BE A TEST, NOT AN EYEBALL. "detection rises with flip magnitude k" is
#       asserted with a one-sided permutation Spearman trend test, not by reading three numbers.
#
#   Also: kappa(I1,I2) is CORROBORATIVE, not gating, in v1 — both meters share the MMS-CTC aligner, so a
#   shared misalignment makes them agree on the wrong syllable and inflates kappa. It becomes a hard gate
#   only once I2 runs an independent aligner. Default required=("I1",); kappa is logged.
#
# INPUT CONTRACT — identical to tone_gate.py: a "score dict" is one clip's tone_probe.probe_score() or
#   tone_f0_abs.score_abs_*() output. We read pred (list[str|None], H/M/L) and target (list[str], H/M/L)
#   and RE-AGGREGATE from raw per-TBU pairs via tone_gate.aggregate (averaging per-clip accuracies is
#   wrong — clips have different TBU counts). Everything is pure python so it tests on CPU-only audio_env.

import random

try:
    from . import tone_gate as tg
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_gate as tg

CLASSES = tg.CLASSES                 # ("H","M","L"); labels are strings, order is cosmetic here
aggregate = tg.aggregate            # re-export so nb27 has a single import surface
cohen_kappa = tg.cohen_kappa
cocovered_pairs = tg.cocovered_pairs
_ge = tg._ge
_safe_div = tg._safe_div


# ============================== population statistics ==============================
def balanced_accuracy(agg):
    """Mean per-class recall over classes that the instrument actually LABELED (covered > 0).

    This is the prior-free accuracy: chance ~= 1/(#present classes) regardless of the tone mix, and a
    flatten-to-Mid model is penalized because its H/L recall collapses. A class with support but zero
    covered TBUs contributes recall NaN -> it is excluded HERE and caught separately by the coverage
    floor (so abstaining on H/L cannot inflate this number into a pass on its own)."""
    recs = [agg["per_class"][c]["recall"] for c in CLASSES if agg["per_class"][c]["covered"] > 0]
    recs = [r for r in recs if r == r]                       # drop NaN
    return sum(recs) / len(recs) if recs else float("nan")


def weighted_f1(agg):
    """Support-weighted macro F1 over LABELED TBUs, from tone_gate.aggregate's confusion. Matches
    tone_probe.weighted_f1 semantics — the ONLY Layer-0 number with a validated nb22 reference
    (SLR86 wF1 >= 0.80), so it is kept as an I1 sub-check alongside balanced accuracy."""
    conf, pc = agg["confusion"], agg["per_class"]
    total = sum(pc[c]["covered"] for c in CLASSES)
    if not total:
        return float("nan")
    wf1 = 0.0
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[a][c] for a in CLASSES if a != c)
        fn = sum(conf[c][b] for b in CLASSES if b != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        wf1 += f1 * pc[c]["covered"] / total
    return wf1


def class_recall(agg, c):
    return agg["per_class"][c]["recall"]


def kappa_vs_orthography(scores):
    """Cohen kappa of instrument-pred vs orthographic-target over all covered TBUs in the population."""
    pairs = [(p, t) for s in scores for p, t in zip(s.get("pred") or [], s.get("target") or [])
             if p is not None and t in CLASSES]
    return cohen_kappa(pairs), len(pairs)


# ============================== bootstrap (resample CLIPS) ==============================
def bootstrap_lcb(scores, stat_fn, n_boot=1000, q=0.10, seed=0):
    """Lower confidence bound of stat_fn(aggregate(scores)) by clip-level bootstrap.

    Resamples whole CLIPS with replacement (not TBUs) so within-clip correlation is respected, recomputes
    the statistic, and returns (point_estimate, lower_bound) where lower_bound is the q-quantile of the
    bootstrap distribution (q=0.10 => one-sided 90% lower bound). NaN draws are dropped. Pure python; a
    1000x bootstrap over ~240 clips is sub-second."""
    point = stat_fn(aggregate(scores))
    N = len(scores)
    if N == 0:
        return point, float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(N)] for _ in range(N)]
        v = stat_fn(aggregate(sample))
        if v == v:
            vals.append(v)
    if not vals:
        return point, float("nan")
    vals.sort()
    lo = vals[min(len(vals) - 1, int(q * len(vals)))]
    return point, lo


def bootstrap_kappa_lcb(pairs, n_boot=1000, q=0.10, seed=0):
    """Lower confidence bound of cohen_kappa over a list of (a,b) pairs, by pair-level bootstrap.
    (Pair-level — not clip-level — because the kappa caller already flattened to pairs; acceptable for a
    corroborative statistic.)"""
    point = cohen_kappa(pairs)
    n = len(pairs)
    if n == 0:
        return point, float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        k = cohen_kappa(sample)
        if k == k:
            vals.append(k)
    if not vals:
        return point, float("nan")
    vals.sort()
    return point, vals[min(len(vals) - 1, int(q * len(vals)))]


# ============================== monotonicity trend test ==============================
def _rankdata(xs):
    """Average ranks (1-based), ties shared. Pure python."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                            # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs, ys):
    rx, ry = _rankdata(xs), _rankdata(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def trend_increasing(xs, ys, n_perm=2000, seed=0):
    """One-sided permutation Spearman test that ys INCREASES with xs.
    Returns (rho, p). p = P(rho_perm >= rho_obs) under random relabeling of ys. Use for PSOLA detection
    rate vs flip magnitude k: a flat or decreasing instrument gets a large p and FAILS the gate."""
    rho = spearman_rho(xs, ys)
    if rho != rho:
        return rho, 1.0
    rng = random.Random(seed)
    ys2 = list(ys)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(ys2)
        r = spearman_rho(xs, ys2)
        if r == r and r >= rho - 1e-12:
            ge += 1
    return rho, (ge + 1) / (n_perm + 1)


# ============================== control helpers ==============================
def permute_targets(scores, seed=0):
    """A label-PERMUTATION control: each clip keeps its pred[] but its target[] is shuffled. Scored in
    balanced-acc this lands at ~1/3 (prior-free chance) for ANY meter, so a real meter must beat it.
    (This is the cheap, audio-free shuffled-target control of L0-4.)"""
    rng = random.Random(seed)
    out = []
    for s in scores:
        t = list(s.get("target") or [])
        rng.shuffle(t)
        out.append(dict(s, target=t))
    return out


def force_all_mid(scores):
    """Degenerate 'always predict Mid' baseline (pred forced to 'M' wherever it was covered). balanced-acc
    must land at ~1/3 — a sanity check that the gate's headline statistic actually penalizes Mid-only."""
    out = []
    for s in scores:
        pred = ["M" if p is not None else None for p in (s.get("pred") or [])]
        out.append(dict(s, pred=pred))
    return out


# ============================== role-aware Layer-0 gate ==============================
# nb27's first run PROVED the two instruments measure DIFFERENT things, so they are certified for DIFFERENT
# ROLES (not held to one tone bar that only I1 could fake and only a precise meter could pass):
#
#   role="quality" — the AfriHuBERT probe (I1). It agrees with orthography on clear speech (bal_acc 0.78)
#       but is largely PITCH-BLIND: flat-F0 audio barely hurts it (monotone 0.78->0.54) and scrambling
#       which-tone-where doesn't move it (margin 0.07). So it CANNOT certify tone CORRECTNESS — it is a
#       "clear Yoruba speech / anti-collapse" meter. GATE: balanced accuracy + coverage + shuffled-target
#       control. wF1 / kappa / oracle / margin are computed and REPORTED, not gated.
#
#   role="tone" — the F0-residual meter (I2). It genuinely READS PITCH — proven because flattening the
#       pitch collapses it to chance (monotone 0.33) — but coarsely (~0.58, near the F0-rule ceiling).
#       GATE on the evidence that it reads pitch AND the pitch is right: a realistic balanced-accuracy bar,
#       per-class H/L recall, the MONOTONE-COLLAPSE control (flat F0 -> chance), and the tone-scramble
#       MARGIN (real - psola_shuffle). The PSOLA single-flip oracle is REPORTED only: its binary test
#       needs near-perfect per-class recall no F0-rule meter has, and its flip magnitude had a sizing bug.
#
# This is honest SCOPING, not bar-lowering. A flat-tone checkpoint still FAILS (I2 collapses to chance); a
# mumbling checkpoint still FAILS (I1 bal_acc drops). What changes is that each instrument is held to what
# it VALIDLY measures, instead of to a precision-tone bar neither could meet.

SHARED_THRESHOLDS = {
    "min_support": 150,                  # covered H and L TBUs each; else INDETERMINATE
    "control_bal_acc_max": 0.40,         # shuffled-target must sit at prior-free chance (~1/3)
    "bootstrap_n": 1000, "bootstrap_q": 0.10,
}
ROLE_THRESHOLDS = {
    "quality": {"bal_acc_min": 0.70, "cov_min": 0.70},
    "tone":    {"bal_acc_min": 0.52, "cov_min": 0.60, "hl_recall_min": 0.40,
                "monotone_max": 0.40, "margin_min": 0.08},
}


def instrument_layer0(scores, inst, controls, oracle, role="tone", thresholds=None, seed=0):
    """Role-aware Layer-0 checks for ONE instrument (pure; applies thresholds to pre-computed bundles).

    role="quality": gate balanced-accuracy + coverage + shuffled-target control ONLY (it is a speech-
        quality / anti-collapse meter; everything else is REPORTED).
    role="tone": additionally gate per-class H/L recall, the monotone-COLLAPSE control (flat F0 -> chance,
        proving it reads pitch), and the tone-scramble MARGIN. The PSOLA oracle is REPORTED only.

    controls = {"shuffled_target","monotone","all_mid","psola_shuffle","codec": [score dicts]}  (missing
        controls are reported NaN; a control REQUIRED for the role's evidence precondition, if absent,
        forces INDETERMINATE rather than a silent pass).
    oracle = {"detect_by_k","strong_detect","false_flip_rate","trend_p","n_flips"}  (reported only).
    """
    if role not in ROLE_THRESHOLDS:
        raise ValueError(f"unknown role {role!r}; expected one of {list(ROLE_THRESHOLDS)}")
    th = {**SHARED_THRESHOLDS, **ROLE_THRESHOLDS[role]}
    if thresholds:
        th.update(thresholds)
    controls = controls or {}
    bn, bq = th["bootstrap_n"], th["bootstrap_q"]
    agg = aggregate(scores)

    # support precondition (covered H and L)
    cov_H = agg["per_class"]["H"]["covered"]
    cov_L = agg["per_class"]["L"]["covered"]
    support_ok = (cov_H >= th["min_support"]) and (cov_L >= th["min_support"])

    # headline statistics (all computed; some gated, some reported, per role)
    bal = balanced_accuracy(agg)
    bal_ok = _ge(bal, th["bal_acc_min"])
    cov_ok = _ge(agg["coverage"], th["cov_min"])
    Hrec, Hrec_lcb = bootstrap_lcb(scores, lambda a: class_recall(a, "H"), bn, bq, seed)
    Lrec, Lrec_lcb = bootstrap_lcb(scores, lambda a: class_recall(a, "L"), bn, bq, seed + 1)
    wf1 = weighted_f1(agg)
    kpairs = [(p, t) for s in scores for p, t in zip(s.get("pred") or [], s.get("target") or [])
              if p is not None and t in CLASSES]
    kappa, kappa_lcb = bootstrap_kappa_lcb(kpairs, bn, bq, seed + 2)

    # controls -> balanced accuracy (NaN when a control is absent)
    def _ctrl_bal(name):
        cs = controls.get(name)
        return balanced_accuracy(aggregate(cs)) if cs else float("nan")
    shuf_bal = _ctrl_bal("shuffled_target")
    mono_bal = _ctrl_bal("monotone")
    allmid_bal = _ctrl_bal("all_mid")
    scram_bal = _ctrl_bal("psola_shuffle")
    codec_bal = _ctrl_bal("codec")
    shuf_ctrl_ok = (shuf_bal != shuf_bal) or (shuf_bal <= th["control_bal_acc_max"])
    margin = (bal - scram_bal) if (bal == bal and scram_bal == scram_bal) else float("nan")
    codec_drop = (bal - codec_bal) if (bal == bal and codec_bal == codec_bal) else float("nan")

    # oracle: REPORTED diagnostic only (demoted — too strict for a coarse meter + a fixed sizing caveat)
    oracle_view = None
    if oracle is not None:
        det = oracle.get("detect_by_k") or {}
        strong = oracle.get("strong_detect")
        if strong is None:
            strong = max(det.items(), key=lambda kv: kv[0])[1] if det else float("nan")
        oracle_view = {"strong_detect": strong, "false_flip_rate": oracle.get("false_flip_rate"),
                       "trend_p": oracle.get("trend_p"), "n_flips": oracle.get("n_flips"),
                       "detect_by_k": det}

    # role-specific gating + evidence precondition
    checks = {"balanced_acc": bal_ok, "shuffle_control": shuf_ctrl_ok}
    if role == "quality":
        checks["coverage"] = cov_ok                      # a quality meter must label most speech
        evidence_ok = bool(controls.get("shuffled_target"))
    else:  # tone — coverage is REPORTED; the >=150 covered-H/L support precondition guards abstention
        checks["H_recall"] = _ge(Hrec, th["hl_recall_min"])
        checks["L_recall"] = _ge(Lrec, th["hl_recall_min"])
        checks["monotone_collapse"] = (mono_bal != mono_bal) or (mono_bal <= th["monotone_max"])
        checks["margin"] = _ge(margin, th["margin_min"])
        evidence_ok = bool(controls.get("shuffled_target") and controls.get("monotone")
                           and controls.get("psola_shuffle"))

    inst_pass = support_ok and evidence_ok and all(checks.values())

    return {
        "role": role, "pass": bool(inst_pass),
        "support_ok": support_ok, "evidence_ok": evidence_ok, "n_covered": {"H": cov_H, "L": cov_L},
        "balanced_acc": bal, "coverage": agg["coverage"],
        "H_recall": Hrec, "H_recall_lcb": Hrec_lcb, "L_recall": Lrec, "L_recall_lcb": Lrec_lcb,
        "weighted_f1": wf1, "kappa": kappa, "kappa_lcb": kappa_lcb,
        "controls": {"shuffled_target": shuf_bal, "monotone": mono_bal, "all_mid": allmid_bal,
                     "psola_shuffle": scram_bal, "codec": codec_bal},
        "margin": margin, "codec_drop": codec_drop, "oracle": oracle_view,
        "checks": checks, "thresholds": th, "aggregate": agg,
    }


def layer0_gate(real, controls, oracle, roles, marginals=None, kappa_pairs=None,
                thresholds=None, seed=0):
    """Assemble the role-aware Layer-0 verdict.

    real     : {inst: [score dicts]} on the REAL TEST split (orthography target).
    controls : {inst: {control_name: [score dicts]}}.   oracle: {inst: oracle bundle} (reported only).
    roles    : {inst: "quality"|"tone"} — what each instrument is certified AS. The binding verdict gates
               BOTH (the quality meter and the tone meter must each pass their role checks).
    kappa_pairs: co-covered (I1,I2) pairs — REPORTED (expected LOW: the two roles measure different things).

    layer0_verdict: "INDETERMINATE" if any instrument has thin support or missing role-evidence; else
    "PASS" iff every instrument passes its role checks; else "FAIL".
    """
    per_instrument = {}
    any_indeterminate = False
    all_pass = True
    for inst, role in roles.items():
        if inst not in real:
            per_instrument[inst] = {"error": "missing real scores", "pass": False,
                                    "support_ok": False, "evidence_ok": False, "role": role}
            any_indeterminate = True
            continue
        r = instrument_layer0(real[inst], inst, (controls or {}).get(inst, {}),
                              (oracle or {}).get(inst), role=role, thresholds=thresholds, seed=seed)
        per_instrument[inst] = r
        if not r["support_ok"] or not r["evidence_ok"]:
            any_indeterminate = True       # thin support OR missing role-evidence -> not trustworthy
        all_pass = all_pass and r["pass"]

    cross = None
    if kappa_pairs is not None:
        cross = {"kappa_I1_I2": cohen_kappa(kappa_pairs), "n_pairs": len(kappa_pairs), "gating": False,
                 "note": "expected LOW — I1=quality and I2=tone measure different things"}

    verdict = "INDETERMINATE" if any_indeterminate else ("PASS" if all_pass else "FAIL")
    return {
        "layer0_verdict": verdict,
        "roles": dict(roles),
        "per_instrument": per_instrument,
        "cross_instrument": cross,
        "marginals": marginals,
    }


def explain(verdict):
    """Human-readable summary of a layer0_gate() verdict for notebook logs."""
    def f(x):
        return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) and x == x else str(x))
    lines = [f"LAYER-0 GATE: {verdict['layer0_verdict']}  (roles={verdict['roles']})"]
    for inst, r in verdict["per_instrument"].items():
        if "error" in r:
            lines.append(f"  [{inst}] ERROR: {r['error']}")
            continue
        if not r["support_ok"]:
            lines.append(f"  [{inst}] INDETERMINATE — thin support (covered H={r['n_covered']['H']} "
                         f"L={r['n_covered']['L']}; need >= {r['thresholds']['min_support']})")
        if not r["evidence_ok"]:
            lines.append(f"  [{inst}] INDETERMINATE — missing role-evidence control(s) "
                         f"(need shuffled-target{'/monotone/psola_shuffle' if r['role']=='tone' else ''})")
        c = r["controls"]
        lines.append(f"  [{inst}/{r['role']}] {'pass' if r['pass'] else 'FAIL'}  "
                     f"bal_acc={f(r['balanced_acc'])} cov={f(r['coverage'])} "
                     f"wF1={f(r['weighted_f1'])} kappa={f(r['kappa'])}")
        lines.append(f"        H rec={f(r['H_recall'])} L rec={f(r['L_recall'])} | "
                     f"shuf={f(c['shuffled_target'])} mono={f(c['monotone'])} "
                     f"scramble={f(c['psola_shuffle'])} margin={f(r['margin'])} codecΔ={f(r['codec_drop'])}")
        if r.get("oracle"):
            o = r["oracle"]
            lines.append(f"        oracle (reported): detect={f(o['strong_detect'])} "
                         f"ff={f(o['false_flip_rate'])} trend_p={f(o['trend_p'])}")
        failed = [k for k, v in r["checks"].items() if not v]
        if failed:
            lines.append(f"        failed: {', '.join(failed)}")
    if verdict.get("cross_instrument"):
        cx = verdict["cross_instrument"]
        lines.append(f"  kappa(I1,I2)={f(cx['kappa_I1_I2'])} (reported, n={cx['n_pairs']}; "
                     f"low is EXPECTED — different roles)")
    return "\n".join(lines)
