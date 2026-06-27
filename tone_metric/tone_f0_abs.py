# coding=utf-8
# tone_f0_abs.py — Instrument I2: ABSOLUTE per-TBU H/M/L tone from F0, downdrift-removed.
#
# WHY THIS EXISTS (the second, method-independent tone meter)
#   I1 = the AfriHuBERT SSL probe (tone_probe.probe_score) is validated (0.867 vs orthography on REAL
#   held-out speech, 0.40 on bad gen). But a single instrument scored against its own training target
#   shares a blind spot with no cross-check: a learned spurious cue (e.g. timbre<->tone correlation) or a
#   transcription-convention bias would pass undetected. I2 measures tone from a COMPLETELY DIFFERENT
#   signal — the raw F0 (pitch) contour — so when I1 and I2 BOTH agree with the written tone on real
#   speech and BOTH drop on tone-shuffled audio, that agreement is real convergent validity, not an echo.
#
#   tone_eval_v2.py already scores tone TRANSITIONS (rise/level/fall). I2 instead emits ABSOLUTE per-TBU
#   classes H/M/L in the SAME label space as probe_score, so tone_gate.py can aggregate and cross-check
#   the two instruments TBU-for-TBU. It reuses v2.precompute()/_tbu_semitones() verbatim (F0 + windows +
#   per-TBU semitone targets) and only changes the final step: declination removal + register anchoring +
#   threshold classification.
#
# THE TONAL PHONETICS (why absolute F0 needs care)
#   Yoruba H/M/L are F0 targets, but two effects corrupt naive "high pitch = High tone":
#     (1) DOWNDRIFT/declination: F0 falls ~5 st across an utterance, so a late H sits below an early M.
#         -> we DETREND: fit a declination slope (st/s), clamp to the physical range, subtract it.
#     (2) SPEAKER REGISTER: "high" is relative to THIS speaker's range. -> we anchor a Mid reference and
#         classify each TBU's detrended residual RELATIVE to it: residual >= +theta_h -> H,
#         residual <= -theta_l -> L, else M.
#
#   DETREND HAS TWO MODES (the label-leak guard):
#     - "blind"  (DEFAULT, the one that counts): Theil-Sen robust slope of (semitone vs time) over ALL
#        TBUs, using NO knowledge of the target tones. Mid reference = median of residuals (or a
#        speaker-global `mid_ref` from calibration). NOTHING here reads the answer key.
#     - "fw"     (diagnostic only): reuses v2._detrend, which Frisch-Waugh-centers the slope fit on the
#        KNOWN target tones. More accurate but ANSWER-KEY-DEPENDENT. If a checkpoint passes in "fw" but
#        collapses in "blind", the answer-key centering was doing the work, not the acoustics -> not a
#        valid pass. nb27 runs both; the gate trusts "blind".
#
# CALIBRATION (uncalibrated defaults — DO NOT trust cross-run numbers before nb27 freezes them)
#   theta_h / theta_l (semitone half-widths of the Mid band) and the SwiftF0 confidence are PLACEHOLDERS.
#   nb27 grid-searches them so REAL Yoruba scores high while shuffled-target and monotone-resynth controls
#   stay at chance, then persists S3 tts_data/yoruba/tone_v2/f0_abs_calibration.v1.json. A speaker-global
#   Mid anchor (from gold) can be passed as `mid_ref` to remove the per-utterance tone-imbalance bias.
#
# ALIGNMENT INDEPENDENCE — HONEST CAVEAT (v1 limitation, tracked)
#   This v1 reuses v2.forced_tbu_windows (MMS-yor CTC) for TBU time windows — the SAME aligner the probe
#   uses. So I2 is MEASUREMENT-independent (raw F0 vs SSL features) but NOT yet ALIGNMENT-independent: if
#   the aligner places a window on the wrong syllable, both meters score the wrong syllable and can agree
#   on garbage. Consequence: per-instrument accuracy-vs-orthography is sound (alignment is reliable on
#   real read speech), but kappa(I1,I2) is only CORROBORATION, never the primary gate (tone_gate.py
#   already demotes it for exactly this reason). v2 upgrade = an independent aligner (MFA-Yoruba if a
#   pretrained model exists, else acoustic vowel-nucleus landmarks); pass `wins=` to override.
#
# PUBLIC:
#   score_abs_from_precomputed(pre, theta_h=.., theta_l=.., mode="blind", mid_ref=None, late_frac=.5)
#       -> dict(accuracy, coverage, n_scored, n_tbu, n_trans, pred[H/M/L|None], target, per_class,
#               slope, mid_ref, mode, method, backend)   # SAME key set as tone_probe.probe_score
#   f0_abs_score(wav, sr, text, asr=.., proc=.., device=.., emissions=.., n16=.., wins=None, **thr)
#       -> same dict  (precompute + score in one call)

import math

try:
    from . import tone_eval_v2 as v2
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_eval_v2 as v2

CLASSES = ("H", "M", "L")
DEFAULT_THETA_H = 1.0   # semitones above the Mid register center to call H (PLACEHOLDER -> nb27 calibrates)
DEFAULT_THETA_L = 1.0   # semitones below the Mid register center to call L (PLACEHOLDER -> nb27 calibrates)


# ----------------------------- blind declination (no answer key) -----------------------------
def _theil_sen_slope(xs, ys, clamp=v2._SLOPE_CLAMP):
    """Median of pairwise slopes (Theil-Sen) of ys vs xs, clamped to the physical declination range.
    Robust to a few mis-pitched TBUs and — unlike v2._detrend — uses NO tone labels. Pure python."""
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if abs(dx) > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0
    slopes.sort()
    k = len(slopes)
    med = slopes[k // 2] if k % 2 else 0.5 * (slopes[k // 2 - 1] + slopes[k // 2])
    return min(max(med, clamp[0]), clamp[1])


def _median(vals):
    s = sorted(vals)
    k = len(s)
    if k == 0:
        return float("nan")
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


def _blind_residuals(sts):
    """(residuals, slope) from per-TBU (semitone, time) points using a Theil-Sen declination fit with NO
    target-tone knowledge. residuals[i] is None where sts[i] is None (unvoiced/unaligned TBU)."""
    pts = [(mt, st) for p in sts if p is not None for (st, mt) in [p]]
    if len(pts) < 3:
        # too few points to estimate a slope; fall back to zero-slope (raw semitones)
        return [None if p is None else p[0] for p in sts], 0.0
    xs = [t for t, _ in pts]
    ys = [s for _, s in pts]
    b = _theil_sen_slope(xs, ys)
    return [None if p is None else (p[0] - b * p[1]) for p in sts], b


# ----------------------------- scoring core (model-free; unit-testable) -----------------------------
def score_abs_from_precomputed(pre, theta_h=DEFAULT_THETA_H, theta_l=DEFAULT_THETA_L,
                               mode="blind", mid_ref=None, late_frac=0.5):
    """Absolute per-TBU H/M/L from a v2.precompute() dict. Returns the SAME key set as
    tone_probe.probe_score (pred/target are H/M/L; coverage is per-TBU). accuracy is NaN when nothing is
    scoreable — ALWAYS read it with coverage."""
    tones = pre["tones"]
    n_tbu = len(tones)
    out = dict(accuracy=float("nan"), coverage=0.0, n_scored=0, n_tbu=n_tbu,
               n_trans=max(0, n_tbu - 1), pred=[None] * n_tbu, target=list(tones), per_class={},
               slope=0.0, mid_ref=None, mode=mode, method=f"f0-abs-{mode}", backend=pre.get("backend"))
    if n_tbu == 0:
        out["pred"] = []
        return out

    sts = v2._tbu_semitones(pre, late_frac=late_frac)   # per-TBU (semitone, mid_time) or None

    if mode == "fw":
        res, slope = v2._detrend(sts, tones)            # answer-key-centered (diagnostic only)
    else:
        res, slope = _blind_residuals(sts)              # blind Theil-Sen (the mode that counts)
    out["slope"] = float(slope)

    live = [r for r in res if r is not None]
    if not live:
        return out

    # Register anchor: a speaker-global Mid from calibration if given, else the blind per-utterance
    # median of residuals (no answer-key use).
    center = float(mid_ref) if mid_ref is not None else _median(live)
    out["mid_ref"] = center

    pred = [None] * n_tbu
    for i, r in enumerate(res):
        if r is None:
            continue
        d = r - center
        pred[i] = "H" if d >= theta_h else ("L" if d <= -theta_l else "M")

    scored = [(p, t) for p, t in zip(pred, tones) if p is not None]
    # per-class recall on covered TBUs (standalone convenience; tone_gate re-aggregates from pred/target)
    per_class = {}
    for c in CLASSES:
        cov = sum(1 for p, t in zip(pred, tones) if t == c and p is not None)
        cor = sum(1 for p, t in zip(pred, tones) if t == c and p == c)
        per_class[c] = (cor / cov) if cov else float("nan")

    out.update(pred=pred, n_scored=len(scored),
               coverage=len(scored) / n_tbu if n_tbu else 0.0, per_class=per_class)
    if scored:
        out["accuracy"] = sum(p == t for p, t in scored) / len(scored)
    return out


# ----------------------------- one-call convenience -----------------------------
def f0_abs_score(wav, sr, text, asr=None, proc=None, device="cpu", emissions=None, n16=None,
                 wins=None, theta_h=DEFAULT_THETA_H, theta_l=DEFAULT_THETA_L,
                 mode="blind", mid_ref=None, late_frac=0.5, confidence=0.9):
    """precompute (F0 + alignment) + score. Pass `wins` (list aligned with tbu_seq(text)) to inject an
    INDEPENDENT aligner and override the shared MMS-yor windows (the alignment-independence upgrade)."""
    pre = v2.precompute(wav, sr, text, asr=asr, proc=proc, device=device,
                        emissions=emissions, n16=n16, confidence=confidence)
    if wins is not None:
        if len(wins) != len(pre["tones"]):
            raise ValueError(f"wins length {len(wins)} != #TBUs {len(pre['tones'])}")
        pre = dict(pre, wins=list(wins), method="forced-align-independent")
    return score_abs_from_precomputed(pre, theta_h=theta_h, theta_l=theta_l,
                                      mode=mode, mid_ref=mid_ref, late_frac=late_frac)
