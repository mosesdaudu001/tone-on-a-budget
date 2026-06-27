# coding=utf-8
# tone_eval_v2.py — downdrift-aware Yoruba tone metric (Phase 0b of YORUBA_WAY_FORWARD.md).
#
# WHY v1 (tone_eval.py) IS RETIRED AS A SCORE (kept only for the floor demo, nb20 §0a):
#   v1 classifies each syllable's median F0 against the UTTERANCE-GLOBAL median (±0.8 st). Yoruba has
#   DOWNDRIFT/downstep — H falls ~5 st across one utterance (Laniran & Clements 2003), corpus downtrend
#   ≈ −12 Hz/s (van Niekerk & Barnard 2012) — so early L/M read as H and late H read as M/L on PERFECT
#   speech (expected score ~0.5-0.6, ≈ chance on long sentences; the all-Mid baseline is 0.373 and the
#   base model scored 0.388 = the metric floor, not the model). v1 also silently predicts "M" for
#   unvoiced syllables (free credit on creaky final-L) and its alignment-window builder drops toned
#   syllabic nasals (ń/ǹ) — len mismatch → whole-utterance failure (the GRPO ~30% zero-gradient steps).
#
# WHAT v2 DOES INSTEAD (red-team spec 2026-06-10):
#   - Scores TONE TRANSITIONS between adjacent tone-bearing units (TBUs): rise / level / fall — the cue
#     native perception actually uses (Carter-Enyi 2016: ~+2/−3 st category thresholds, asymmetric).
#     Direction is downdrift-robust by construction; we additionally DETREND (OLS slope of TBU semitones
#     vs time, clamped to the physical declination range) so long-gap transitions aren't biased "fall".
#   - Per-TBU F0 = median over the LATE half of the TBU window (tone targets are realized late in the
#     vowel) of VOICED frames only.
#   - UNVOICED/UNALIGNED TBU = ABSTENTION, never "M": transitions touching it are not scored; we report
#     (accuracy-on-covered, coverage) as a PAIR. Mumbled audio ⇒ low coverage, not free credit.
#     Treat coverage < ~0.7 as a failure signal regardless of accuracy.
#   - F0 backend: SwiftF0 (pip swift-f0, MIT; 90.4% PTDB, ~90x faster than CREPE) preferred;
#     pyworld.harvest fallback. Backend is injectable for tests.
#   - Alignment: forced_tbu_windows() fixes v1's two alignment bugs — it strips ONLY tone marks
#     (keeps dot-below: ọ/ẹ/ṣ are real MMS-yor vocab entries; v1's strip of ALL combining marks turned
#     ọ→o and degraded alignment) and maps TBUs from the ORIGINAL text units (toned nasals included)
#     to flat-char positions, returning None per-TBU instead of failing the whole utterance.
#
# CALIBRATION: rise_st/fall_st defaults (+0.6/−0.9) are uncalibrated placeholders. nb20 §10 grid-searches
# them so REAL Yoruba speech scores ≥ ~0.85 while shuffled-target and monotone controls stay at chance,
# then persists them (S3 tone_v2_calibration.json). Do not trust cross-run comparisons before that.
#
# PUBLIC:
#   tbu_seq(text)                          -> ["H","M","L",...]   (delegates to tone_eval)
#   precompute(wav, sr, text, asr=..., proc=..., device=..., emissions=None, n16=None)
#       -> dict(f0, times, wins, tones, method, n16)   # heavy work, cache for grid search
#   score_from_precomputed(pre, rise_st=.6, fall_st=.9, detrend=True, late_frac=0.5) -> dict
#   tone_transition_score(wav, sr, text, ...) -> dict  # precompute + score in one call
#   forced_tbu_windows(wav, sr, text, asr, proc, device, emissions=None, n16=None)
#       -> list[(t0,t1) | None] aligned with tbu_seq(text)
import math
import unicodedata as ud

try:
    from . import tone_eval
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_eval

_TONE_LEVEL = {"H": 2, "M": 1, "L": 0}
_TONE_COMBINING = {"̀", "́", "̄"}   # grave, acute, macron — the ONLY marks we strip
# physical declination range, st/s: downdrift is a DECLINE (≤0); −12 Hz/s ≈ −1..−1.7 st/s in speech range
_SLOPE_CLAMP = (-2.0, 0.0)


def tbu_seq(text):
    return tone_eval.target_tone_seq(text)


# ----------------------------- F0 backends -----------------------------
_SWIFT = None


def extract_f0_v2(wav, sr, fmin=65.0, fmax=400.0, confidence=0.9):
    """(f0_hz with NaN where unvoiced, frame_times_s). SwiftF0 preferred, pyworld fallback.
    NOTE: SwiftF0 voicing on SYNTHETIC/degraded audio can be conservative — nb20 calibrates the
    confidence threshold alongside rise/fall; coverage reporting absorbs the rest."""
    import numpy as np
    w = np.asarray(wav, dtype="float32")
    if w.ndim > 1:
        w = w.mean(-1)
    global _SWIFT
    try:
        from swift_f0 import SwiftF0
        if _SWIFT is None or _SWIFT[1] != (fmin, fmax, confidence):
            _SWIFT = (SwiftF0(fmin=fmin, fmax=fmax, confidence_threshold=confidence), (fmin, fmax, confidence))
        r = _SWIFT[0].detect_from_array(w, int(sr))
        f0 = np.where(r.voicing, r.pitch_hz, np.nan)
        return f0, np.asarray(r.timestamps, dtype="float64"), "swift-f0"
    except Exception:
        f0, t = tone_eval.extract_f0(w, sr, fmin=fmin, fmax=fmax)   # pyworld/pyin path
        return f0, t, "pyworld/pyin"


# ----------------------------- alignment windows -----------------------------
def _flat_for_align(text):
    """Tone-mark-only flattening for CTC alignment against MMS-yor.
    Returns (flat_lower:str, tbu_pos:list[int]) where tbu_pos[i] = index in flat of the base char of
    the i-th TBU of tbu_seq(text) — INCLUDING toned syllabic nasals (v1 dropped them: bug).
    Keeps dot-below (ọ/ẹ/ṣ are in the MMS-yor vocab); strips ONLY grave/acute/macron; NFC-recomposes."""
    flat_parts, tbu_pos, off = [], [], 0
    for unit in tone_eval._units(text):
        d = ud.normalize("NFD", unit)
        kept = "".join(c for c in d if c not in _TONE_COMBINING)
        s = ud.normalize("NFC", kept).lower()
        if tone_eval._tbu_tone(unit) is not None:
            tbu_pos.append(off if s else None)   # standalone-mark units never occur (merged by _units)
        flat_parts.append(s)
        off += len(s)
    return "".join(flat_parts), tbu_pos


def forced_tbu_windows(wav, sr, text, asr, proc, device, emissions=None, n16=None):
    """Per-TBU (t0,t1) windows from MMS-yor CTC forced alignment. SAME length/order as tbu_seq(text);
    entries are None where that TBU couldn't be aligned (abstention downstream) — NEVER a whole-list
    failure for one bad char (v1 bug). Returns None only if alignment itself is impossible.
    Pass `emissions` (raw MMS logits [T,V] on-device) + `n16` to reuse a shared ASR forward."""
    try:
        import numpy as np, torch, librosa
        import torchaudio.functional as AF
        flat, tbu_pos = _flat_for_align(text)
        if len(flat) < 2 or not tbu_pos:
            return None
        vocab = proc.tokenizer.get_vocab()
        blank = proc.tokenizer.pad_token_id if proc.tokenizer.pad_token_id is not None else 0
        if emissions is not None and n16 is not None:
            emis = torch.log_softmax(emissions.float(), dim=-1).cpu()
            n_samp16 = int(n16)
        else:
            w = np.asarray(wav, dtype="float32"); w = w.mean(-1) if w.ndim > 1 else w
            w16 = librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w
            iv = proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                emis = torch.log_softmax(asr(iv).logits, dim=-1)[0].cpu()
            n_samp16 = int(w16.shape[0])
        ids, char_idx = [], []
        for j, ch in enumerate(flat):
            tid = vocab.get("|" if ch == " " else ch)
            if tid is None or tid == blank:
                continue
            ids.append(tid); char_idx.append(j)
        if len(ids) < 2:
            return None
        targets = torch.tensor([ids], dtype=torch.int32)
        aligned, scores = AF.forced_align(emis.unsqueeze(0), targets, blank=blank)
        spans = AF.merge_tokens(aligned[0], scores[0], blank=blank)
        sec_per_frame = n_samp16 / 16000.0 / emis.shape[0]
        char_time = {cj: (sp.start * sec_per_frame, sp.end * sec_per_frame)
                     for sp, cj in zip(spans, char_idx)}
        return [char_time.get(p) if p is not None else None for p in tbu_pos]
    except Exception:
        return None


# ----------------------------- precompute + score -----------------------------
def precompute(wav, sr, text, asr=None, proc=None, device="cpu", emissions=None, n16=None,
               fmin=65.0, fmax=400.0, confidence=0.9):
    """All heavy per-clip work (F0 + alignment), cacheable for threshold grid-search.
    Returns dict(f0, times, wins, tones, method, backend). wins entries may be None (abstain)."""
    import numpy as np
    tones = tbu_seq(text)
    f0, times, backend = extract_f0_v2(wav, sr, fmin=fmin, fmax=fmax, confidence=confidence)
    wins, method = None, "proportional"
    if asr is not None and proc is not None:
        wins = forced_tbu_windows(wav, sr, text, asr, proc, device, emissions=emissions, n16=n16)
        if wins is not None and len(wins) == len(tones):
            method = "forced-align"
        else:
            wins = None
    if wins is None:
        span = tone_eval._voiced_span(f0, times)
        wins = (tone_eval._proportional_windows(span, len(tones)) if span is not None
                else [None] * len(tones))
    return dict(f0=np.asarray(f0, dtype="float64"), times=np.asarray(times, dtype="float64"),
                wins=wins, tones=tones, method=method, backend=backend)


def _tbu_semitones(pre, late_frac=0.5):
    """Per-TBU (semitone, mid_time) from the LATE part of each window; None where unvoiced/unaligned."""
    import numpy as np
    f0, times = pre["f0"], pre["times"]
    out = []
    for win in pre["wins"]:
        if win is None:
            out.append(None); continue
        t0, t1 = win
        lt0 = t0 + late_frac * (t1 - t0)
        m = (times >= lt0) & (times < t1) & (~np.isnan(f0))   # half-open: abutting windows don't share frames
        if not m.any():                                   # fall back to the whole window before abstaining
            m = (times >= t0) & (times < t1) & (~np.isnan(f0))
        if not m.any():
            out.append(None); continue
        st = 12.0 * math.log2(max(float(np.nanmedian(f0[m])), 1e-6) / 100.0)
        out.append((st, (t0 + t1) / 2.0))
    return out


def _detrend(sts, tones, clamp=_SLOPE_CLAMP):
    """Remove a clamped global declination slope (st/s) from TBU semitones. Returns (residuals, slope).
    The slope is fit on TONE-CENTERED data (Frisch–Waugh: subtract each target-tone group's mean time and
    mean semitone before pooling the OLS) — a raw fit would absorb any tone pattern that trends with time
    (HHH...MMM...LLL reads as a steep "declination" and its level transitions misclassify; adversarial
    review 2026-06-10). Centering on the KNOWN diacritic tones removes the tone-time correlation exactly;
    flat/wrong audio still gets b≈0, so this is not gameable. Clamp = physical downdrift range."""
    pts = [(mt, st, tn) for (p, tn) in zip(sts, tones) if p is not None for (st, mt) in [p]]
    if len(pts) < 4:
        return [None if p is None else p[0] for p in sts], 0.0
    groups = {}
    for t, s, tn in pts:
        groups.setdefault(tn, []).append((t, s))
    cx, cy = [], []
    for lst in groups.values():
        mx = sum(t for t, _ in lst) / len(lst)
        my = sum(s for _, s in lst) / len(lst)
        cx.extend(t - mx for t, _ in lst)
        cy.extend(s - my for _, s in lst)
    den = sum(x * x for x in cx)
    b = 0.0 if den < 1e-9 else sum(x * y for x, y in zip(cx, cy)) / den
    b = min(max(b, clamp[0]), clamp[1])
    return [None if p is None else p[0] - b * p[1] for p in sts], b


def score_from_precomputed(pre, rise_st=0.6, fall_st=0.9, detrend=True, late_frac=0.5):
    """Transition score from a precompute() dict. Returns:
    dict(accuracy, coverage, n_scored, n_trans, n_tbu, pred, target, per_class, slope, method, backend)
    accuracy is NaN when nothing is scoreable; ALWAYS read it together with coverage."""
    import numpy as np
    tones = pre["tones"]
    n_tbu = len(tones)
    out = dict(accuracy=float("nan"), coverage=0.0, n_scored=0, n_trans=max(0, n_tbu - 1), n_tbu=n_tbu,
               pred=[], target=[], per_class={}, slope=0.0, method=pre["method"], backend=pre["backend"])
    if n_tbu < 2:
        return out
    sts = _tbu_semitones(pre, late_frac=late_frac)
    res, slope = _detrend(sts, tones) if detrend else ([None if p is None else p[0] for p in sts], 0.0)
    out["slope"] = float(slope)
    pred, target, hits = [], [], {"R": [0, 0], "level": [0, 0], "F": [0, 0]}
    for i in range(1, n_tbu):
        td = _TONE_LEVEL[tones[i]] - _TONE_LEVEL[tones[i - 1]]
        tcls = "R" if td > 0 else ("F" if td < 0 else "level")
        if res[i] is None or res[i - 1] is None:
            pred.append(None); target.append(tcls)
            continue
        d = res[i] - res[i - 1]
        pcls = "R" if d >= rise_st else ("F" if d <= -fall_st else "level")
        pred.append(pcls); target.append(tcls)
        hits[tcls][1] += 1
        hits[tcls][0] += int(pcls == tcls)
    scored = [(p, t) for p, t in zip(pred, target) if p is not None]
    out.update(pred=pred, target=target, n_scored=len(scored),
               coverage=len(scored) / out["n_trans"] if out["n_trans"] else 0.0,
               per_class={k: (h / n if n else float("nan")) for k, (h, n) in hits.items()})
    if scored:
        out["accuracy"] = float(np.mean([p == t for p, t in scored]))
    return out


def tone_transition_score(wav, sr, text, asr=None, proc=None, device="cpu", emissions=None, n16=None,
                          rise_st=0.6, fall_st=0.9, detrend=True, late_frac=0.5, confidence=0.9):
    """One-call convenience: precompute + score. For grid search, call precompute once per clip and
    score_from_precomputed per threshold setting instead."""
    pre = precompute(wav, sr, text, asr=asr, proc=proc, device=device,
                     emissions=emissions, n16=n16, confidence=confidence)
    return score_from_precomputed(pre, rise_st=rise_st, fall_st=fall_st,
                                  detrend=detrend, late_frac=late_frac)
