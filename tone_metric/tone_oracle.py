# tone_oracle.py — PSOLA tone-flip ORACLE: human-free ground truth for the Layer-0 gate (L0-5).
#
# WHY THIS EXISTS
#   The two tone meters (I1 SSL probe, I2 F0-residual) are validated against ORTHOGRAPHY on real speech.
#   But orthography only tells us the LEXICAL tone — it can't prove a meter RESPONDS CAUSALLY to a pitch
#   change at a specific syllable. The oracle supplies that: take a REAL clip with KNOWN tones, then
#   PSOLA-shift ONE syllable's F0 by a measured number of semitones to FLIP its tone (push an L up toward
#   H, or an H down toward L), leaving every other syllable's pitch untouched. A trustworthy meter must
#   (a) DETECT the flip at that syllable, (b) NOT change its verdict on the untouched syllables
#   (false-flip <= 0.10), and (c) detect MORE often as the flip gets bigger (monotone in k). No human.
#
#   This is meter SENSITIVITY, not aligner correctness: both meters share the MMS-CTC aligner, so if a
#   window sits on the wrong syllable the flip lands on the wrong syllable for both and they "agree" on a
#   mislabeled TBU. Keep the oracle on REAL READ SPEECH (alignment reliable) and treat it as a
#   sensitivity test. The independent-aligner upgrade (tone_f0_abs wins= override) is the orthogonal fix.
#
# PARSELMOUTH RECIPE (verified: exact in-window semitone shift, ~0 out-of-window leakage, length-preserved)
#   Sound -> "To Manipulation" -> "Extract pitch tier" -> "Multiply frequencies", t0, t1, factor
#   (factor = 2**(semitones/12); clamps to [t0,t1)) -> "Replace pitch tier" -> "Get resynthesis
#   (overlap-add)". Run in /home/moses/audio_env (praat-parselmouth 0.4.7). f0min/f0max MUST bracket the
#   shifted pitch (default 65/400 match v2.extract_f0_v2; raise f0max for very high voices).
#
# I2 REGISTER ANCHOR (required): tone_f0_abs classifies relative to the per-utterance median residual.
#   Flipping one L->H raises that median and can nudge a borderline OTHER TBU across a threshold -> a
#   false-flip BY CONSTRUCTION. The notebook must FREEZE I2's mid_ref from the CLEAN clip (read the
#   `mid_ref` field of the clean I2 score) and bind score_I2 to that fixed anchor BEFORE calling the
#   oracle. The oracle itself stays meter-agnostic: it only calls the scorer closures it is given.
#
# PUBLIC
#   psola_shift_window(wav, t0, t1, semitones, sr=24000, f0min=65, f0max=400) -> np.float32 wav
#   psola_roundtrip(wav, sr, f0min, f0max)   -> resynth with NO pitch edit (artifact-only neg control)
#   psola_flatten(wav, sr, f0min, f0max)     -> F0 flattened to its mean (monotone-resynth neg control)
#   measure_delta_HL(pre)                    -> median(H semitones) - median(L semitones), or None
#   run_oracle_clip(wav, sr, text, pre, base_scores, score_fns, ks, target_classes) -> list[row dict]
#   summarize_oracle(rows, instrument, full_k) -> {detect_by_k, false_flip_rate, trend_rho, trend_p,
#                                                  n_flips, control_trip}   # feeds tone_layer0 L0-5

try:
    from . import tone_eval_v2 as v2
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_eval_v2 as v2
try:
    from . import tone_layer0 as L0
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_layer0 as L0

SR = 24000
F0MIN, F0MAX = 65.0, 600.0   # f0max raised 400->600 (A/B rebuild §2/§7): a full opposite-band reset on a high
                             # voice pushes well past 400 Hz; the ceiling MUST bracket it or PSOLA clips the flip.


# ----------------------------- parselmouth primitives (lazy import) -----------------------------
def _snd(wav, sr=SR):
    import numpy as np
    import parselmouth
    w = np.asarray(wav, dtype="float64")
    w = w.mean(-1) if w.ndim > 1 else w
    return parselmouth.Sound(w, sampling_frequency=sr)


def psola_shift_window(wav, t0, t1, semitones, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """Multiply F0 by 2**(semitones/12) for points in [t0,t1) ONLY; PSOLA-resynthesize. Length-preserving.
    Out-of-window pitch is untouched (verified: 0.00 st delta outside the window)."""
    import numpy as np
    from parselmouth.praat import call
    manip = call(_snd(wav, sr), "To Manipulation", 0.01, f0min, f0max)
    ptier = call(manip, "Extract pitch tier")
    call(ptier, "Multiply frequencies", float(t0), float(t1), 2.0 ** (semitones / 12.0))
    call([manip, ptier], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values).reshape(-1).astype("float32")


def psola_shift_windows(wav, shifts, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """Apply MANY per-window semitone shifts in ONE Manipulation pass (no compounding artifacts).
    shifts = [(t0, t1, semitones), ...] over DISJOINT windows (e.g. each TBU). Used to build a tone-
    SCRAMBLED, timbre-matched control: shift each syllable away from its lexical tone, then score vs the
    CORRECT orthography — a meter that reads TONE collapses; one that reads timbre/codec does not."""
    import numpy as np
    from parselmouth.praat import call
    manip = call(_snd(wav, sr), "To Manipulation", 0.01, f0min, f0max)
    ptier = call(manip, "Extract pitch tier")
    for (t0, t1, st) in shifts:
        if st:
            call(ptier, "Multiply frequencies", float(t0), float(t1), 2.0 ** (st / 12.0))
    call([manip, ptier], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values).reshape(-1).astype("float32")


def psola_roundtrip(wav, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """Resynthesize through Manipulation with NO pitch edit: same PSOLA artifact, tone class UNCHANGED.
    The artifact negative control — proves the meters flag a TONE change, not the resynthesis itself."""
    import numpy as np
    from parselmouth.praat import call
    manip = call(_snd(wav, sr), "To Manipulation", 0.01, f0min, f0max)
    out = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values).reshape(-1).astype("float32")


def psola_flatten(wav, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """Flatten the whole pitch contour to its mean (a monotone/robotic clip). The monotone-resynth
    negative control: a tone meter must COLLAPSE here (predict ~all Mid or abstain)."""
    import numpy as np
    from parselmouth.praat import call
    manip = call(_snd(wav, sr), "To Manipulation", 0.01, f0min, f0max)
    ptier = call(manip, "Extract pitch tier")
    n = int(call(ptier, "Get number of points"))
    if n == 0:
        out = call(manip, "Get resynthesis (overlap-add)")
        return np.asarray(out.values).reshape(-1).astype("float32")
    times = [call(ptier, "Get time from index", i) for i in range(1, n + 1)]
    vals = [call(ptier, "Get value at index", i) for i in range(1, n + 1)]
    mean_hz = sum(vals) / len(vals)
    for i in range(n, 0, -1):
        call(ptier, "Remove point", i)
    for t in times:
        call(ptier, "Add point", float(t), float(mean_hz))
    call([manip, ptier], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values).reshape(-1).astype("float32")


# ============================================================================================
# A/B rebuild (PSOLA_RECIPE.md §2/§7): grow the flip window to the WHOLE voiced rhyme and RESET its
# contour to the opposite tone band (not blind-multiply a CTC sliver). The pure-geometry helpers
# (voiced_rhyme_window / tone_level_residuals / build_flip_contour / the pts builder) carry NO
# parselmouth dependency — parselmouth is lazy-imported only inside the resynthesis fns
# (psola_set_contour / reimpose_contour) so the window/contour logic is unit-testable headless.
# ============================================================================================
def voiced_rhyme_window(pre, i, min_ms=60.0):
    """Grow the FULL voiced rhyme around TBU i from the F0 contour — NOT the ~20-60 ms CTC sliver wins[i]
    (the root cause of the inaudible flips). Seed = center of wins[i]; expand left/right over CONTIGUOUS
    non-NaN pre['f0'] frames until F0 goes unvoiced/silent each side; CLAMP so the window never crosses
    into the nearest aligned neighbour wins[i±1] (no bleed). Returns (t0,t1) seconds, or None if the
    grown rhyme has < min_ms of voiced speech. Pure numpy — no parselmouth."""
    import numpy as np
    wins = pre["wins"]
    if i < 0 or i >= len(wins) or wins[i] is None:
        return None
    f0 = np.asarray(pre["f0"], dtype="float64")
    times = np.asarray(pre["times"], dtype="float64")
    if f0.size == 0 or times.size != f0.size:
        return None
    w0, w1 = float(wins[i][0]), float(wins[i][1])
    seed_t = 0.5 * (w0 + w1)
    # clamp bounds = the nearest ALIGNED neighbour's facing edge (never bleed into a neighbour TBU)
    left_clamp = float(times[0])
    for j in range(i - 1, -1, -1):
        if wins[j] is not None:
            left_clamp = max(left_clamp, float(wins[j][1]))
            break
    right_clamp = float(times[-1]) + 1e-6
    for j in range(i + 1, len(wins)):
        if wins[j] is not None:
            right_clamp = min(right_clamp, float(wins[j][0]))
            break
    voiced = ~np.isnan(f0)
    # seed frame: nearest voiced frame to the window center, preferring INSIDE wins[i]
    in_win = np.where((times >= w0) & (times < w1) & voiced)[0]
    if in_win.size:
        seed = int(in_win[int(np.argmin(np.abs(times[in_win] - seed_t)))])
    else:
        cand = np.where(voiced & (times >= left_clamp) & (times < right_clamp))[0]
        if cand.size == 0:
            return None
        seed = int(cand[int(np.argmin(np.abs(times[cand] - seed_t)))])
    lo = seed
    while lo - 1 >= 0 and voiced[lo - 1] and times[lo - 1] >= left_clamp:
        lo -= 1
    hi = seed
    while hi + 1 < f0.size and voiced[hi + 1] and times[hi + 1] < right_clamp:
        hi += 1
    t0, t1 = float(times[lo]), float(times[hi])
    if (t1 - t0) * 1000.0 < float(min_ms):
        return None
    return (t0, t1)


def tone_level_residuals(pre):
    """(medH, medL, mid_ref, slope) for the CLEAN clip, all in the blind declination-removed residual space
    the classifier decides in. medH/medL = median residual of H / L TBUs; mid_ref = median over ALL TBUs;
    slope = the blind Theil-Sen declination (st/s) used to re-add declination when synthesizing a flip.
    medH/medL/mid_ref are NaN if that group is empty. Pure python (no parselmouth)."""
    try:
        from . import tone_f0_abs as f0a
    except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
        import tone_f0_abs as f0a
    sts = v2._tbu_semitones(pre)
    res, slope = f0a._blind_residuals(sts)
    tones = pre["tones"]
    Hs = [r for r, t in zip(res, tones) if r is not None and t == "H"]
    Ls = [r for r, t in zip(res, tones) if r is not None and t == "L"]
    allr = [r for r in res if r is not None]
    medH = _median(Hs) if Hs else float("nan")
    medL = _median(Ls) if Ls else float("nan")
    mid_ref = _median(allr) if allr else float("nan")
    return medH, medL, mid_ref, float(slope)


def build_flip_contour(t0, t1, r_target, slope, step=0.01):
    """Dense [(time_s, hz), ...] across [t0,t1] that RESETS the rhyme to residual r_target, with declination
    re-added: st_abs(t) = r_target + slope*t ; hz = 100 * 2**(st_abs/12)  (100 Hz ref matches
    v2._tbu_semitones, so the re-analyzed residual lands back at ~r_target). Pure python."""
    pts = []
    if step <= 0:
        step = 0.01
    t, end = float(t0), float(t1)
    while t <= end + 1e-9:
        st_abs = r_target + slope * t
        pts.append((t, 100.0 * (2.0 ** (st_abs / 12.0))))
        t += step
    return pts


def _orig_contour_pts(pre, t0, t1):
    """[(time_s, hz), ...] = the clip's OWN measured F0 over [t0,t1) voiced frames (the CORRECT-twin contour).
    Pure numpy — split out from reimpose_contour so it is testable without parselmouth."""
    import numpy as np
    f0 = np.asarray(pre["f0"], dtype="float64")
    times = np.asarray(pre["times"], dtype="float64")
    # t1-INCLUSIVE so the correct twin's trailing point lands at the SAME time as build_flip_contour's
    # (which loops while t <= t1) — both twins then carry an identical point grid at the rhyme edges and
    # differ ONLY in the in-rhyme pitch (PSOLA_RECIPE.md artifact-matching).
    return [(float(times[k]), float(f0[k])) for k in range(f0.size)
            if t0 <= times[k] <= t1 and not np.isnan(f0[k])]


# ----------------------------- parselmouth resynthesis (lazy import) -----------------------------
def psola_set_contour(wav, t0, t1, pts, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """DENSE pitch-tier RESET over [t0,t1): drop the existing pitch points in that range, lay down
    pts=[(t,hz),...], Replace, PSOLA-resynthesize. Modelled on psola_flatten's remove/add/Replace machinery
    (vs psola_shift_window's in-place 'Multiply frequencies'): this OVERWRITES the contour instead of scaling
    it, so the rhyme lands SQUARELY in the target band. Used for BOTH twins (flip = reset to opposite band;
    correct = reset to original values) so they carry an IDENTICAL resynthesis artifact. parselmouth lazy."""
    import numpy as np
    from parselmouth.praat import call
    manip = call(_snd(wav, sr), "To Manipulation", 0.01, f0min, f0max)
    ptier = call(manip, "Extract pitch tier")
    n = int(call(ptier, "Get number of points"))
    for idx in range(n, 0, -1):                       # high->low so indices stay valid after removal
        t = call(ptier, "Get time from index", idx)
        if t0 <= t < t1:
            call(ptier, "Remove point", idx)
    for (t, hz) in pts:
        if hz and hz > 0:
            call(ptier, "Add point", float(t), float(hz))
    call([manip, ptier], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values).reshape(-1).astype("float32")


def reimpose_contour(wav, pre, t0, t1, sr=SR, f0min=F0MIN, f0max=F0MAX):
    """CORRECT twin: re-impose the clip's OWN measured F0 over the SAME rhyme via the SAME dense-pitch-tier
    machinery (psola_set_contour) the flipped twin uses — so both twins share the identical artifact and the
    ONLY perceptible difference is the in-rhyme pitch. parselmouth lazy (inside psola_set_contour)."""
    return psola_set_contour(wav, t0, t1, _orig_contour_pts(pre, t0, t1), sr=sr, f0min=f0min, f0max=f0max)


# ----------------------------- the clip's own H-L spread -----------------------------
def measure_delta_HL(pre, late_frac=0.5):
    """median(H) - median(L) of the clip's DECLINATION-REMOVED per-TBU residuals — the flip UNIT, in the
    SAME space the classifier (tone_f0_abs) decides in, so k=1.0 really moves a syllable a full H-L tonal
    distance. (Sizing the flip in RAW semitones — as the first version did — UNDER-shoots, because
    downdrift compresses the raw H-L gap, so flips land short of the opposite tone band and the oracle
    under-detects. Fixed here.) None if the clip lacks both voiced H and voiced L TBUs."""
    try:
        from . import tone_f0_abs as f0a
    except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
        import tone_f0_abs as f0a
    sts = v2._tbu_semitones(pre, late_frac=late_frac)
    res, _slope = f0a._blind_residuals(sts)   # blind Theil-Sen detrend, no answer key
    tones = pre["tones"]
    Hs = [r for r, t in zip(res, tones) if r is not None and t == "H"]
    Ls = [r for r, t in zip(res, tones) if r is not None and t == "L"]
    if not Hs or not Ls:
        return None
    return float(_median(Hs) - _median(Ls))


def _median(xs):
    s = sorted(xs)
    k = len(s)
    if k == 0:
        return float("nan")
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


# ----------------------------- one-clip oracle -----------------------------
def run_oracle_clip(wav, sr, text, pre, base_scores, score_fns,
                    ks=(0.5, 1.0, 1.5), target_classes=("H", "L"), min_delta_HL=1.0,
                    tiny_st=0.2, sr_out=None, max_cands=None):
    """Flip eligible TBUs of ONE clip and record each meter's response.

    pre         : v2.precompute(wav, sr, text, ...) — supplies wins, tones; reused so the flip windows
                  match the meters' alignment.
    base_scores : {name: clean score dict} — the meters' verdict on the UNCHANGED clip (mid_ref already
                  frozen for I2 by the caller). Only TBUs BOTH meters got right are flipped (clean causal
                  test: a non-detection then means the flip wasn't read, not that the meter was already wrong).
    score_fns   : {name: fn(wav)->score dict} — re-scores a modified wav; pred[] aligned with pre['tones'].

    Returns a list of row dicts:
      flip rows : {kind:'flip', tbu, src, k, semitones, meter, pred_at_tbu, expect, detected,
                   false_flips, ff_total}
      ctrl rows : {kind:'control', tbu, control, meter, pred_at_tbu, src, flipped}
    """
    sr = sr_out or sr
    dHL = measure_delta_HL(pre)
    if dHL is None or dHL < min_delta_HL:
        return []
    tones, wins = pre["tones"], pre["wins"]
    names = list(score_fns)
    # eligible TBUs: target H or L, aligned, and EVERY meter got it right on the clean clip
    cands = [i for i, (t, w) in enumerate(zip(tones, wins))
             if t in target_classes and w is not None
             and all((base_scores[n]["pred"][i] if i < len(base_scores[n]["pred"]) else None) == t
                     for n in names)]
    if max_cands is not None:
        cands = cands[:max_cands]                          # bound the per-clip MMS-forward cost
    rows = []
    for i in cands:
        t0, t1 = wins[i]
        src = tones[i]
        sign = 1.0 if src == "L" else -1.0                 # flip away from the lexical pole
        expect = "H" if src == "L" else "L"
        for k in ks:
            st = sign * k * dHL
            y = psola_shift_window(wav, t0, t1, st, sr=sr)
            for n in names:
                s = score_fns[n](y)
                p = s["pred"][i] if i < len(s["pred"]) else None
                base_pred = base_scores[n]["pred"]
                ff_changed = sum(1 for j, (a, b) in enumerate(zip(base_pred, s["pred"]))
                                 if j != i and a is not None and b is not None and a != b)
                ff_total = sum(1 for j, (a, b) in enumerate(zip(base_pred, s["pred"]))
                               if j != i and a is not None and b is not None)
                rows.append(dict(kind="flip", tbu=i, src=src, k=k, semitones=st, meter=n,
                                 pred_at_tbu=p, expect=expect, detected=(p == expect),
                                 false_flips=ff_changed, ff_total=ff_total))
    # negative controls on the first half of candidates: must NOT change tone class
    for i in cands[:max(1, len(cands) // 2)]:
        t0, t1 = wins[i]
        controls = {"tiny+%.1fst" % tiny_st: psola_shift_window(wav, t0, t1, tiny_st, sr=sr),
                    "roundtrip": psola_roundtrip(wav, sr=sr)}
        for label, y in controls.items():
            for n in names:
                s = score_fns[n](y)
                p = s["pred"][i] if i < len(s["pred"]) else None
                rows.append(dict(kind="control", tbu=i, control=label, meter=n,
                                 pred_at_tbu=p, src=tones[i],
                                 flipped=(p is not None and p != tones[i])))
    return rows


# ----------------------------- pool + summarize across clips -----------------------------
def summarize_oracle(rows, instrument, full_k=None, trend_perm=2000, seed=0):
    """Aggregate pooled oracle rows for ONE instrument into the L0-5 bundle tone_layer0 consumes.

    detect_by_k      : {k -> detection rate}
    strong_detect    : detection rate at `full_k` (or the largest k present) — the L0-5 'strong detect'
    false_flip_rate  : changed-class-on-untouched / total-untouched, pooled
    trend_rho/trend_p: one-sided permutation Spearman of `detected` vs |semitones| over flip rows
    control_trip     : fraction of negative-control TBUs whose class changed (must stay low)
    n_flips          : number of flip observations
    full_k           : the k whose detection is the 'strong detect' (defaults to the largest k present)
    """
    flips = [r for r in rows if r["kind"] == "flip" and r["meter"] == instrument]
    ctrls = [r for r in rows if r["kind"] == "control" and r["meter"] == instrument]
    detect_by_k = {}
    for k in sorted({r["k"] for r in flips}):
        rk = [r for r in flips if r["k"] == k]
        detect_by_k[k] = (sum(r["detected"] for r in rk) / len(rk)) if rk else float("nan")
    if detect_by_k:
        strong_detect = detect_by_k.get(full_k, detect_by_k[max(detect_by_k)])
    else:
        strong_detect = float("nan")
    ff_changed = sum(r["false_flips"] for r in flips)
    ff_total = sum(r["ff_total"] for r in flips)
    false_flip_rate = (ff_changed / ff_total) if ff_total else float("nan")
    xs = [abs(r["semitones"]) for r in flips]
    ys = [1 if r["detected"] else 0 for r in flips]
    rho, p = L0.trend_increasing(xs, ys, n_perm=trend_perm, seed=seed) if len(flips) >= 4 else (float("nan"), 1.0)
    trip = (sum(r["flipped"] for r in ctrls) / len(ctrls)) if ctrls else float("nan")
    return {"detect_by_k": detect_by_k, "strong_detect": strong_detect,
            "false_flip_rate": false_flip_rate, "trend_rho": rho, "trend_p": p,
            "control_trip": trip, "n_flips": len(flips)}
