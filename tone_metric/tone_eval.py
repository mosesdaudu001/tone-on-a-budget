# coding=utf-8
# tone_eval.py — per-syllable Yoruba TONE accuracy (F0-based). CER (MMS-yor) is TONE-BLIND; this is the
# metric that actually sees tone, so it's the success signal for tone adaptation + the future GRPO reward.
#
# DESIGN (chosen 2026-06-06):
#   - REFERENCE tones come from the transcript DIACRITICS: acute=High, grave=Low, macron/unmarked=Mid
#     (Yoruba). One tone-bearing unit (TBU) per vowel (and any tone-marked syllabic nasal).
#   - ALIGNMENT = "ASR timestamps": CTC forced-alignment of the (tone-stripped) target to the audio via
#     the MMS-yor emissions (torchaudio.functional.forced_align) -> per-vowel time window. If that path
#     is unavailable/fails, FALL BACK to splitting the voiced span into N equal TBU windows (robust, no
#     ASR). MFA forced alignment is the deferred upgrade once we have intelligible speech.
#   - CLASSIFY each TBU by its median F0 in SEMITONES relative to the utterance median: > +THRESH = H,
#     < -THRESH = L, else M. v1 = a RELATIVE signal to compare runs; THRESH needs calibration on real
#     Yoruba audio (start 0.8 st). F0 via librosa.pyin (no new heavy dep).
#
# PUBLIC: target_tone_seq(text) -> ["H","M","L",...] ; tone_accuracy(wav, sr, text, asr=None, proc=None,
#         device="cpu", thresh_st=0.8) -> dict(accuracy, n, target, pred, method, ...)
import math
import unicodedata as ud

# Yoruba oral vowels (NFC, incl. dotted ẹ ọ). Tone marks are combining (NFD).
_VOWELS = set("aeiou") | {"ẹ", "ọ"}
_ACUTE, _GRAVE, _MACRON = "́", "̀", "̄"     # high, low, mid
_TONE_MARKS = {_ACUTE: "H", _GRAVE: "L", _MACRON: "M"}


def _tbu_tone(ch):
    """If `ch` is a tone-bearing unit (vowel, or any char carrying a tone mark) return its tone H/M/L,
    else None. A bare vowel (no mark) is Mid (Yoruba mid tone is usually unmarked)."""
    d = ud.normalize("NFD", ch)
    base = "".join(c for c in d if not (0x0300 <= ord(c) <= 0x036F))
    marks = [c for c in d if 0x0300 <= ord(c) <= 0x036F]
    tone = next((_TONE_MARKS[m] for m in marks if m in _TONE_MARKS), None)
    is_vowel = ud.normalize("NFC", base).lower() in _VOWELS
    if tone is not None:
        return tone                # explicitly toned (vowel or syllabic nasal)
    if is_vowel:
        return "M"                 # unmarked vowel -> Mid
    return None


def _units(text):
    """Group each base char with its TRAILING combining marks, so a standalone combining tone mark
    (the unavoidable `ọ̀` = ọ + U+0300 case) attaches to its vowel instead of becoming its own unit."""
    units = []
    for ch in text:
        if units and 0x0300 <= ord(ch) <= 0x036F:
            units[-1] += ch
        else:
            units.append(ch)
    return units


def target_tone_seq(text):
    """Ordered tone (H/M/L) of every tone-bearing unit in `text`. Robust to NFC/NFD and standalone
    combining marks (the dotted-vowel+tone case)."""
    return [t for t in (_tbu_tone(u) for u in _units(text)) if t is not None]


# ----------------------------- F0 + classification -----------------------------
def extract_f0(wav, sr, fmin=60.0, fmax=400.0):
    """Return (f0_hz, frame_times_s); f0 is NaN where unvoiced. Prefers pyworld.harvest (C-fast, ~10-20x
    faster than librosa.pyin — pyin dominated the GRPO step time); falls back to pyin if pyworld is absent.
    NOTE: switching the F0 backend shifts tone-accuracy numbers slightly vs the pyin probe -> recalibrate
    RewardCfg.tone_floor on the first GRPO batch (it's relative within a group, so the gradient is fine)."""
    import numpy as np
    w = np.asarray(wav, dtype="float64")
    if w.ndim > 1:
        w = w.mean(-1)
    try:
        import pyworld as pw
        f0, t = pw.harvest(np.ascontiguousarray(w), sr, f0_floor=fmin, f0_ceil=fmax, frame_period=20.0)
        return np.where(f0 > 0, f0, np.nan), t
    except Exception:
        import librosa
        hop = 256
        f0, _, _ = librosa.pyin(w.astype("float32"), fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=1024, hop_length=hop)
        return f0, librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)


def _voiced_span(f0, times):
    import numpy as np
    v = np.where(~np.isnan(f0))[0]
    if len(v) == 0:
        return None
    return float(times[v[0]]), float(times[v[-1]])


def _classify_window(f0, times, t0, t1, median_f0, thresh_st):
    """Median F0 in [t0,t1] -> H/M/L in semitones relative to the utterance median (or None if unvoiced)."""
    import numpy as np
    m = (times >= t0) & (times <= t1) & (~np.isnan(f0))
    if not m.any():
        return None
    seg = float(np.nanmedian(f0[m]))
    st = 12.0 * math.log2(max(seg, 1e-6) / max(median_f0, 1e-6))
    return "H" if st > thresh_st else ("L" if st < -thresh_st else "M")


def _proportional_windows(span, n):
    """Split a voiced [t0,t1] span into n equal TBU windows (the no-ASR fallback)."""
    t0, t1 = span
    step = (t1 - t0) / max(n, 1)
    return [(t0 + i * step, t0 + (i + 1) * step) for i in range(n)]


def _forced_windows(wav, sr, text, asr, proc, device, emissions=None, n16=None):
    """ASR-timestamp path: CTC forced-align the tone-stripped target -> one time window per TBU.
    Returns a list of (t0,t1) the SAME length/order as target_tone_seq(text), or None on any failure
    (caller falls back to proportional). Best-effort; validated in Colab (needs torchaudio+MMS).
    If `emissions` (raw MMS logits [T,V], cpu) + `n16` (#16 kHz samples) are supplied, the ASR forward is
    REUSED from the caller (compute_reward shares one forward with CER) instead of run a second time."""
    try:
        import numpy as np, torch, librosa
        import torchaudio.functional as AF
        # tone-stripped, lowercased target restricted to the ASR vocab
        flat = "".join(c for c in ud.normalize("NFD", text) if not (0x0300 <= ord(c) <= 0x036F))
        flat = ud.normalize("NFC", flat).lower()
        vocab = proc.tokenizer.get_vocab()
        blank = proc.tokenizer.pad_token_id if proc.tokenizer.pad_token_id is not None else 0
        # emissions (log-probs) over frames at 16 kHz — reuse the shared forward if provided, else run it
        if emissions is not None and n16 is not None:
            # log_softmax ON the emissions' device (matches the old GPU path bit-for-bit), THEN .cpu() for forced_align
            emis = torch.log_softmax(emissions.float(), dim=-1).cpu()      # [T, V]; raw logits -> log-probs
            n_samp16 = int(n16)
        else:
            w = np.asarray(wav, dtype="float32"); w = w.mean(-1) if w.ndim > 1 else w
            w16 = librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w
            iv = proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                emis = torch.log_softmax(asr(iv).logits, dim=-1)[0].cpu()  # [T, V]
            n_samp16 = int(w16.shape[0])
        # token ids for each CHAR of flat (chars not in vocab -> skipped, with index bookkeeping)
        ids, char_idx = [], []
        for j, ch in enumerate(flat):
            tid = vocab.get(ch, vocab.get("|" if ch == " " else ch))
            if tid is None or tid == blank:
                continue
            ids.append(tid); char_idx.append(j)
        if len(ids) < 2:
            return None
        targets = torch.tensor([ids], dtype=torch.int32)
        aligned, scores = AF.forced_align(emis.unsqueeze(0), targets, blank=blank)
        spans = AF.merge_tokens(aligned[0], scores[0])                     # one span per target id
        sec_per_frame = n_samp16 / 16000.0 / emis.shape[0]
        # map char-position -> (t0,t1); then keep only TBU chars in original `text` order
        char_time = {}
        for sp, cj in zip(spans, char_idx):
            char_time[cj] = (sp.start * sec_per_frame, sp.end * sec_per_frame)
        # TBUs are positions in `flat` whose char is a vowel/toned unit; align to target_tone_seq order
        wins = []
        for j, ch in enumerate(flat):
            if _tbu_tone(ch) is not None:
                wins.append(char_time.get(j))
        # need a window for every TBU; if some are missing (out-of-vocab), bail to fallback
        if not wins or any(wn is None for wn in wins):
            return None
        return wins
    except Exception:
        return None


def tone_accuracy(wav, sr, text, asr=None, proc=None, device="cpu", thresh_st=0.8,
                  emissions=None, n16=None):
    """Per-TBU Yoruba tone accuracy of `wav` vs the tones implied by `text`'s diacritics.
    Returns dict(accuracy, n, target, pred, method, n_voiced_frac). asr/proc = MMS-yor (Wav2Vec2ForCTC,
    AutoProcessor); if omitted or alignment fails, uses the proportional-voiced-split fallback.
    `emissions`/`n16`: optional precomputed MMS logits to reuse the caller's ASR forward (see _forced_windows)."""
    import numpy as np
    target = target_tone_seq(text)
    n = len(target)
    out = dict(accuracy=float("nan"), n=n, target=target, pred=[], method="none", n_voiced_frac=0.0)
    if n == 0:
        return out
    f0, times = extract_f0(wav, sr)
    voiced = float(np.mean(~np.isnan(f0))) if len(f0) else 0.0
    out["n_voiced_frac"] = voiced
    span = _voiced_span(f0, times)
    if span is None:
        out["method"] = "no-voiced"
        return out                          # silent/mumbled audio -> no F0 -> can't score tone
    median_f0 = float(np.nanmedian(f0))
    wins, method = None, "proportional"
    if asr is not None and proc is not None:
        wins = _forced_windows(wav, sr, text, asr, proc, device, emissions=emissions, n16=n16)
        if wins is not None and len(wins) == n:
            method = "forced-align"
    if wins is None or len(wins) != n:
        wins = _proportional_windows(span, n)
        method = "proportional"
    pred = [_classify_window(f0, times, t0, t1, median_f0, thresh_st) or "M" for (t0, t1) in wins]
    acc = float(np.mean([p == t for p, t in zip(pred, target)]))
    out.update(accuracy=acc, pred=pred, method=method)
    return out
