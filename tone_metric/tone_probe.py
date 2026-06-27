# coding=utf-8
# tone_probe.py — learned Yoruba tone scorer ("metric v3", the STRONG tier of Phase 0b).
#
# WHY: the quick tier (tone_eval_v2, threshold-on-ΔF0 transitions) capped at ~0.61 on TONE-GOLD audio
# (nb21: Bible 0.57 / SLR86 0.68 / per-class uniformly ~0.6) — a representation ceiling, not threshold
# placement, matching the literature (raw-F0 rules underperform learned features for tone everywhere).
# RECIPE (Osakuade & King, Speech Prosody 2026, arXiv 2604.07467 — verified): continuous SSL latents
# (AfriHuBERT) probed with a 1-layer LSTM (h=128) over VOWEL-ALIGNED segments reach weighted F1 0.92 for
# Yoruba tone on BibleTTS; discrete/quantized units drop to 0.66-0.83. The SSL latents already encode
# contextual pitch normalization (downdrift etc.) — that's what the threshold rule could never see.
#
# PIECES:
#   encode_latents(wav, sr, encoder, fe, device, layers) -> ({layer: [T,768] fp16 cpu}, dur_s)
#   tbu_segments(latents, wins, min_frames=2)            -> list[[t,768] | None] per TBU (abstention kept)
#   ToneProbe (LSTM h=128 -> mean-pool -> linear 3)      ; CLASSES = ["L","M","H"]
#   train_probe(...)  class-weighted CE, Adam, best-on-dev weighted-F1 checkpointing
#   probe_score(...)  same output contract as tone_eval_v2.score (accuracy/coverage/pred/target) so the
#                     Phase-3 dashboard can swap metrics without changes
#   save_probe/load_probe  (.pt with meta: layer, encoder id, classes, calibration provenance)
#
# Encoder + feature-extractor are INJECTED (HF AutoModel/AutoFeatureExtractor in prod, fakes in tests).
# AfriHuBERT: ajesujoba/AfriHuBERT — hubert, hidden 768, 12 layers, 20 ms frames (conv stride 320@16k).
import math

import torch
import torch.nn as nn

try:
    from . import tone_eval_v2 as v2
except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
    import tone_eval_v2 as v2

FRAME_SEC = 0.02                     # HuBERT conv stride 320 / 16000
CLASSES = ["L", "M", "H"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_ENCODER_ID = "ajesujoba/AfriHuBERT"


# ----------------------------- latent extraction -----------------------------
def encode_latents(wav, sr, encoder, fe, device, layers=(9,)):
    """One encoder forward; returns ({layer: [T,768] fp16 cpu}, n_seconds_at_16k).
    `encoder` must accept input_values and return .hidden_states (output_hidden_states=True);
    `fe` is the matching feature extractor. Layer 0 = conv output, 1..12 = transformer layers."""
    import numpy as np, librosa
    w = np.asarray(wav, dtype="float32")
    if w.ndim > 1:
        w = w.mean(-1)
    w16 = librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w
    iv = fe(w16, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        out = encoder(iv, output_hidden_states=True)
    return {L: out.hidden_states[L][0].half().cpu() for L in layers}, len(w16) / 16000.0


def tbu_segments(latents, wins, min_frames=2):
    """Slice per-TBU frame windows out of a [T,768] latent track. wins entries may be None (abstain);
    too-short windows also abstain. Returns list aligned with the TBU order."""
    T = latents.shape[0]
    segs = []
    for w in wins:
        if w is None:
            segs.append(None); continue
        i0 = max(0, int(w[0] / FRAME_SEC))
        i1 = min(T, int(math.ceil(w[1] / FRAME_SEC)))
        segs.append(latents[i0:i1] if i1 - i0 >= min_frames else None)
    return segs


# ----------------------------- the probe -----------------------------
class ToneProbe(nn.Module):
    def __init__(self, in_dim=768, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, len(CLASSES))
        self.in_dim, self.hidden = in_dim, hidden

    def forward(self, padded, lengths):
        """padded [B,Tmax,in_dim] float32, lengths [B] long -> logits [B,3] (mean over valid steps)."""
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.shape[1], device=out.device)[None, :] < lengths[:, None].to(out.device))
        pooled = (out * mask[..., None]).sum(1) / lengths[:, None].to(out.device).clamp(min=1)
        return self.head(pooled)


def _collate(segs, device):
    lengths = torch.tensor([s.shape[0] for s in segs], dtype=torch.long)
    assert int(lengths.min()) > 0, "zero-length segment reached _collate (use min_frames>=1 upstream)"
    Tmax = int(lengths.max())
    batch = torch.zeros(len(segs), Tmax, segs[0].shape[1], dtype=torch.float32)
    for i, s in enumerate(segs):
        batch[i, : s.shape[0]] = s.float()
    return batch.to(device), lengths.to(device)


@torch.no_grad()
def predict(probe, segs, device, bs=512):
    """segs: list of [t,768] (no Nones). Returns list of class strings."""
    probe.eval()
    preds = []
    for i in range(0, len(segs), bs):
        batch, lengths = _collate(segs[i:i + bs], device)
        preds.extend(CLASSES[int(k)] for k in probe(batch, lengths).argmax(-1))
    return preds


def weighted_f1(y_true, y_pred):
    """Support-weighted macro F1 over CLASSES (dependency-free). Returns (weighted_f1, per_class_f1)."""
    per, total = {}, len(y_true)
    wf1 = 0.0
    for c in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = f1
        wf1 += f1 * (tp + fn) / max(total, 1)
    return wf1, per


def train_probe(train_segs, train_labels, dev_segs, dev_labels, device, in_dim=768, hidden=128,
                epochs=8, bs=256, lr=1e-3, seed=0, log=print):
    """Class-weighted CE + Adam; keeps the best dev weighted-F1 state. Inputs: parallel lists of
    [t,768] tensors and 'H'/'M'/'L' labels (no Nones). Returns (probe, best_dev_f1, history)."""
    torch.manual_seed(seed)
    probe = ToneProbe(in_dim=in_dim, hidden=hidden).to(device)
    counts = {c: max(1, sum(1 for l in train_labels if l == c)) for c in CLASSES}
    w = torch.tensor([len(train_labels) / (len(CLASSES) * counts[c]) for c in CLASSES],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    y = torch.tensor([CLASS_TO_ID[l] for l in train_labels], dtype=torch.long)
    idx = list(range(len(train_segs)))
    best_f1, best_state, hist = -1.0, None, []
    import random as _rnd
    rng = _rnd.Random(seed)
    for ep in range(epochs):
        probe.train()
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            batch, lengths = _collate([train_segs[j] for j in sel], device)
            loss = crit(probe(batch, lengths), y[sel].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel)
        dev_pred = predict(probe, dev_segs, device)
        f1, per = weighted_f1(dev_labels, dev_pred)
        hist.append(dict(epoch=ep, loss=tot / len(idx), dev_f1=f1, per_class=per))
        log(f"[probe] ep {ep}: loss {tot/len(idx):.4f} | dev wF1 {f1:.3f} | " +
            " ".join(f"{c}:{per[c]:.2f}" for c in CLASSES))
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v_.detach().cpu().clone() for k, v_ in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe, best_f1, hist


# ----------------------------- scoring (dashboard-compatible) -----------------------------
def probe_score(wav, sr, text, probe, encoder, fe, asr=None, proc=None, device="cpu",
                emissions=None, n16=None, layer=9, min_frames=2):
    """Score one clip's tone against its diacritics with the probe. Returns the SAME KEY SET as
    tone_eval_v2.score_from_precomputed so dashboards can swap metrics — but note the SEMANTIC diffs:
    coverage here is per-TBU (v2's is per-transition; one abstained TBU costs 2 transitions there) and
    pred/target are H/M/L classes (v2's are R/level/F transitions). v2-only fields are emitted as
    neutral placeholders (n_trans, per_class={}, slope=0.0, backend=None). method is "probe" /
    "probe-abstain" — NEVER feed this into gates that expect "forced-align". Abstention on
    unaligned/too-short TBUs; read accuracy WITH coverage. emissions/n16 reuse a shared MMS forward."""
    import numpy as np
    tones = v2.tbu_seq(text)
    out = dict(accuracy=float("nan"), coverage=0.0, n_scored=0, n_tbu=len(tones),
               n_trans=max(0, len(tones) - 1), pred=[], target=tones, per_class={},
               slope=0.0, backend=None, method="probe-abstain")
    if not tones:
        return out
    wins = v2.forced_tbu_windows(wav, sr, text, asr, proc, device, emissions=emissions, n16=n16)
    if wins is None or len(wins) != len(tones):
        return out
    lat, _ = encode_latents(wav, sr, encoder, fe, device, layers=(layer,))
    segs = tbu_segments(lat[layer], wins, min_frames=min_frames)
    live = [(i, s) for i, s in enumerate(segs) if s is not None]
    pred = [None] * len(tones)
    if live:
        for (i, _), p in zip(live, predict(probe, [s for _, s in live], device)):
            pred[i] = p
    scored = [(p, t) for p, t in zip(pred, tones) if p is not None]
    out.update(pred=pred, n_scored=len(scored), coverage=len(scored) / len(tones), method="probe")
    if scored:
        out["accuracy"] = float(np.mean([p == t for p, t in scored]))
    return out


# ----------------------------- persistence -----------------------------
def save_probe(probe, path, meta):
    torch.save(dict(state=probe.state_dict(), in_dim=probe.in_dim, hidden=probe.hidden, meta=meta), path)


def load_probe(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    probe = ToneProbe(in_dim=blob["in_dim"], hidden=blob["hidden"]).to(device)
    probe.load_state_dict(blob["state"])
    probe.eval()
    return probe, blob.get("meta", {})
