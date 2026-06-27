# coding=utf-8
# GRPO reward for Yoruba TONE on Qwen3-TTS. GATED design (not additive): TONE is the only climber;
# intelligibility (CER), voice (SSIM), length, voicing, and tone-non-constancy are HARD GATES. Rationale
# (design workflow + probe): tone and CER are tradeable per-sample (8/20 prompts have negative within-prompt
# corr; best-tone samples reach CER ~0.2) so an *additive* CER guard lets GRPO buy tone with intelligibility
# (Prosody-RL arXiv:2509.18531 documents this monotone/quality collapse). Gates forbid that trade.
#
# Reward (per sample):
#   if any gate fails           -> 0.0                      (the group floor)
#   else R_tone + 0.05*R_ssim   where
#     R_tone = clip((tone_acc - TONE_FLOOR)/(1 - TONE_FLOOR), 0, 1)   # monotone-floor -> 0; perfect -> 1
#     R_ssim = clip((ssim - SSIM_BASE)/(1 - SSIM_BASE), 0, 1)         # tiny voice tie-break only
# Gates: voiced-fraction >= MIN_VOICED ; tone alignment == "forced-align" (proportional fallback is
#   gameable) ; CER <= CER_GATE ; length in [LEN_LO, LEN_HI]*expected ; predicted tone seq non-constant.
#
# TONE_FLOOR clamps below-floor samples to r_tone=0. Because the GRPO advantage subtracts the group MEAN, the
# floor's absolute value is NOT a reward baseline (a constant cancels in the advantage) — its only real effect
# is the CLAMP, which destroys within-group ORDERING of low-tone samples (0.20 and 0.35 both -> 0 -> no gradient
# between them). A HIGH floor therefore throws away gradient and risks zero-variance (skipped) groups -> silent
# no-op. Keep it LOW so r_tone preserves the full within-group ordering; require_tone_variety blocks monotone
# gaming. (Raise it later only if the model games tone toward a single dominant class.)
from dataclasses import dataclass, field


@dataclass
class RewardCfg:
    tone_floor: float = 0.15         # LOW on purpose: floor only CLAMPS (advantage subtracts the mean) -> low
                                     # floor keeps within-group ordering -> dense gradient. See note above.
    ssim_base: float = 0.30          # ECAPA cosine a non-matching voice already scores; nudge is relative to it
    ssim_weight: float = 0.05        # voice tie-break only (tone is the climber)
    min_voiced: float = 0.35         # gate: F0-on-silence / near-silent hack
    require_forced_align: bool = True  # gate: proportional fallback is gameable -> never rewardable
    cer_gate: float = 0.25           # gate: KEEP the best-TONE samples (probe: they reach CER 0.19-0.22);
                                     # 0.18 was too tight (rejected exactly the high-tone samples). Blocks worse.
    len_lo: float = 0.6              # gate: dur >= len_lo * expected (anti-truncation)
    len_hi: float = 1.6              # gate: dur <= len_hi * expected (anti no-EOS runaway)
    s_per_char: float = 0.157        # expected seconds per char (probe: 0.157, CV 0.17)
    require_tone_variety: bool = True  # gate: predicted tone seq must be non-constant (anti monotone-collapse)


def expected_dur(text, s_per_char=0.157):
    return max(0.3, s_per_char * len(text))


def length_ok(dur_s, text, cfg: RewardCfg):
    exp = expected_dur(text, cfg.s_per_char)
    return cfg.len_lo * exp <= dur_s <= cfg.len_hi * exp


def score_from_metrics(tone_acc, method, voiced, pred, cer, ssim, dur, text, cfg: RewardCfg):
    """PURE reward from already-computed metrics (unit-testable). Returns dict(reward, gated, gate_fail, ...).
    tone_acc may be NaN (no-voiced); pred is the predicted H/M/L list; method in
    {forced-align, proportional, no-voiced}; ssim in [0,1] (or None -> treated as no nudge)."""
    import math
    gate_fail = None
    if not (voiced is not None and voiced >= cfg.min_voiced):
        gate_fail = "voiced"
    elif cfg.require_forced_align and method != "forced-align":
        gate_fail = "align"
    elif tone_acc is None or (isinstance(tone_acc, float) and math.isnan(tone_acc)):
        gate_fail = "tone_nan"
    elif not (cer <= cfg.cer_gate):
        gate_fail = "cer"
    elif not length_ok(dur, text, cfg):
        gate_fail = "length"
    elif cfg.require_tone_variety and len(set(pred or [])) < 2:
        gate_fail = "monotone"
    if gate_fail is not None:
        return dict(reward=0.0, gated=True, gate_fail=gate_fail, tone=tone_acc, cer=cer,
                    ssim=ssim, dur=dur, method=method, voiced=voiced)
    r_tone = min(1.0, max(0.0, (tone_acc - cfg.tone_floor) / max(1e-6, 1.0 - cfg.tone_floor)))
    r_ssim = 0.0
    if ssim is not None:
        r_ssim = min(1.0, max(0.0, (ssim - cfg.ssim_base) / max(1e-6, 1.0 - cfg.ssim_base)))
    reward = r_tone + cfg.ssim_weight * r_ssim
    return dict(reward=float(reward), gated=False, gate_fail=None, tone=tone_acc, cer=cer,
                ssim=ssim, dur=dur, method=method, voiced=voiced, r_tone=r_tone, r_ssim=r_ssim)


class RewardModels:
    """Holds the scoring models so MMS/ECAPA load once. Yoruba CER (tone-blind) + ECAPA speaker-sim.
    Mirrors the probe's MMS loading (nb17 §7). ECAPA is optional (SSIM is only a 0.05 nudge)."""

    def __init__(self, device="cuda", asr_id="facebook/mms-1b-all",
                 ecapa_id="speechbrain/spkrec-ecapa-voxceleb"):
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        self.device = device
        self.asr_proc = AutoProcessor.from_pretrained(asr_id)
        self.asr_proc.tokenizer.set_target_lang("yor")
        asr = Wav2Vec2ForCTC.from_pretrained(asr_id, target_lang="yor", ignore_mismatched_sizes=True)
        asr.load_adapter("yor")
        self.asr = asr.to(device).eval()
        self.ecapa = None
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.ecapa = EncoderClassifier.from_hparams(source=ecapa_id, run_opts={"device": device})
        except Exception as e:
            print(f"[grpo_reward] ECAPA unavailable ({type(e).__name__}); SSIM nudge disabled.", flush=True)

    def _to16k(self, wav_np, sr):
        import numpy as np, librosa
        w = np.asarray(wav_np, dtype="float32")
        if w.ndim > 1:
            w = w.mean(-1)
        return librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w

    def transcribe(self, wav_np, sr):
        import torch
        logits, _ = self.asr_logits(wav_np, sr)
        return self.decode_logits(logits)

    def asr_logits(self, wav_np, sr):
        """ONE MMS-1b forward. Returns (raw logits [T,V] on the ASR device, n16=#samples at 16 kHz). CER-decode
        and forced-align both consume this -> halves the per-sample ASR cost (was two identical 1B forwards).
        Kept ON-DEVICE (not .cpu()) so consumers' argmax/log_softmax run on the SAME device as the old two-
        forward code -> bit-identical CER and forced-align (CPU vs CUDA log_softmax differ at ULP scale)."""
        import torch
        w = self._to16k(wav_np, sr)
        iv = self.asr_proc(w, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logits = self.asr(iv).logits[0].float()             # [T,V] on device; raw (consumers argmax / log_softmax)
        return logits, int(w.shape[0])

    def decode_logits(self, logits):
        """Greedy CTC decode of cached logits (== old transcribe path; log_softmax is monotone so argmax matches)."""
        import torch
        ids = torch.argmax(logits, dim=-1).unsqueeze(0)
        return self.asr_proc.batch_decode(ids)[0]

    @staticmethod
    def cer(ref, hyp):
        import unicodedata, re
        def norm(s):
            s = unicodedata.normalize("NFD", s)
            s = "".join(c for c in s if not unicodedata.combining(c))
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
        r, h = norm(ref), norm(hyp)
        if not r:
            return 0.0 if not h else 1.0
        dp = list(range(len(h) + 1))
        for i, rc in enumerate(r, 1):
            prev = dp[0]; dp[0] = i
            for j, hc in enumerate(h, 1):
                cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rc != hc)); prev = cur
        return dp[len(h)] / len(r)

    def ecapa_embed(self, wav_np, sr):
        import torch
        if self.ecapa is None:
            return None
        w = torch.from_numpy(self._to16k(wav_np, sr)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self.ecapa.encode_batch(w).squeeze()
        return torch.nn.functional.normalize(e, dim=-1)

    def ssim(self, wav_np, sr, ref_wav_np, ref_sr):
        import torch
        a = self.ecapa_embed(wav_np, sr)
        b = self.ecapa_embed(ref_wav_np, ref_sr)
        if a is None or b is None:
            return None
        return float(((torch.dot(a, b).clamp(-1, 1) + 1) / 2).item())   # cosine -> [0,1]


def compute_reward(wav, fs, text, ref_wav, ref_sr, rm: RewardModels, cfg: RewardCfg):
    """Full reward over a generated (wav, fs) for `text`, cloning ref_wav. Computes tone/CER/SSIM/length
    then applies the gated scoring. Imports tone_eval lazily (same dir). The MMS-1b forward is run ONCE
    (asr_logits) and shared by CER-decode + forced-align (was two identical 1B forwards per sample)."""
    import numpy as np
    try:
        from . import tone_eval
    except ImportError:  # flat-import fallback (legacy notebooks on sys.path)
        import tone_eval
    logits, n16 = rm.asr_logits(wav, fs)            # ONE shared MMS forward
    ta = tone_eval.tone_accuracy(wav, fs, text, asr=rm.asr, proc=rm.asr_proc, device=rm.device,
                                 emissions=logits, n16=n16)
    cer = rm.cer(text, rm.decode_logits(logits))
    ssim = rm.ssim(wav, fs, ref_wav, ref_sr) if ref_wav is not None else None
    dur = len(np.asarray(wav).reshape(-1)) / fs
    return score_from_metrics(ta["accuracy"], ta["method"], ta["n_voiced_frac"], ta["pred"],
                              cer, ssim, dur, text, cfg)
