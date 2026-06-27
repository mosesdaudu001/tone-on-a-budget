"""tone_metric — a reference-free lexical-tone fidelity metric for TTS.

Headline metric: ``f0_abs_score`` (the F0-absolute ``tone_i2``). Heavy extras
(``probe_score``/``RewardModels``) lazily import torch, so plain ``import tone_metric``
stays dependency-light. Install extras with ``pip install "tone-metric[probe]"`` /
``[reward]`` / ``[full]``.
"""
import importlib

__version__ = "0.1.0"

# public name -> submodule providing it (imported lazily on first access)
_EXPORTS = {
    # --- core (numpy / scipy / librosa / parselmouth / pyworld) ---
    "f0_abs_score": "tone_f0_abs",          # the tone_i2 score (I2)
    "tone_transition_score": "tone_eval_v2",
    "tone_accuracy": "tone_eval",
    "target_tone_seq": "tone_eval",
    "psola_flatten": "tone_oracle",         # PSOLA pitch-flattening oracle
    "psola_roundtrip": "tone_oracle",
    "run_oracle_clip": "tone_oracle",
    "summarize_oracle": "tone_oracle",
    # --- torch extras: pip install "tone-metric[probe]" / "[reward]" ---
    "probe_score": "tone_probe",            # I1 AfriHuBERT tone probe
    "load_probe": "tone_probe",
    "ToneProbe": "tone_probe",
    "RewardModels": "grpo_reward",          # CER (MMS) + SSIM (ECAPA)
    "compute_reward": "grpo_reward",
}
__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module("." + _EXPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
