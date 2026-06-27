# OmniVoice — Yoruba tone probe

Zero-shot evaluation of **[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)** as a candidate
TTS base for Yoruba, to decide whether to pivot off Qwen3-TTS. **No training happens here.**

## Why this exists
The Qwen3-TTS Yoruba finetune failed in a *bracketed* way: high-LR runs **collapsed** (autoregressive
exposure-bias babble / EOS-runaway), low-LR runs **underfit** to flat, toneless speech. OmniVoice is
worth probing because its generation is **non-autoregressive masked diffusion** with a rule-based
duration estimator and **no learned EOS** — so the *collapse* half of that failure is structurally
impossible. But OmniVoice shares the same no-G2P / no-tone-modeling design, and **Yoruba is its
weakest target** (15.66 h pretraining; reported CER **21.37 > ground truth**). So this probe answers
exactly one question: **does OmniVoice carry real Yoruba tone zero-shot, or is it flat too?**

## What the notebook does
`01_omnivoice_yoruba_probe.ipynb`:
1. Installs `omnivoice` + **pip-installs the public `tone-metric` package** (the tone modules);
   clones OmniVoice (to introspect its real API).
2. Zero-shot synthesizes held-out Yoruba — **tone minimal pairs** (e.g. `ọkọ́` hoe / `ọkọ̀` vehicle /
   `ọkọ` husband, realized in the carrier `Mo rí ___ ní ọjà.`) plus a few longer held-out SLR86 lines
   — voice-cloned from **one clean single-speaker BibleTTS reference**.
3. Scores every wav with the project's existing gate (identical to nb26 `dashboard_eval`):
   - **I2 `tone_i2`** — F0-absolute H/M/L tone meter (`tone_f0_abs.py`), the decisive signal
   - **I1 `i1_acc`** — AfriHuBERT tone probe (quality / anti-collapse, `tone_probe.py`)
   - **CER** (MMS-1b-all `yor`, tone-blind), **SSIM** (ECAPA), **len_ratio** — via `grpo_reward.RewardModels`
4. Uploads wavs + a `results.v1.json` to `s3://codec-audio-data/tts_data/yoruba/omnivoice_probe/`
   for a **native ear** (the decisive check) and prints a **GO / NO-GO-FLAT / MIXED** verdict.

## How to run
1. Open in **Colab on a T4 or L4** (inference only; **not** the A100). sdpa/eager attention only.
2. Run the install cell; let the kernel restart (the `numpy<2` pin).
3. Provide secrets via Colab `userdata`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`.
   (No `GITHUB_TOKEN` needed — the tone modules now install from the public `tone-metric` package.)
4. Run top to bottom. The verdict cell writes `tts_data/yoruba/omnivoice_probe/results.v1.json`.
5. **Listen** to the uploaded wavs — the native ear overrides the automatic gate.

## Go / no-go criterion
- **GO** — `tone_i2` clears chance (~0.33) by ≥0.04 at coverage ≥0.70 **and** CER is sane (well below
  OmniVoice's reported Yoruba CER 21.37) **and** SSIM / len_ratio pass. OmniVoice already shows real
  tone zero-shot → a finetune (`examples/run_finetune.sh`: Qwen3-0.6B backbone, 8 codebooks, lr 1e-5,
  5000 steps, single A100) is worth attempting.
- **NO-GO-FLAT** — `tone_i2` ≈ chance (monotone). Same failure class as the collapsed Qwen3-TTS base;
  this is a **data-side** problem (clean tone-marked native audio), not a base-model problem. Pivoting
  alone won't install tone — return to the data-side fix in `tts/YORUBA_WAY_FORWARD.md`.

## ⚠️ Flywheel / license caveat — read before committing to OmniVoice
OmniVoice's **model weights are Apache-2.0**, but it bundles the **Higgs / Boson audio codec**, whose
license carries a **non-commercial, output-restriction** clause ("do not use outputs to improve any
other model"). The TTS exists to be a **synthetic-data flywheel** feeding the downstream Moshi/
PersonaPlex duplex model — so OmniVoice-generated audio reused as training data may be **contaminated**
by the Boson license. Resolve this licensing question **before** adopting OmniVoice as a flywheel data
generator. (Qwen3-TTS-Tokenizer is Apache-2.0 by contrast. A codec swap for the final encode/decode may
or may not de-risk it — that is a separate question.)

## Files & artifacts
| | |
|---|---|
| `01_omnivoice_yoruba_probe.ipynb` | the probe (this dir) |
| Scoring modules | `../tone_metric/{tone_f0_abs,tone_probe,grpo_reward}.py` |
| I2 calibration | `s3://codec-audio-data/tts_data/yoruba/tone_v2/f0_abs_calibration.v1.json` (nb27) |
| Tone probe `.pt` | `s3://codec-audio-data/tts_data/yoruba/tone_v2/tone_probe_L*` (latest; nb22) |
| Held-out eval texts | `s3://codec-audio-data/tts_data/yoruba_gold/holdouts.v1.json` (`eval_texts`, 200) |
| Minimal pairs | `../tone_metric/minimal_pairs_draft.json` (DRAFT — `verified:false`) |
| Clean ref source | `s3://codec-audio-data/tts_data/yoruba_gold/s1_train.v2.jsonl` (`source=="bible"`) |
| Probe outputs | `s3://codec-audio-data/tts_data/yoruba/omnivoice_probe/` (wavs + `results.v1.json`) |

## OmniVoice API (verified against raw `master` source)
```python
from omnivoice import OmniVoice
import torch
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
audio = model.generate(text="<Yoruba text>", language="yo",   # 'yo', NOT 'yor'
                       ref_audio="ref.wav", ref_text="<ref transcript>")  # -> list[np.ndarray] @ 24 kHz
```
Finetune (when GO): `examples/run_finetune.sh` → `python -m omnivoice.scripts.extract_audio_tokens`
then `accelerate launch -m omnivoice.cli.train` with `examples/config/train_config_finetune.json`.
Manifest JSONL: `{"id", "audio_path", "text", "language_id": "yo"}` (id/audio_path/text mandatory).

## CONFIRM-AT-RUNTIME notes
The notebook self-verifies the riskiest assumptions at runtime (it clones OmniVoice and introspects):
the `generate()` signature, `language="yo"` resolution, `model.sampling_rate==24000`, and the
`transformers>=5.3` (OmniVoice) vs `>=4.46` (gate stack) coexistence. If MMS-1b / AfriHuBERT loading
breaks under transformers 5.3, generate the wavs here and **score in a second session** with a
compatible pin — the wavs are already on S3.

---

## `03_omnivoice_yoruba_finetune.ipynb` — finetune + data-efficiency ablation

Fine-tunes **k2-fsa/OmniVoice** on clean Yoruba via the official two-stage pipeline
(`extract_audio_tokens` → `accelerate launch -m omnivoice.cli.train`) and **doubles as the Paper-1
data-efficiency ablation**. Runs on **1×A100 80GB** (training — NOT a T4/L4 notebook).
**sdpa attention only** (the stock finetune config defaults to `flex_attention`, which we override).

### One knob, two jobs
`HOURS_BUDGET` in §3:
- `None` → all clean data (~33 h bible+slr86); single full finetune = pipeline validation.
- `1.0 / 5.0 / 15.0 / 30.0` → cap train audio to N hours; re-run end-to-end for each curve point.

The cap trims the **input JSONL before Stage 0** (the only mechanism — OmniVoice has no hours/samples
config knob) and **prints exactly what was dropped** (no silent truncation). `STEPS_OVERRIDE` holds
optimisation fixed across budgets for a clean x-axis. `INCLUDE_NAIJAVOICES` (default `False`) is gated
off because NaijaVoices rows ship pre-encoded codes, not per-utterance training wavs.

### What it does
1. Installs `omnivoice` + `accelerate` + `webdataset` + the `tone-metric` package; restarts (numpy<2).
2. Clones OmniVoice and **introspects the real finetune surface at runtime** (config keys, CLI flags,
   `audio_path` field, `language_id="yo"`) — every recon assumption is re-asserted, not trusted.
3. Streams `s1_train.v2.jsonl` (`source=="bible"` + slr86), drops the 200 frozen `holdouts.v1.json`
   eval texts, downloads wavs from `wav_biblefull/`, writes manifests with exact keys
   `{id, audio_path, text, language_id:"yo"}`.
4. **Stage 0** tokenizes with the Higgs codec → WebDataset shards + `data.lst`. **Stage 1**
   `accelerate launch -m omnivoice.cli.train` with a derived config (`init_from_checkpoint="k2-fsa/OmniVoice"`,
   `attn_implementation="sdpa"`, lr 1e-5, bf16, 1 GPU); the FULL config is printed before launch.
5. Uploads the checkpoint → `s3://codec-audio-data/tts_checkpoints/omnivoice_yoruba/<hours>/`.
6. **Evals on the identical nb01 gate** and **appends a row to the persistent data-efficiency table**
   `tts_data/yoruba/omnivoice_ablation/data_efficiency.v1.json` — the Paper-1 ablation artifact
   (records `train_hours_actual`, not the budget).

### AR-vs-non-AR is a SEPARATE axis
This notebook owns the **data-efficiency** axis for the non-AR (masked-diffusion) model. The
**AR-vs-non-AR architecture** axis is owned by the Qwen3-TTS runs (nb26 `s1_verdict.json`). Both use the
**same I2 tone meter + same frozen eval set**, so numbers line up in a 2×N table at matched hours; do
not rebuild the AR side here. Cross-architecture differences (codec, backbone, lr) are uncontrolled —
frame that comparison as suggestive, the per-arch curve as clean.

⚠️ **Higgs/Boson codec license** (Stage 0 runs `eustlb/higgs-audio-v2-tokenizer`): finetuning OmniVoice
is *permitted* (improving a derivative), but using its *output* to train the duplex flywheel is blocked
— see `BOSON_LICENSE_WAIVER_EMAIL.md`.
