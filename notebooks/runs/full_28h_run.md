# OmniVoice Yoruba finetune — run record: `full` (28.5 h)

**Date** 2026-06-25 · **GPU** A100-80GB · **checkpoint** `s3://codec-audio-data/tts_checkpoints/omnivoice_yoruba/full/`
(9 files incl. `model.safetensors` + `optimizer.bin` → resumable). Recovered from a truncated Colab
download; raw outputs were salvaged via brace-matched cell recovery.

## Config (actually used)
backbone Qwen/Qwen3-0.6B · lr 1e-5 · **steps 5000** · **batch_tokens 16384** · grad_accum 1 · bf16 · sdpa ·
seed 4242 · num_audio_codebook 8 · language_ratio 1.0 · warmup_ratio 0.01

## Data
10,731 clips ≈ **28.5 h** clean (BibleTTS + SLR86, `s1_train.v2.jsonl`) · **19 epochs** · holdout texts dropped.

## Training
finished 5000 steps in ~1h05m (1.42 it/s) · final train/loss 3.30 · **eval loss 3.17** ·
checkpoints every 500 (all 10 kept). ⚠️ Colab truncated stdout to the last 5000 lines → only the
steps ~2600→5000 curve survived (eval-loss points: 3000=3.44, …, 5000=3.17, declining).

## Eval on the nb01 tone gate (held-out, same I2 meter as zero-shot)
| mode | n | cer | tone_i2 | ssim | i1_acc | len_ratio |
|---|---|---|---|---|---|---|
| clone (bible ref) | 27 | **0.021** | **0.62–0.68** † | 0.82 | 0.825 | 0.55 ‡ |
| instruct (voice-design) | 16 | 0.020 | **0.85** | – | – | – |

† two stochastic resynthesis passes → 0.619 and 0.675 (persisted: 0.675). Report mean±std with more passes.
‡ len_ratio uses the Qwen3-TTS duration model — **not meaningful for OmniVoice** (rule-based duration).

**Anchors:** zero-shot (nb01) tone_i2 0.598 · native ref ~0.58 · OmniVoice-paper Yoruba CER 21.37.
→ finetune: CER excellent (0.021 ≪ 21.37), tone_i2 0.598 → ~0.62–0.68 (modest, partly within resynthesis noise),
**voice-design (instruct) survived** (0.85).

## Loss curve (full eval series; train series partial — Colab truncated steps 0–2600)
| step | eval_loss | train_loss |
|---|---|---|
| 3000 | 3.44 | 3.38 |
| 3500 | **3.08** | 3.36 |
| 4000 | 3.43 | 3.33 |
| 4500 | **2.93** (lowest) | 3.36 |
| 5000 | 3.17 (final/used) | 3.30 |

**train/loss is FLAT (~3.30–3.42) from step 2600 (epoch 9) → 5000 (epoch 19)** → model **saturated by ~epoch 9**; 5000 steps / 19 epochs is overkill (informs the sweep: use far fewer steps). Eval loss is non-monotonic/noisy (masked-diffusion random-mask eval) — **lowest at checkpoint-4500 (2.93)**, not the final 5000 (3.17) → select checkpoints by tone_i2 + ear, not eval loss.

⚠️ **Only `checkpoint-5000` is on S3** (§8 uploads FINAL_CKPT only). `checkpoint-500…4500` are **local-only → lost on shutdown** unless uploaded.

## Open items / caveats
- **Cloning-collapse check (§12B) did NOT run** — `holdouts['naijavoices_holdout_speakers']` / `manifest.v3.jsonl`
  unresolved. 19 epochs on ~single-speaker bible = real collapse risk → re-run on ckpt-5000 (inference only).
- **tone_i2 at/above the native ~0.58 reference** → validate with the native ear (metric may reward over-articulated tone).
- **Sweep design:** at fixed 5000 steps, 1h/5h/15h points would see 100s of epochs (overfit). Decide
  `STEPS_OVERRIDE` / fix-epochs before running the data-efficiency curve.

Machine-readable: `full_28h_run.json` (config + loss series + eval points).
