# Runtime artifacts (not committed)

These are model weights / calibration produced during development. They are needed to run the
metric and notebooks but are kept out of git. Drop them into this `assets/` folder (or point the
notebooks at them). They currently live on the project's private bucket — upload them to a public
home (e.g. a Hugging Face dataset/model repo) when you release.

| File | What it is | Used by | Source (private) |
|---|---|---|---|
| `f0_abs_calibration.v1.json` | I2 speaker-normalised H/M/L decision boundaries | `tone_f0_abs.py` | `s3://codec-audio-data/tts_data/yoruba/tone_v2/f0_abs_calibration.v1.json` |
| `tone_probe_L*.pt` | trained AfriHuBERT tone-probe weights (I1) | `tone_probe.py` | `s3://codec-audio-data/tts_data/yoruba/tone_v2/tone_probe_L*` |
| `holdouts.v1.json` | 200 held-out Yorùbá eval texts | notebooks | `s3://codec-audio-data/tts_data/yoruba_gold/holdouts.v1.json` |

`minimal_pairs_draft.json` (the tone minimal pairs) **is** committed, under `tone_metric/`.
Note it is marked `verified:false` — have a native speaker confirm the pairs/glosses before you
lean on it publicly.

## Optional: an evaluation dataset

If you later want a citable benchmark artifact (useful for the resource paper, not required for
the poster), package the held-out eval texts + minimal pairs + the synthesized evaluation wavs
as a Hugging Face **dataset** repo. That is separate from the metric itself — the metric is code
and needs no dataset to run.
