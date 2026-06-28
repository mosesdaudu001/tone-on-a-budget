#!/usr/bin/env python3
# coding=utf-8
# build_pilot_form.py — assemble a blind, randomized Yorùbá TONE listening-test PILOT.
#
# WHAT IT BUILDS (one self-contained, offline, mobile-first HTML form a native reviewer taps through on a phone):
#   ~12 REAL NATIVE clips (BibleTTS/SLR86, the known-correct anchor + native ceiling)
#   ~9  MODEL clips @ 0h  (zero-shot OmniVoice base — likely weaker tone)
#   ~9  MODEL clips @ 5h  (the poster's "consolidated" finetune point)
#   ~6  CATCH items (a native anchor duplicated, OR a pitch-flattened "obviously wrong" clip) to screen raters.
#   = ~36 items total, shuffled with a fixed seed so the WITHHELD keymap matches the on-screen order.
#
# GROUNDED IN THE REAL EVAL (do not re-derive these — copied from nb14 §7/§11 and nb07 §6):
#   * native manifest  : s3://codec-audio-data/tts_data/yoruba_gold/s1_train.v2.jsonl
#                        rows -> {id, text, source(bible|slr86), speaker_id, duration_sec, audio_s3_key}
#   * model clips      : SYNTHESIZED on the fly with OmniVoice (base snapshot = 0h; ckpt .../omnivoice_yoruba/5h),
#                        exactly as nb07 §6 (probe_lines = minimal-pair carriers + holdout long sentences,
#                        ref_audio/ref_text = a bible row), seed 4242. NB: a full per-clip model probe set DOES
#                        exist on S3 (nb05/nb13 upload every probe wav under
#                        tts_data/yoruba/omnivoice_ft_probe/{tag}/wav/, incl. the 5h tag), so we are NOT
#                        synthesizing for lack of files. We re-synthesize because those wavs are named
#                        ev_{tag}_{kind}_{abs(hash(text))%99999}.wav with NO sidecar text manifest, and Python
#                        hash() is process-randomized (PYTHONHASHSEED), so the intended text per clip is
#                        unrecoverable — unusable for a pilot that must SHOW the written word. Re-synthesis here
#                        keeps text↔audio paired by construction.
#   * tone_i2          : f0_abs_score(..., asr=MMS-yor, mid_ref=None, theta_h/theta_l from
#                        f0_abs_calibration.v1.json) -> balanced H/M/L recall via _bal_tone_acc. The instrument,
#                        calibration, and aggregation function are IDENTICAL to nb07 one_pass / nb14 §6, but here
#                        they are applied PER CLIP (clip granularity is required for the per-item kappa). The
#                        poster's headline 0.567/0.598/0.633 is _bal_tone_acc over (pred,target) pairs POOLED
#                        across the whole probe set in a pass, then averaged over 5 seeds. A per-clip tone_i2 is a
#                        different, noisier estimator (short carriers have ~6-12 TBUs, so per-class recall is
#                        heavily quantized): it is on the SAME scale/units (chance ≈ 0.33) but is NOT individually
#                        equal to the headline and must NOT be averaged to reproduce it. The pilot's validity rests
#                        on the native-anchor-median threshold (same per-clip computation), not on per-clip values
#                        matching the headline. Stored in the keymap, NEVER shown to the rater.
#
# OUTPUTS (into --out-dir, default this directory):
#   pilot_form.html  — audio embedded as base64 data-URIs (mp3 @ ~32 kbps mono). No external/CDN dependency.
#   keymap.json      — WITHHELD answer key: item_id -> {clip_id, source, intended_text, tone_i2, is_catch,...}.
#
# RUN (in the audio env / Colab with AWS creds + tone-metric + omnivoice present):
#   python build_pilot_form.py                       # full build (downloads + synthesizes + scores)
#   python build_pilot_form.py --dry-run             # list the chosen clips, download/synthesize NOTHING
#   python build_pilot_form.py --n-native 12 --n-model0h 9 --n-model5h 9 --n-catch 6 --seed 4242
#
# Only the native + scoring path needs S3 creds; the model path additionally needs a GPU + omnivoice. If
# omnivoice or a checkpoint is missing the script DEGRADES: it warns, drops the model items, and still ships a
# native-anchor + catch form (so the pilot can run tonight). Fail-loud only for a missing bucket/manifest/calib.

import argparse
import base64
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------- constants (match nb07 / nb14)
BUCKET = "codec-audio-data"
SR = 24000
LANG_CODE = "yo"
BIBLE_MANIFEST = "tts_data/yoruba_gold/s1_train.v2.jsonl"
HOLDOUTS_KEY = "tts_data/yoruba_gold/holdouts.v1.json"
S3_TONE_PREFIX = "tts_data/yoruba/tone_v2"
F0CAL_KEY = f"{S3_TONE_PREFIX}/f0_abs_calibration.v1.json"
CKPT_ROOT = "tts_checkpoints/omnivoice_yoruba"
NATIVE_DUR = (2.0, 6.0)          # seconds — pilot wants short clips (task spec); model carriers are short too
SYNTH_SEED = 4242                # nb07 VARIANCE_SEEDS[0] — deterministic model synthesis
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- secrets (nb07/nb14 _secret)
def _secret(k):
    try:
        from google.colab import userdata
        v = userdata.get(k)
        if v:
            return v
    except Exception:
        pass
    v = os.environ.get(k)
    if v:
        return v
    if not sys.stdin.isatty():
        sys.exit(f"FATAL: {k} not set. In a non-interactive run, set {k} in the environment "
                 f"(e.g. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) or a Colab secret.")
    import getpass
    return getpass.getpass(f"{k}: ")


def connect_s3():
    import boto3
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ["AWS_ACCESS_KEY_ID"] = _secret("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = _secret("AWS_SECRET_ACCESS_KEY")
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception as e:
        sys.exit(f"FATAL: cannot reach s3://{BUCKET} ({e}). Set AWS creds in the env / Colab secrets.")
    return s3


# ----------------------------------------------------------------------------- tone_i2 aggregation (VERBATIM)
def _bal_tone_acc(pairs):
    """Balanced per-class (H/M/L) recall over (pred,target) TBU pairs — the SAME aggregation function nb07
    one_pass / nb14 §6 use. Kept byte-identical so the units match (chance ≈ 0.33). NOTE: the poster's headline
    pools pairs across the whole probe set before calling this; here it is called PER CLIP, which is a noisier
    estimator that does not average back to the pooled headline (see module header)."""
    if not pairs:
        return float("nan")
    import numpy as np
    tot, cor = defaultdict(int), defaultdict(int)
    for pp, tt in pairs:
        tot[tt] += 1
        cor[tt] += int(pp == tt)
    recs = [cor[c] / tot[c] for c in tot if tot[c] > 0]
    return float(np.mean(recs)) if recs else float("nan")


# ----------------------------------------------------------------------------- scoring stack (lazy, heavy)
class Scorer:
    """Holds MMS-yor (CER + aligner) and the frozen F0 calibration; produces a per-clip tone_i2 the SAME way
    nb07/nb14 do. Built lazily so --dry-run needs no torch/GPU."""

    def __init__(self, s3, device=None):
        import torch
        from tone_metric import tone_f0_abs as f0a
        from tone_metric.grpo_reward import RewardModels
        self.f0a = f0a
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rm = RewardModels(device=self.device)
        cal = os.path.join(tempfile.gettempdir(), "f0cal.json")
        try:
            s3.download_file(BUCKET, F0CAL_KEY, cal)
        except Exception as e:
            sys.exit(f"FATAL: cannot read s3://{BUCKET}/{F0CAL_KEY} ({e}). "
                     f"The F0 calibration is required for tone_i2 scoring.")
        c = json.load(open(cal))
        self.th, self.tl = c["theta_h"], c["theta_l"]
        self.mode, self.late = c.get("mode", "blind"), c.get("late_frac", 0.5)
        print(f"[scorer] device={self.device} theta_h={self.th} theta_l={self.tl} "
              f"mode={self.mode} late_frac={self.late}", flush=True)

    def tone_i2(self, wav, text):
        """Per-clip tone_i2 (balanced H/M/L recall over this clip's pairs). One shared MMS forward, mid_ref=None
        — the instrument/calibration are identical to the nb07/nb14 model-eval call, so values are on the SAME
        scale (chance ≈ 0.33). It is computed at CLIP granularity (needed for per-item kappa) and is NOT
        individually equal to, nor averageable into, the pass-pooled headline 0.567 / 0.598 / 0.633."""
        logits, n16 = self.rm.asr_logits(wav, SR)
        i2 = self.f0a.f0_abs_score(wav, SR, text, asr=self.rm.asr, proc=self.rm.asr_proc, device=self.device,
                                   emissions=logits, n16=n16, theta_h=self.th, theta_l=self.tl,
                                   mode=self.mode, mid_ref=None, late_frac=self.late)
        pairs = [(pp, tt) for pp, tt in zip(i2["pred"], i2["target"]) if pp is not None]
        return _bal_tone_acc(pairs), i2["coverage"], len(pairs)


# ----------------------------------------------------------------------------- audio io helpers
def _read_wav(path):
    import soundfile as sf
    import soxr
    w, sr = sf.read(path, dtype="float32")
    w = w.mean(-1) if getattr(w, "ndim", 1) > 1 else w
    if sr != SR:
        w = soxr.resample(w, sr, SR).astype("float32")
    return w


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def encode_data_uri(wav, workdir, name, bitrate="32k", max_seconds=6.5):
    """Compress a mono float32 array to a small base64 audio data-URI. mp3 via ffmpeg when present (tiny);
    otherwise fall back to an uncompressed WAV data-URI (bigger but always works)."""
    import numpy as np
    import soundfile as sf
    w = np.asarray(wav, dtype="float32").reshape(-1)
    if max_seconds and len(w) > int(max_seconds * SR):
        w = w[: int(max_seconds * SR)]
    wav_path = os.path.join(workdir, name + ".wav")
    sf.write(wav_path, w, SR, subtype="PCM_16")
    if _have_ffmpeg():
        mp3_path = os.path.join(workdir, name + ".mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                        "-ac", "1", "-b:a", bitrate, mp3_path], check=True)
        data = open(mp3_path, "rb").read()
        return "data:audio/mpeg;base64," + base64.b64encode(data).decode(), len(data)
    # ffmpeg absent: uncompressed PCM_16 WAV. ~10× larger than mp3@32k — across ~36 clips the embedded HTML can
    # reach tens of MB (heavy for WhatsApp). Warn loudly; install ffmpeg (nb15 apt-installs it) to keep it small.
    data = open(wav_path, "rb").read()
    print(f"[warn] ffmpeg not found — embedding {name} as uncompressed WAV ({len(data)/1e6:.2f} MB). "
          f"Install ffmpeg to emit ~10× smaller mp3 clips.", flush=True)
    return "data:audio/wav;base64," + base64.b64encode(data).decode(), len(data)


# ----------------------------------------------------------------------------- native clip selection (nb14 §11)
def select_native(s3, n, per_spk_cap, rng):
    """Stream the BibleTTS/SLR86 manifest, keep short clips spread across speakers. Returns row dicts with the
    EXACT schema fields nb14 uses (audio_s3_key/text/source/speaker_id/duration_sec)."""
    rows, per_spk = [], Counter()
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=BIBLE_MANIFEST)
    except Exception as e:
        sys.exit(f"FATAL: cannot read s3://{BUCKET}/{BIBLE_MANIFEST} ({e}). "
                 f"The native manifest is required to select anchor clips.")
    body = io.TextIOWrapper(obj["Body"], encoding="utf-8")
    candidates = []
    for raw in body:
        r = json.loads(raw)
        d = float(r.get("duration_sec", 0.0))
        if not (NATIVE_DUR[0] <= d <= NATIVE_DUR[1]):
            continue
        if not (r.get("text") or "").strip():
            continue
        if not r.get("audio_s3_key"):
            continue
        candidates.append(r)
    rng.shuffle(candidates)            # spread across the manifest, not just the head
    for r in candidates:
        spk = str(r.get("speaker_id", "?"))
        if per_spk[spk] >= per_spk_cap:
            continue
        per_spk[spk] += 1
        rows.append(dict(clip_id=str(r["id"]), text=r["text"].strip(), source="native",
                         speaker=spk, audio_s3_key=r["audio_s3_key"], dur=float(r.get("duration_sec", 0.0))))
        if len(rows) >= n:
            break
    return rows


# ----------------------------------------------------------------------------- model probe lines (nb07 §5)
def build_probe_lines(s3, code_dir, rng):
    """Reproduce nb07 §5: minimal-pair carrier sentences + holdout long sentences. Each has a KNOWN intended
    text, which is what the reviewer is shown."""
    mp = json.load(open(os.path.join(code_dir, "minimal_pairs_draft.json")))
    carrier = mp["carriers"][0]["template"]
    lines = []
    for s_ in mp["sets"]:
        for j, it in enumerate(s_["items"]):
            lines.append(dict(kind="minpair", text=carrier.replace("___", it["text"])))
    try:
        hold = json.loads(s3.get_object(Bucket=BUCKET, Key=HOLDOUTS_KEY)["Body"].read())
        for e in hold.get("eval_texts", []):
            lines.append(dict(kind="long", text=e["text"]))
    except Exception as e:
        print(f"[warn] holdouts unavailable ({e}); using minimal-pair carriers only for model clips.", flush=True)
    rng.shuffle(lines)
    # prefer natural 'long' sentences first, then fill with minimal-pair carriers
    lines.sort(key=lambda x: 0 if x["kind"] == "long" else 1)
    return lines


def get_bible_ref(s3, workdir):
    """A clean bible row used as the voice/text reference for OmniVoice synthesis (nb07 §5 / nb12)."""
    body = io.TextIOWrapper(s3.get_object(Bucket=BUCKET, Key=BIBLE_MANIFEST)["Body"], encoding="utf-8")
    for raw in body:
        r = json.loads(raw)
        if r.get("source") == "bible" and 3.0 <= float(r.get("duration_sec", 0)) <= 10.0:
            p = os.path.join(workdir, "ref.wav")
            s3.download_file(BUCKET, r["audio_s3_key"], p)
            return p, r["text"]
    raise RuntimeError("no bible ref row in manifest")


# ----------------------------------------------------------------------------- OmniVoice synthesis (nb07 §6)
def _load_omnivoice(local_dir, device):
    import torch
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained(local_dir, device_map=("cuda:0" if device == "cuda" else "cpu"),
                                     dtype=torch.float16)


def _materialize_ckpt(s3, prefix, dest, ov_base):
    """Pull a training checkpoint's top-level files + copy the aux tokenizer/codec from the base snapshot
    (nb07 materialize_ckpt + pull_top_level)."""
    aux = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
           "merges.txt", "chat_template.jinja", "generation_config.json", "added_tokens.json"]
    os.makedirs(dest, exist_ok=True)
    tok, n = None, 0
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix + "/")
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            tail = o["Key"][len(prefix) + 1:]
            if not tail or "/" in tail:
                continue
            s3.download_file(BUCKET, o["Key"], os.path.join(dest, os.path.basename(o["Key"])))
            n += 1
        if r.get("IsTruncated"):
            tok = r.get("NextContinuationToken")
        else:
            break
    if n == 0:
        raise RuntimeError(f"nothing under s3://{BUCKET}/{prefix}/ — was the checkpoint uploaded?")
    have = set(os.listdir(dest))
    for fn in aux:
        if fn not in have and os.path.exists(os.path.join(ov_base, fn)):
            shutil.copy(os.path.join(ov_base, fn), os.path.join(dest, fn))
    at = os.path.join(dest, "audio_tokenizer")
    if not os.path.isdir(at):
        shutil.copytree(os.path.join(ov_base, "audio_tokenizer"), at)
    return dest


def synthesize_model_clips(s3, n0, n5, workdir, code_dir, rng):
    """Synthesize n0 clips from the zero-shot base and n5 from the 5h checkpoint, exactly as nb07 §6. Returns
    (clips, base_ok, ckpt5_ok). Any import/GPU/checkpoint failure DEGRADES to fewer/zero model clips (warns)."""
    import numpy as np
    import torch
    if n0 <= 0 and n5 <= 0:
        return [], False, False
    try:
        from huggingface_hub import snapshot_download
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        ov_base = None
        for _ in range(3):
            try:
                ov_base = snapshot_download("k2-fsa/OmniVoice", max_workers=1, etag_timeout=60)
                break
            except Exception as e:
                print("  base prefetch retry:", type(e).__name__, e, flush=True)
        assert ov_base and os.path.isdir(os.path.join(ov_base, "audio_tokenizer")), "OmniVoice base snapshot missing"
    except Exception as e:
        print(f"[warn] OmniVoice unavailable ({e}); SKIPPING all model clips — native+catch form only.", flush=True)
        return [], False, False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_path, ref_text = get_bible_ref(s3, workdir)
    probe = build_probe_lines(s3, code_dir, rng)

    def _gen(model, lines, n, cond):
        torch.manual_seed(SYNTH_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SYNTH_SEED)
        np.random.seed(SYNTH_SEED)
        out = []
        for p in lines[:n]:
            a = model.generate(text=p["text"], language=LANG_CODE, ref_audio=ref_path, ref_text=ref_text)
            w = a[0] if isinstance(a, (list, tuple)) else a
            out.append(dict(clip_id=f"{cond}_{len(out):02d}", text=p["text"], source=cond,
                            wav=np.asarray(w, dtype="float32").reshape(-1)))
        return out

    clips, base_ok, ckpt5_ok = [], False, False
    if n0 > 0:
        try:
            m = _load_omnivoice(ov_base, device)
            clips += _gen(m, probe, n0, "model0h")
            base_ok = True
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[warn] 0h synthesis failed ({e}); dropping model0h clips.", flush=True)
    if n5 > 0:
        try:
            dest = _materialize_ckpt(s3, f"{CKPT_ROOT}/5h", os.path.join(workdir, "ckpt5h"), ov_base)
            m = _load_omnivoice(dest, device)
            # different probe lines from the 0h set so the rater never hears the same sentence twice across models
            clips += _gen(m, probe[n0:] or probe, n5, "model5h")
            ckpt5_ok = True
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[warn] 5h synthesis failed ({e}); dropping model5h clips.", flush=True)
    return clips, base_ok, ckpt5_ok


# ----------------------------------------------------------------------------- HTML (embedded VERBATIM)
def render_html(items_for_js, seed):
    """items_for_js: ordered list of {item_id, text, audio} (audio = data URI). Intro/end text + button labels
    are embedded VERBATIM per the spec. No external/CDN dependency — inline CSS/JS, inline audio."""
    items_json = json.dumps(items_for_js, ensure_ascii=False)
    html = r"""<!DOCTYPE html>
<html lang="yo">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Yorùbá Tone Listening Pilot</title>
<style>
  :root { --pad: 16px; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin: 0; font-family: -apple-system, system-ui, "Segoe UI", Roboto, Arial, sans-serif;
         background: #f4f6f8; color: #18202a; line-height: 1.45; }
  .wrap { max-width: 640px; margin: 0 auto; padding: var(--pad); min-height: 100vh; }
  .card { background: #fff; border-radius: 16px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
  h1 { font-size: 22px; margin: 0 0 12px; }
  p.lead { font-size: 17px; }
  .progress { font-size: 15px; color: #5a6b7b; margin: 4px 0 14px; font-weight: 600; }
  .stim { font-size: 26px; font-weight: 700; text-align: center; padding: 18px 8px; margin: 6px 0 14px;
          background: #eef3f8; border-radius: 12px; word-break: break-word; }
  audio { width: 100%; margin: 6px 0 18px; }
  button { font-size: 18px; font-family: inherit; border: none; border-radius: 14px; padding: 18px;
           width: 100%; margin: 9px 0; cursor: pointer; font-weight: 600; }
  .choice { background: #e9eef3; color: #18202a; border: 2px solid transparent; }
  .choice.sel { border-color: #1769e0; background: #dcebff; }
  .nav { display: flex; gap: 10px; margin-top: 18px; }
  .nav button { flex: 1; }
  .prim { background: #1769e0; color: #fff; }
  .prim:disabled { background: #aebfce; cursor: not-allowed; }
  .ghost { background: #e3e8ee; color: #2a3744; }
  .code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; background: #0e1726;
          color: #d7e2ef; padding: 14px; border-radius: 10px; word-break: break-all; white-space: pre-wrap; }
  a.dl { display: inline-block; margin-top: 12px; color: #1769e0; font-weight: 600; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">

  <div id="intro" class="card">
    <h1>Yorùbá Tone Check 🎧</h1>
    <p class="lead">You'll hear short Yorùbá clips. For each one, the written word or sentence is shown. Tap whether the TONE (ohùn) of the voice matches the written word. About 20 minutes. Headphones help but a phone is fine. There are no wrong answers.</p>
    <button class="prim" onclick="start()">Start</button>
  </div>

  <div id="task" class="card hidden">
    <div class="progress" id="prog"></div>
    <div class="stim" id="stim"></div>
    <audio id="player" controls preload="none"></audio>
    <button class="choice" id="c_ok"  onclick="choose('ok')">✓ Tone is correct</button>
    <button class="choice" id="c_bad" onclick="choose('bad')">✗ Tone is wrong</button>
    <button class="choice" id="c_un"  onclick="choose('unsure')">— Not sure</button>
    <div class="nav">
      <button class="ghost" id="back" onclick="prev()">‹ Back</button>
      <button class="prim"  id="next" onclick="next_()" disabled>Next ›</button>
    </div>
  </div>

  <div id="done" class="card hidden">
    <h1>Done 🙏</h1>
    <p class="lead">Thank you! Tap "Copy my answers" below, then paste them back to Moses in your chat.</p>
    <button class="prim" onclick="copyAns()">Copy my answers</button>
    <div class="code" id="ansbox"></div>
    <a class="dl" id="csv" href="#" download="pilot_answers.csv">Download CSV instead</a>
  </div>

</div>
<script>
const ITEMS = __ITEMS_JSON__;
const SEED  = __SEED__;
const LSKEY = "yor_tone_pilot_" + SEED;
let answers = {};
let idx = 0;

try { answers = JSON.parse(localStorage.getItem(LSKEY)) || {}; } catch(e) { answers = {}; }
function save(){ try { localStorage.setItem(LSKEY, JSON.stringify(answers)); } catch(e){} }

function start(){
  document.getElementById("intro").classList.add("hidden");
  document.getElementById("task").classList.remove("hidden");
  // resume at first unanswered item
  idx = 0;
  for (let i=0;i<ITEMS.length;i++){ if(!answers[ITEMS[i].item_id]){ idx=i; break; } if(i===ITEMS.length-1) idx=i; }
  render();
}
function render(){
  const it = ITEMS[idx];
  document.getElementById("prog").textContent = (idx+1) + " / " + ITEMS.length;
  document.getElementById("stim").textContent = it.text;
  const pl = document.getElementById("player");
  pl.src = it.audio; pl.load();
  const cur = answers[it.item_id] || null;
  for (const [id,val] of [["c_ok","ok"],["c_bad","bad"],["c_un","unsure"]]){
    document.getElementById(id).classList.toggle("sel", cur===val);
  }
  document.getElementById("next").disabled = !cur;
  document.getElementById("back").disabled = (idx===0);
  document.getElementById("next").textContent = (idx===ITEMS.length-1) ? "Finish" : "Next ›";
}
function choose(val){
  answers[ITEMS[idx].item_id] = val; save();
  for (const [id,v] of [["c_ok","ok"],["c_bad","bad"],["c_un","unsure"]]){
    document.getElementById(id).classList.toggle("sel", v===val);
  }
  document.getElementById("next").disabled = false;
}
function next_(){
  if(!answers[ITEMS[idx].item_id]) return;
  if(idx===ITEMS.length-1){ finish(); return; }
  idx++; render();
}
function prev(){ if(idx>0){ idx--; render(); } }
function compact(){
  let parts = ["PILOT1"];
  for (const it of ITEMS){ if(answers[it.item_id]) parts.push(it.item_id+"="+answers[it.item_id]); }
  return parts.join(";");
}
function csvBlob(){
  let rows = ["item_id,answer"];
  for (const it of ITEMS){ rows.push(it.item_id+","+(answers[it.item_id]||"")); }
  return "data:text/csv;charset=utf-8," + encodeURIComponent(rows.join("\n"));
}
function finish(){
  document.getElementById("task").classList.add("hidden");
  document.getElementById("done").classList.remove("hidden");
  document.getElementById("ansbox").textContent = compact();
  document.getElementById("csv").href = csvBlob();
}
function copyAns(){
  const code = compact();
  // capture the button synchronously — window.event is gone by the time the async clipboard promise resolves
  const btn = document.querySelector("#done button.prim");
  const done = () => { if(btn){ btn.textContent="Copied ✓"; setTimeout(()=>btn.textContent="Copy my answers",1600); } };
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(code).then(done).catch(()=>fallbackCopy(code,done));
  } else { fallbackCopy(code, done); }
}
function fallbackCopy(code, done){
  const t=document.createElement("textarea"); t.value=code;
  t.contentEditable="true"; t.readOnly=false;
  t.style.position="fixed"; t.style.top="0"; t.style.left="0"; t.style.opacity="0";
  document.body.appendChild(t);
  let ok=false;
  try {
    const range=document.createRange(); range.selectNodeContents(t);
    const sel=window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    t.setSelectionRange(0, code.length);
    ok = document.execCommand("copy");
  } catch(e){ ok=false; }
  document.body.removeChild(t);
  if(ok){ done(); } else { alert("Copy failed — long-press the code box to copy."); }
}
</script>
</body>
</html>
"""
    return html.replace("__ITEMS_JSON__", items_json).replace("__SEED__", str(seed))


# ----------------------------------------------------------------------------- assemble
def main():
    ap = argparse.ArgumentParser(description="Build the Yorùbá tone listening-test pilot kit.")
    ap.add_argument("--n-native", type=int, default=12)
    ap.add_argument("--n-model0h", type=int, default=9)
    ap.add_argument("--n-model5h", type=int, default=9)
    ap.add_argument("--n-catch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--per-spk-cap", type=int, default=3, help="max native clips per studio speaker")
    ap.add_argument("--catch-mode", choices=["dup", "degrade"], default="dup",
                    help="dup=duplicate a native anchor (expected ✓); degrade=PSOLA pitch-flatten (expected ✗)")
    ap.add_argument("--bitrate", default="32k", help="mp3 bitrate for embedded clips (ffmpeg)")
    ap.add_argument("--dry-run", action="store_true", help="list chosen clips; download/synthesize NOTHING")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    s3 = connect_s3()

    # tone_metric ships minimal_pairs_draft.json as package data; fall back to the repo copy.
    try:
        import tone_metric
        code_dir = os.path.dirname(tone_metric.__file__)
    except Exception:
        code_dir = os.path.join(os.path.dirname(HERE), "tone_metric")

    # -------- selection (no heavy work yet) --------
    native_rows = select_native(s3, args.n_native, args.per_spk_cap, rng)
    print(f"[select] native clips chosen: {len(native_rows)} "
          f"(speakers: {len({r['speaker'] for r in native_rows})})", flush=True)

    if args.dry_run:
        print("\n=== DRY RUN — chosen NATIVE clips ===")
        for r in native_rows:
            print(f"  {r['clip_id']:>10} spk {r['speaker']:>6} {r['dur']:.1f}s | {r['text'][:60]}")
        print(f"\n=== would synthesize MODEL clips: {args.n_model0h}×0h + {args.n_model5h}×5h "
              f"(OmniVoice base + {CKPT_ROOT}/5h, seed {SYNTH_SEED}) ===")
        try:
            probe = build_probe_lines(s3, code_dir, rng)
            for p in probe[: args.n_model0h + args.n_model5h]:
                print(f"  [{p['kind']:>7}] {p['text'][:60]}")
        except Exception as e:
            print("  (could not preview probe lines:", e, ")")
        print(f"\n=== would add {args.n_catch} CATCH items (mode={args.catch_mode}) ===")
        print("\nDRY RUN complete — no audio downloaded, no model loaded, no files written.")
        return

    work = tempfile.mkdtemp(prefix="pilot_")
    scorer = Scorer(s3)

    # -------- gather clips with audio arrays --------
    clips = []   # each: dict(clip_id, text, source, wav)
    for r in native_rows:
        lp = os.path.join(work, f"nat_{r['clip_id']}.wav")
        s3.download_file(BUCKET, r["audio_s3_key"], lp)
        clips.append(dict(clip_id=r["clip_id"], text=r["text"], source="native", wav=_read_wav(lp)))

    model_clips, base_ok, ckpt5_ok = synthesize_model_clips(
        s3, args.n_model0h, args.n_model5h, work, code_dir, rng)
    clips += model_clips
    if not base_ok and args.n_model0h:
        print("[warn] no model0h clips in the final set.", flush=True)
    if not ckpt5_ok and args.n_model5h:
        print("[warn] no model5h clips in the final set.", flush=True)

    # -------- catch items --------
    natives = [c for c in clips if c["source"] == "native"]
    catch_items = []
    for i in range(args.n_catch):
        if not natives:
            break
        base = natives[i % len(natives)]
        if args.catch_mode == "degrade":
            from tone_metric import psola_flatten
            import numpy as np
            w = np.asarray(psola_flatten(base["wav"], SR), dtype="float32").reshape(-1)
            catch_items.append(dict(clip_id=f"catch_deg_{i:02d}", text=base["text"], source="catch",
                                    wav=w, catch_expected="bad"))   # pitch-flattened -> tone is wrong
        else:
            catch_items.append(dict(clip_id=f"catch_dup_{base['clip_id']}", text=base["text"], source="catch",
                                    wav=base["wav"], catch_expected="ok"))  # real native dup -> tone is correct
    clips += catch_items

    # -------- score tone_i2 per clip --------
    print(f"[score] computing tone_i2 for {len(clips)} clips ...", flush=True)
    for c in clips:
        try:
            ti2, cov, npairs = scorer.tone_i2(c["wav"], c["text"])
        except Exception as e:
            print("  tone_i2 fail", c["clip_id"], "->", e, flush=True)
            ti2, cov, npairs = float("nan"), 0.0, 0
        c["tone_i2"], c["coverage"], c["n_pairs"] = ti2, cov, npairs

    # -------- shuffle + assign item ids (fixed seed; keymap matches) --------
    order = list(range(len(clips)))
    random.Random(args.seed).shuffle(order)
    shuffled = [clips[i] for i in order]

    items_for_js, keymap = [], {}
    total_bytes = 0
    for n, c in enumerate(shuffled, 1):
        item_id = f"item{n:02d}"
        uri, nbytes = encode_data_uri(c["wav"], work, item_id, bitrate=args.bitrate)
        total_bytes += nbytes
        items_for_js.append(dict(item_id=item_id, text=c["text"], audio=uri))
        keymap[item_id] = dict(clip_id=c["clip_id"], source=c["source"], intended_text=c["text"],
                               tone_i2=(None if c["tone_i2"] != c["tone_i2"] else round(float(c["tone_i2"]), 4)),
                               coverage=round(float(c["coverage"]), 3), n_pairs=int(c["n_pairs"]),
                               is_catch=(c["source"] == "catch"),
                               catch_expected=c.get("catch_expected"))

    os.makedirs(args.out_dir, exist_ok=True)
    html_path = os.path.join(args.out_dir, "pilot_form.html")
    key_path = os.path.join(args.out_dir, "keymap.json")
    open(html_path, "w", encoding="utf-8").write(render_html(items_for_js, args.seed))
    json.dump(keymap, open(key_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # -------- report --------
    src_counts = Counter(k["source"] for k in keymap.values())
    html_mb = os.path.getsize(html_path) / 1e6
    ear = [it["item_id"] for it in items_for_js[:4]]
    print("\n" + "=" * 64)
    print(f"  wrote {html_path}  ({html_mb:.2f} MB, audio ~{total_bytes/1e6:.2f} MB @ {args.bitrate})")
    print(f"  wrote {key_path}  ({len(keymap)} items: {dict(src_counts)})")
    print(f"  EAR-CHECK these 4 item ids in nb15 / the html: {', '.join(ear)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
