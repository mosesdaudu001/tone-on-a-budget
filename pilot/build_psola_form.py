#!/usr/bin/env python3
# coding=utf-8
# build_psola_form.py — assemble a PSOLA-ISOLATED, blind, randomized Yorùbá TONE listening test.
#
# WHY THIS KIT EXISTS (the rigorous successor to build_pilot_form.py)
#   The pilot (build_pilot_form.py) asked a native to judge ✓/✗ on REAL *TTS* clips. That CONFOUNDS tone with
#   synthetic naturalness: a "✗ tone is wrong" tap could mean "the model's pitch is off" OR "the voice sounds
#   robotic / mispronounced". There is no ground truth (we don't know the model's true per-syllable tone), so a
#   low AUROC can't be blamed on the metric vs. the stimulus. THIS kit removes the confound by construction:
#
#     Take a REAL NATIVE clip with KNOWN tones. Make TWO twins that are BIT-FOR-BIT identical except the F0 of
#     ONE syllable:
#        correct  = psola_roundtrip(wav)                      -> tone UNCHANGED, full PSOLA resynthesis artifact
#        flipped  = psola_shift_window(wav, t0, t1, ±K·ΔHL)   -> ONE TBU's tone flipped (L->H or H->L), SAME artifact
#     Both twins carry the identical resynthesis artifact, so the ONLY perceptible difference is the tone of that
#     one syllable. Ground truth is therefore known: correct -> ✓, flipped -> ✗. If a native can hear the flip
#     (flipped-acc ≫ chance) the stimulus is valid; whether the deployed tone_i2 metric AGREES with that native
#     read is then a clean, confound-free test.
#
# GROUNDED IN EXISTING CODE (imported, not re-derived — see the report):
#   * native loader + S3 plumbing + HTML + per-clip tone_i2 : pilot/build_pilot_form.py  (import as `bp`)
#       bp.connect_s3 / bp._secret / bp.select_native / bp._read_wav / bp.encode_data_uri / bp.render_html /
#       bp.Scorer / bp._bal_tone_acc / bp.BUCKET / bp.SR / bp.BIBLE_MANIFEST   (none pull torch/omnivoice at
#       import time — the heavy paths are lazy inside Scorer / synthesize_model_clips, which we never touch).
#   * PSOLA primitives                                       : tone_metric/tone_oracle.py
#       oracle.psola_shift_window (flip ONE window's F0) / oracle.psola_roundtrip (artifact-only twin) /
#       oracle.measure_delta_HL (the flip UNIT: median(H)-median(L) of declination-removed residuals; k=1.0 =
#       a full H<->L tonal distance — so K·ΔHL really moves a syllable past the opposite tone band).
#       We use the PRIMITIVES, never run_oracle_clip (that sweeps many k for a detection curve; we want ONE
#       decisive stimulus per clip).
#   * alignment windows + tones                              : tone_metric/tone_eval_v2.py  (v2.precompute)
#       pre["tones"] = orthographic H/M/L per TBU; pre["wins"][i] = (t0,t1) sec window of TBU i (None = unaligned).
#   * per-TBU prediction of the deployed meter               : tone_metric/tone_f0_abs.py  (f0a)
#       f0a.score_abs_from_precomputed(pre, theta_*, mode, mid_ref=None) -> pred[] aligned with pre["tones"].
#
# mid_ref CHOICE (documented):
#   Every clean/twin tone_i2 here is scored with mid_ref=None — the per-utterance median residual register
#   anchor, EXACTLY as the deployed metric (nb07/nb14, bp.Scorer.tone_i2) and the poster headline. We are asking
#   "does the DEPLOYED metric agree with a human on a known tone flip", so we must score the way it is deployed.
#   tone_oracle.py's note about FREEZING I2's mid_ref from the clean clip is for CAUSAL ATTRIBUTION (isolating
#   "did the meter respond to THIS flip vs a register shift it induced") — a different question than "does the
#   shipped metric track human perception". We deliberately do NOT freeze mid_ref.
#
# OUTPUTS (into --out-dir, default this dir):
#   psola_form.html      — self-contained, mobile-first, BLIND (no condition / tone_i2 / GT leaked). Same intro,
#                          buttons and copy/CSV machinery as the pilot (bp.render_html, embedded VERBATIM).
#   keymap_psola.json    — WITHHELD ground truth, one entry per item (see KEYMAP SCHEMA at the bottom of main()).
#
# RUN (audio env / Colab with AWS creds + tone_metric + parselmouth):
#   python build_psola_form.py                 # full build
#   python build_psola_form.py --dry-run       # list chosen clips, manipulate NOTHING
#   python build_psola_form.py --n-clips 14 --k 1.25 --n-catch 6 --seed 4242

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                 # so `import build_pilot_form` resolves when run as a script

import build_pilot_form as bp               # reuse: S3 plumbing, native loader, HTML, Scorer, tone_i2 aggregation

# tone_metric primitives (the package bp.Scorer already depends on, so it is importable wherever this runs)
try:
    from tone_metric import tone_eval_v2 as v2
    from tone_metric import tone_oracle as oracle
    from tone_metric import tone_f0_abs as f0a
except Exception as e:  # pragma: no cover
    sys.exit(f"FATAL: cannot import tone_metric (tone_eval_v2 / tone_oracle / tone_f0_abs): {e}. "
             f"Install it (pip install git+https://github.com/mosesdaudu001/tone-on-a-budget.git).")

SR = bp.SR
PSOLA_DUR = (2.0, 5.0)        # SHORT clips: one flipped syllable is more salient in a 2-5s clip (task spec)
MIN_DELTA_HL = 2.0            # skip clips whose H/L residual spread < ~theta_h+theta_l (~2 st): below this a
                             # k=1.25 flip may only nudge L->M, not cross the opposite tone band. At 2.0 the
                             # minimum flip is 1.25*2.0 = 2.5 st, guaranteed past the band. Ear-check + the
                             # flipped-acc>50% gate remain the runtime backstop.


# ----------------------------------------------------------------------------- deployed-metric scoring (mid_ref=None)
def score_full(scorer, wav, text):
    """tone_i2 scalar + the full f0_abs dict (pred/target/coverage) + the v2 precompute, computed the SAME way
    the metric is DEPLOYED (one shared MMS forward, blind Theil-Sen detrend, mid_ref=None per-utterance anchor).
    Returns (tone_i2, f0abs_dict, pre)."""
    logits, n16 = scorer.rm.asr_logits(wav, SR)
    pre = v2.precompute(wav, SR, text, asr=scorer.rm.asr, proc=scorer.rm.asr_proc,
                        device=scorer.device, emissions=logits, n16=n16)
    d = f0a.score_abs_from_precomputed(pre, theta_h=scorer.th, theta_l=scorer.tl,
                                       mode=scorer.mode, mid_ref=None, late_frac=scorer.late)
    pairs = [(p, t) for p, t in zip(d["pred"], d["target"]) if p is not None]
    return bp._bal_tone_acc(pairs), d, pre


# ----------------------------------------------------------------------------- one native clip -> a flip pair
def process_clip(scorer, clip, k, rng):
    """Score the CLEAN native clip, pick ONE eligible TBU (H or L, aligned, read CORRECTLY by the clean meter),
    and build the two artifact-matched twins. Returns a dict of everything needed to emit two blind items + the
    withheld key, or None if the clip has no usable flip."""
    import numpy as np
    wav = np.asarray(clip["wav"], dtype="float32").reshape(-1)
    text = clip["text"]
    ti2_clean, clean, pre = score_full(scorer, wav, text)

    dHL = oracle.measure_delta_HL(pre)                       # the flip unit (declination-removed H-L spread)
    if dHL is None or dHL < MIN_DELTA_HL:
        return None
    tones, wins, pred = pre["tones"], pre["wins"], clean["pred"]
    elig = [i for i, (t, w) in enumerate(zip(tones, wins))
            if t in ("H", "L") and w is not None
            and i < len(pred) and pred[i] == t]             # correctly-read, unambiguous, aligned H/L syllable
    if not elig:
        return None
    i = rng.choice(elig)
    t0, t1 = wins[i]
    sign = 1.0 if tones[i] == "L" else -1.0                 # push L up toward H, or H down toward L
    st = sign * float(k) * float(dHL)
    flip_dir = "L->H" if tones[i] == "L" else "H->L"

    correct_wav = oracle.psola_roundtrip(wav, sr=SR)        # tone unchanged, full PSOLA artifact
    flipped_wav = oracle.psola_shift_window(wav, t0, t1, st, sr=SR)   # ONE TBU flipped, SAME artifact
    ti2_correct, _, _ = score_full(scorer, correct_wav, text)
    ti2_flipped, _, _ = score_full(scorer, flipped_wav, text)

    return dict(clip_id=clip["clip_id"], text=text, tbu_index=int(i), flip_dir=flip_dir,
                dHL=float(dHL), semitones=float(st), tone_i2_clean=ti2_clean,
                correct=dict(wav=correct_wav, tone_i2=ti2_correct),
                flipped=dict(wav=flipped_wav, tone_i2=ti2_flipped))


def make_catch(scorer, clip, k_catch, rng, kind):
    """A screening item with an OBVIOUS answer. kind='catch_flipped' -> a HUGE flip (must sound WRONG, expect ✗);
    kind='catch_correct' -> a plain roundtrip (must sound RIGHT, expect ✓). Reuses the same native pool."""
    import numpy as np
    wav = np.asarray(clip["wav"], dtype="float32").reshape(-1)
    text = clip["text"]
    if kind == "catch_correct":
        w = oracle.psola_roundtrip(wav, sr=SR)
        ti2, _, _ = score_full(scorer, w, text)
        return dict(clip_id=clip["clip_id"], text=text, wav=w, condition="catch_correct", expect="ok",
                    flip_dir=None, tbu_index=None, semitones=0.0, tone_i2=ti2, is_catch=True)
    # catch_flipped: need an eligible TBU to flip hard
    _ti2c, clean, pre = score_full(scorer, wav, text)
    tones, wins, pred = pre["tones"], pre["wins"], clean["pred"]
    dHL = oracle.measure_delta_HL(pre)
    elig = [i for i, (t, w) in enumerate(zip(tones, wins))
            if t in ("H", "L") and w is not None and i < len(pred) and pred[i] == t]
    if dHL is None or dHL < MIN_DELTA_HL or not elig:
        return None
    i = rng.choice(elig)
    t0, t1 = wins[i]
    sign = 1.0 if tones[i] == "L" else -1.0
    st = sign * float(k_catch) * float(dHL)
    # huge k_catch flip can push pitch well above the default 400 Hz ceiling on high voices -> widen the
    # bracket so the resynthesis stays clean (tone_oracle warns f0max MUST bracket the shifted pitch).
    w = oracle.psola_shift_window(wav, t0, t1, st, sr=SR, f0max=600.0)
    ti2, _, _ = score_full(scorer, w, text)
    return dict(clip_id=clip["clip_id"], text=text, wav=w, condition="catch_flipped", expect="bad",
                flip_dir=("L->H" if tones[i] == "L" else "H->L"), tbu_index=int(i),
                semitones=float(st), tone_i2=ti2, is_catch=True)


# ----------------------------------------------------------------------------- shuffle that keeps twins apart
def separate_pairs(items, key="pair_id"):
    """In-place reorder so no two ADJACENT items share a pair_id (a clip's correct + flipped twin never abut)."""
    n = len(items)
    for _ in range(400):
        conflict = next((k for k in range(n - 1) if items[k][key] == items[k + 1][key]), None)
        if conflict is None:
            return items
        k = conflict
        moved = False
        for j in range(n):
            if j in (k, k + 1):
                continue
            a = items[k + 1][key]
            b = items[j][key]
            ok_here = items[k][key] != b and (k + 2 >= n or items[k + 2][key] != b)
            ok_there = (j == 0 or items[j - 1][key] != a) and (j + 1 >= n or items[j + 1][key] != a)
            if ok_here and ok_there:
                items[k + 1], items[j] = items[j], items[k + 1]
                moved = True
                break
        if not moved:
            return items                                    # give up (rare: too few distinct pairs)
    return items


# ----------------------------------------------------------------------------- assemble
def main():
    ap = argparse.ArgumentParser(description="Build the PSOLA-isolated Yorùbá tone listening test.")
    ap.add_argument("--n-clips", type=int, default=14, help="number of native clips -> that many flip PAIRS")
    ap.add_argument("--k", type=float, default=1.25, help="flip size in units of ΔHL (k=1.0 = full H<->L distance)")
    ap.add_argument("--k-catch", type=float, default=2.5, help="HUGE-flip size for catch items (obviously wrong)")
    ap.add_argument("--n-catch", type=int, default=6, help="catch items (~half huge-flip ✗, half roundtrip ✓)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--per-spk-cap", type=int, default=8, help="max clips per studio speaker (voice control)")
    ap.add_argument("--bitrate", default="32k", help="mp3 bitrate for embedded clips (ffmpeg)")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--dry-run", action="store_true", help="list chosen clips; manipulate / score NOTHING")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # fail loud on the hard prerequisites (parselmouth is the whole point of this kit)
    try:
        import parselmouth  # noqa: F401
    except Exception as e:
        sys.exit(f"FATAL: praat-parselmouth is required for PSOLA but is not importable ({e}). "
                 f"pip install praat-parselmouth (run in the audio env, e.g. /home/moses/audio_env).")

    s3 = bp.connect_s3()

    # over-select: many clips are skipped (no H/L spread, or no correctly-read aligned H/L TBU)
    need = args.n_clips + args.n_catch
    pool = bp.select_native(s3, max(need * 5, 40), args.per_spk_cap, rng)
    print(f"[select] candidate native clips: {len(pool)} "
          f"(speakers: {len({r['speaker'] for r in pool})})", flush=True)

    if args.dry_run:
        print("\n=== DRY RUN — candidate NATIVE clips (no PSOLA, no scoring, nothing written) ===")
        for r in pool[:need * 3]:
            print(f"  {r['clip_id']:>12} spk {r['speaker']:>6} {r['dur']:.1f}s | {r['text'][:60]}")
        print(f"\nwould build ~{args.n_clips} flip PAIRS (k={args.k}·ΔHL) + {args.n_catch} catch "
              f"(k_catch={args.k_catch}); shuffle seed {args.seed}.")
        print("DRY RUN complete.")
        return

    work = tempfile.mkdtemp(prefix="psola_")
    scorer = bp.Scorer(s3)                                  # MMS-yor + frozen F0 calibration (heavy; lazy)

    # download + PSOLA-process candidates until we have enough valid flip pairs (+ a few reserved for catch)
    pairs, reserved, used = [], [], 0
    for r in pool:
        if len(pairs) >= args.n_clips and len(reserved) >= args.n_catch:
            break
        lp = os.path.join(work, f"nat_{r['clip_id']}.wav")
        try:
            s3.download_file(bp.BUCKET, r["audio_s3_key"], lp)
            clip = dict(clip_id=r["clip_id"], text=r["text"], wav=bp._read_wav(lp))
        except Exception as e:
            print(f"  [skip] download/read {r['clip_id']}: {e}", flush=True)
            continue
        used += 1
        try:
            res = process_clip(scorer, clip, args.k, rng)
        except Exception as e:
            print(f"  [skip] process {r['clip_id']}: {e}", flush=True)
            continue
        if res is None:
            continue
        if len(pairs) < args.n_clips:
            pairs.append(res)
        else:
            reserved.append(clip)                          # keep the raw clip for catch generation
        print(f"  [ok] {r['clip_id']}  flip {res['flip_dir']}  ΔHL={res['dHL']:.2f}st  "
              f"shift={res['semitones']:+.2f}st  tbu#{res['tbu_index']}  "
              f"tone_i2 clean/correct/flipped={res['tone_i2_clean']:.3f}/"
              f"{res['correct']['tone_i2']:.3f}/{res['flipped']['tone_i2']:.3f}", flush=True)

    if not pairs:
        sys.exit("FATAL: no native clip yielded a usable flip (no H/L spread or no correctly-read aligned H/L "
                 "TBU). Try --per-spk-cap higher or widen the manifest.")

    # ---- build catch items (reuse reserved raw clips; fall back to pair clips if the pool ran dry) ----
    n_huge = args.n_catch // 2
    n_round = args.n_catch - n_huge
    catch_src = reserved + [dict(clip_id=p["clip_id"], text=p["text"],
                                 wav=p["correct"]["wav"]) for p in pairs]  # fallback sources
    catch_items, ci = [], 0
    for kind, count in (("catch_flipped", n_huge), ("catch_correct", n_round)):
        made = 0
        for clip in catch_src[ci:]:
            ci += 1
            try:
                c = make_catch(scorer, clip, args.k_catch, rng, kind)
            except Exception as e:
                print(f"  [skip] catch {kind} {clip['clip_id']}: {e}", flush=True)
                c = None
            if c is None:
                continue
            catch_items.append(c)
            made += 1
            if made >= count:
                break

    # ---- flatten to blind items + withheld keymap ----
    items = []   # each: dict(text, wav, clip_id, condition, expect, flip_dir, tbu_index, semitones,
                 #            tone_i2, is_catch, pair_id)
    for p in pairs:
        pid = f"pair_{p['clip_id']}"
        items.append(dict(text=p["text"], wav=p["correct"]["wav"], clip_id=p["clip_id"],
                          condition="correct", expect="ok", flip_dir=None, tbu_index=p["tbu_index"],
                          semitones=0.0, tone_i2=p["correct"]["tone_i2"], is_catch=False, pair_id=pid))
        items.append(dict(text=p["text"], wav=p["flipped"]["wav"], clip_id=p["clip_id"],
                          condition="flipped", expect="bad", flip_dir=p["flip_dir"], tbu_index=p["tbu_index"],
                          semitones=p["semitones"], tone_i2=p["flipped"]["tone_i2"], is_catch=False, pair_id=pid))
    for n, c in enumerate(catch_items):
        items.append(dict(text=c["text"], wav=c["wav"], clip_id=c["clip_id"], condition=c["condition"],
                          expect=c["expect"], flip_dir=c["flip_dir"], tbu_index=c["tbu_index"],
                          semitones=c["semitones"], tone_i2=c["tone_i2"], is_catch=True,
                          pair_id=f"catch_{n:02d}"))

    # shuffle (fixed seed -> keymap matches), then pull a clip's two twins apart
    random.Random(args.seed).shuffle(items)
    separate_pairs(items, key="pair_id")

    # ---- encode audio + write the two files ----
    items_for_js, keymap = [], {}
    total_bytes = 0
    for n, c in enumerate(items, 1):
        item_id = f"item{n:02d}"
        uri, nbytes = bp.encode_data_uri(c["wav"], work, item_id, bitrate=args.bitrate, max_seconds=PSOLA_DUR[1] + 0.5)
        total_bytes += nbytes
        items_for_js.append(dict(item_id=item_id, text=c["text"], audio=uri))
        ti2 = c["tone_i2"]
        keymap[item_id] = dict(
            clip_id=c["clip_id"], condition=c["condition"], expect=c["expect"], flip_dir=c["flip_dir"],
            tbu_index=c["tbu_index"], semitones=round(float(c["semitones"]), 4),
            intended_text=c["text"],
            tone_i2=(None if ti2 is None or ti2 != ti2 else round(float(ti2), 4)),
            is_catch=bool(c["is_catch"]), pair_id=c["pair_id"])

    os.makedirs(args.out_dir, exist_ok=True)
    html_path = os.path.join(args.out_dir, "psola_form.html")
    key_path = os.path.join(args.out_dir, "keymap_psola.json")
    open(html_path, "w", encoding="utf-8").write(bp.render_html(items_for_js, args.seed))
    json.dump(keymap, open(key_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- report + ear-check list (correct+flipped twins of the first few real pairs) ----
    cond_counts = Counter(v["condition"] for v in keymap.values())
    id_by_pair = {}
    for iid, v in keymap.items():
        if not v["is_catch"]:
            id_by_pair.setdefault(v["pair_id"], {})[v["condition"]] = iid
    ear_pairs = [(pid, d.get("correct"), d.get("flipped"))
                 for pid, d in id_by_pair.items() if "correct" in d and "flipped" in d][:6]
    html_mb = os.path.getsize(html_path) / 1e6
    print("\n" + "=" * 70)
    print(f"  wrote {html_path}  ({html_mb:.2f} MB, audio ~{total_bytes/1e6:.2f} MB @ {args.bitrate})")
    print(f"  wrote {key_path}  ({len(keymap)} items: {dict(cond_counts)})")
    print(f"  clips processed: {used}; flip pairs: {len(pairs)}; catch: {len(catch_items)}")
    print(f"  EAR-CHECK these {len(ear_pairs)} pairs in nb16 (listen: flipped must sound like WRONG TONE):")
    for pid, cid, fid in ear_pairs:
        print(f"     {pid}: correct={cid}  flipped={fid}")
    print("=" * 70)

    # KEYMAP SCHEMA (per item_id): {clip_id, condition['correct'|'flipped'|'catch_correct'|'catch_flipped'],
    #   expect['ok'|'bad'], flip_dir['L->H'|'H->L'|None], tbu_index, semitones, intended_text, tone_i2,
    #   is_catch, pair_id}. WITHHELD — never sent to the reviewer.


if __name__ == "__main__":
    main()
