#!/usr/bin/env python3
# coding=utf-8
# build_psola_form.py — assemble a PSOLA-isolated, blind, A/B-PAIRED Yorùbá TONE listening test.
#
# WHY THE A/B REBUILD (PSOLA_RECIPE.md — the authoritative spec)
#   The prior kit's flips were INAUDIBLE: it shifted F0 only inside the ~20-60 ms MMS-CTC sliver wins[i], a
#   handful of pitch pulses, while the surrounding original contour dominated the percept (proof: 6/14 pairs
#   had byte-identical deployed tone_i2 between correct and flipped twin). THE FIX (this file):
#     1. window  = the WHOLE voiced rhyme, grown from the F0 contour  (oracle.voiced_rhyme_window)
#     2. flip    = RESET the rhyme's contour to the OPPOSITE tone band (oracle.build_flip_contour +
#                  oracle.psola_set_contour), NOT a blind multiply of a sliver
#     3. correct = re-impose the clip's OWN measured contour over the SAME rhyme via the SAME dense-pitch-tier
#                  machinery (oracle.reimpose_contour) — so BOTH twins carry an IDENTICAL artifact
#     4. HARD salience self-check: re-extract F0 from the SYNTHESIZED flipped wav and REJECT the clip unless
#        the flip actually LANDED in the opposite band (realized ΔF0 ≥ 3 st, frozen-mid_ref residual past the
#        band, frozen-mid_ref pred flips class). No failing clip enters the keymap or HTML.
#
# THE TASK (A/B forced choice): a TRIAL = one sentence, two clips A and B of the SAME utterance/voice/artifact
#   — the correct twin and the flipped twin, with the correct side RANDOMIZED per trial (side_map in keymap).
#   The reviewer picks whichever sounds like correct Yorùbá.
#
# mid_ref — TWO scorings (PSOLA_RECIPE.md §4):
#   tone_i2_frozen   : mid_ref FROZEN from the CLEAN clip's I2 mid_ref field. The metric-core sensitivity column
#                      — ALL metric↔human agreement (paired-win / AUROC / point-biserial) uses this.
#   tone_i2_deployed : mid_ref=None (per-utterance anchor, as shipped nb07/nb14/poster). Reported only as the
#                      localized-blindness caveat; NEVER used for the agreement claim.
#
# OUTPUTS (into --out-dir, default this dir):
#   psola_form.html      — self-contained, mobile-first, BLIND A/B form (no side_map / condition / tone_i2 leaked).
#   keymap_psola.json    — WITHHELD ground truth, one entry per TRIAL (see KEYMAP SCHEMA at the bottom of main()).
#
# RUN (audio env / Colab with AWS creds + tone_metric + parselmouth):
#   python build_psola_form.py                 # full build (default --n-clips 24)
#   python build_psola_form.py --dry-run       # list chosen clips, manipulate NOTHING
#   python build_psola_form.py --n-clips 24 --n-catch 5 --seed 4242

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                 # so `import build_pilot_form` resolves when run as a script

import build_pilot_form as bp               # reuse: S3 plumbing, native loader, encode_data_uri, Scorer, _bal_tone_acc

# tone_metric primitives (the package bp.Scorer already depends on, so it is importable wherever this runs)
try:
    from tone_metric import tone_eval_v2 as v2
    from tone_metric import tone_oracle as oracle
    from tone_metric import tone_f0_abs as f0a
except Exception as e:  # pragma: no cover
    sys.exit(f"FATAL: cannot import tone_metric (tone_eval_v2 / tone_oracle / tone_f0_abs): {e}. "
             f"Install it (pip install git+https://github.com/mosesdaudu001/tone-on-a-budget.git).")

SR = bp.SR
PSOLA_DUR = (2.0, 5.0)        # SHORT clips: one flipped syllable is more salient in a 2-5s clip
MIN_DELTA_HL = 2.0            # require ≥2 st between the clip's median H and median L residual: below this the
                             # opposite-band reset is not cleanly separable. The realized-F0 self-check is the
                             # hard runtime backstop regardless.
MIN_REALIZED_ST = 3.0        # salience self-check (a): |median F0(flipped) − median F0(correct)| over the rhyme
                             # must be ≥ this many semitones, measured on the RE-EXTRACTED synthesized audio.


# --------------------------------------------------------------- deployed/frozen-metric scoring (mid_ref switch)
def score_full(scorer, wav, text, mid_ref=None):
    """tone_i2 scalar + the full f0_abs dict (pred/target/mid_ref) + the v2 precompute, scored the SAME way the
    metric runs (one shared MMS forward, blind Theil-Sen detrend). mid_ref=None => deployed per-utterance anchor;
    mid_ref=<float> => FROZEN-clean anchor. Returns (tone_i2, f0abs_dict, pre)."""
    logits, n16 = scorer.rm.asr_logits(wav, SR)
    # fmax=600 on EVERY path (PSOLA_RECIPE.md §2/§7): the frozen-mid_ref residual self-check must measure the
    # FULL synthesized range, whose ceiling is oracle.F0MAX=600; the 400 Hz default would clip a high-voice flip.
    pre = v2.precompute(wav, SR, text, asr=scorer.rm.asr, proc=scorer.rm.asr_proc,
                        device=scorer.device, emissions=logits, n16=n16, fmax=600.0)
    d = f0a.score_abs_from_precomputed(pre, theta_h=scorer.th, theta_l=scorer.tl,
                                       mode=scorer.mode, mid_ref=mid_ref, late_frac=scorer.late)
    pairs = [(p, t) for p, t in zip(d["pred"], d["target"]) if p is not None]
    return bp._bal_tone_acc(pairs), d, pre


# --------------------------------------------------------------- F0 / residual probes (pure, model-free helpers)
def _median_st_in_window(wav, t0, t1):
    """Re-extract F0 from a SYNTHESIZED wav and return the median semitone (100 Hz ref) over the rhyme's voiced
    frames, or None. This measures the REALIZED contour of the synthesized audio — the salience ground truth."""
    import numpy as np
    # fmax=600 matches the PSOLA synthesis ceiling (oracle.F0MAX) on EVERY path (PSOLA_RECIPE.md §2/§7):
    # a high-female H-flip can realize F0 > the 400 Hz default and would be mis-tracked by the verifier.
    f0, times, _ = v2.extract_f0_v2(np.asarray(wav, dtype="float32").reshape(-1), SR, fmax=600.0)
    f0 = np.asarray(f0, dtype="float64")
    times = np.asarray(times, dtype="float64")
    m = (times >= t0) & (times < t1) & (~np.isnan(f0))
    if not m.any():
        return None
    return 12.0 * math.log2(max(float(np.nanmedian(f0[m])), 1e-6) / 100.0)


def _residual_at(pre, i):
    """Blind declination-removed residual of TBU i (the value the frozen-mid_ref classifier thresholds), or None."""
    sts = v2._tbu_semitones(pre)
    res, _slope = f0a._blind_residuals(sts)
    return res[i] if i < len(res) else None


def salience_check(med_st_flipped, med_st_correct, r_flipped, mid_ref, theta_h, theta_l,
                   pred_flipped, expect, min_realized_st=MIN_REALIZED_ST):
    """HARD salience predicate (PSOLA_RECIPE.md §2) — PURE, no audio. The flip is accepted ONLY if all three:
      (a) |median F0(flipped) − median F0(correct)| over the rhyme ≥ min_realized_st  (REALIZED on synth audio)
      (b) the flipped residual lands in the OPPOSITE band under the frozen-clean mid_ref:
            expect 'H' => r ≥ mid_ref + theta_h ;  expect 'L' => r ≤ mid_ref − theta_l
      (c) the frozen-mid_ref oracle prediction at this TBU flipped to the opposite class (pred == expect)
    Returns a dict of the sub-results + 'passed'."""
    realized = (abs(med_st_flipped - med_st_correct)
                if (med_st_flipped is not None and med_st_correct is not None) else None)
    delta_ok = realized is not None and realized >= float(min_realized_st)
    if expect == "H":
        band_ok = (r_flipped is not None and mid_ref is not None and r_flipped >= mid_ref + theta_h)
    else:  # expect 'L'
        band_ok = (r_flipped is not None and mid_ref is not None and r_flipped <= mid_ref - theta_l)
    pred_ok = (pred_flipped == expect)
    return dict(realized_st=realized, delta_ok=bool(delta_ok), band_ok=bool(band_ok),
                pred_flip_ok=bool(pred_ok), passed=bool(delta_ok and band_ok and pred_ok))


# --------------------------------------------------------------- one native clip -> an artifact-matched A/B pair
def process_clip(scorer, clip, rng):
    """Score the CLEAN native clip, freeze its mid_ref, then for each eligible TBU grow the FULL voiced rhyme,
    RESET it to the opposite tone band (flipped twin) and re-impose the original contour over the same rhyme
    (correct twin). Enforce the §2 hard salience self-check on the SYNTHESIZED flip; the FIRST TBU that passes
    is shipped. Returns a pair dict, or None if no TBU yields an audible, in-band, class-flipping reset."""
    import numpy as np
    wav = np.asarray(clip["wav"], dtype="float32").reshape(-1)
    text = clip["text"]
    _ti2_clean, clean, pre = score_full(scorer, wav, text, mid_ref=None)
    mid_ref_clean = clean.get("mid_ref")
    if mid_ref_clean is None:
        return None

    medH, medL, _mid_blind, slope = oracle.tone_level_residuals(pre)
    if not (medH == medH and medL == medL) or (medH - medL) < MIN_DELTA_HL:
        return None

    tones, wins, pred = pre["tones"], pre["wins"], clean["pred"]
    elig = [k for k, (t, w) in enumerate(zip(tones, wins))
            if t in ("H", "L") and w is not None and k < len(pred) and pred[k] == t]
    rng.shuffle(elig)
    th, tl = scorer.th, scorer.tl

    for i in elig:
        rhyme = oracle.voiced_rhyme_window(pre, i)
        if rhyme is None:
            continue
        t0, t1 = rhyme
        src = tones[i]
        expect = "H" if src == "L" else "L"
        # target the opposite band, guaranteeing residual separation past the frozen-clean Mid ± theta
        if expect == "H":
            r_target = max(medH, mid_ref_clean + th + 0.5)
            tilt = -1.0                                  # H: gentle natural fall (st/s) on top of declination
        else:
            r_target = min(medL, mid_ref_clean - tl - 0.5)
            tilt = -0.2                                  # L: near-flat
        contour = oracle.build_flip_contour(t0, t1, r_target, slope + tilt)
        flipped_wav = oracle.psola_set_contour(wav, t0, t1, contour, sr=SR, f0max=600.0)
        correct_wav = oracle.reimpose_contour(wav, pre, t0, t1, sr=SR, f0max=600.0)

        # ---- HARD salience self-check on the SYNTHESIZED audio ----
        med_flip = _median_st_in_window(flipped_wav, t0, t1)
        med_corr = _median_st_in_window(correct_wav, t0, t1)
        ti2_f_fz, d_fz, pre_flip = score_full(scorer, flipped_wav, text, mid_ref=mid_ref_clean)
        r_flip = _residual_at(pre_flip, i)
        pred_flip = d_fz["pred"][i] if i < len(d_fz["pred"]) else None
        sal = salience_check(med_flip, med_corr, r_flip, mid_ref_clean, th, tl, pred_flip, expect)
        if not sal["passed"]:
            continue

        # ---- both twins, scored TWICE (frozen + deployed) ----
        ti2_c_fz, _, _ = score_full(scorer, correct_wav, text, mid_ref=mid_ref_clean)
        ti2_c_dp, _, _ = score_full(scorer, correct_wav, text, mid_ref=None)
        ti2_f_dp, _, _ = score_full(scorer, flipped_wav, text, mid_ref=None)
        return dict(clip_id=clip["clip_id"], text=text, tbu_index=int(i), flip_dir=f"{src}->{expect}",
                    rhyme=[float(t0), float(t1)], r_target=float(r_target), slope=float(slope),
                    realized_st=float(sal["realized_st"]), mid_ref_clean=float(mid_ref_clean),
                    correct=dict(wav=correct_wav, tone_i2_frozen=ti2_c_fz, tone_i2_deployed=ti2_c_dp),
                    flipped=dict(wav=flipped_wav, tone_i2_frozen=ti2_f_fz, tone_i2_deployed=ti2_f_dp),
                    salience=sal)
    return None


def make_catch(scorer, clip, rng):
    """An OBVIOUS A/B catch: one side the clean re-imposed twin (correct), one side an EXTREME opposite-band
    reset (clearly wrong tone). Falls back to whole-clip psola_flatten if no eligible TBU. Returns a dict with
    wav_correct + wav_bad, or None."""
    import numpy as np
    wav = np.asarray(clip["wav"], dtype="float32").reshape(-1)
    text = clip["text"]
    _ti2, clean, pre = score_full(scorer, wav, text, mid_ref=None)
    mid_ref_clean = clean.get("mid_ref")
    medH, medL, _m, slope = oracle.tone_level_residuals(pre)
    th, tl = scorer.th, scorer.tl
    if mid_ref_clean is not None and medH == medH and medL == medL:
        tones, wins, pred = pre["tones"], pre["wins"], clean["pred"]
        elig = [k for k, (t, w) in enumerate(zip(tones, wins))
                if t in ("H", "L") and w is not None and k < len(pred) and pred[k] == t]
        rng.shuffle(elig)
        for i in elig:
            rhyme = oracle.voiced_rhyme_window(pre, i)
            if rhyme is None:
                continue
            t0, t1 = rhyme
            src = tones[i]
            expect = "H" if src == "L" else "L"
            if expect == "H":
                r_target = max(medH, mid_ref_clean + th) + 3.0      # EXTREME — unmistakable
            else:
                r_target = min(medL, mid_ref_clean - tl) - 3.0
            contour = oracle.build_flip_contour(t0, t1, r_target, slope)
            bad = oracle.psola_set_contour(wav, t0, t1, contour, sr=SR, f0max=600.0)
            good = oracle.reimpose_contour(wav, pre, t0, t1, sr=SR, f0max=600.0)
            # salience self-check (parity with process_clip): an EXTREME catch flip must actually land. If
            # parselmouth no-op'd, the realized ΔF0 collapses — skip this TBU (else fall through to flatten).
            med_bad = _median_st_in_window(bad, t0, t1)
            med_good = _median_st_in_window(good, t0, t1)
            if med_bad is None or med_good is None or abs(med_bad - med_good) < 6.0:
                continue
            return dict(clip_id=clip["clip_id"], text=text, wav_correct=good, wav_bad=bad,
                        tbu_index=int(i), flip_dir=f"{src}->{expect}(extreme)")
    # fallback: monotone whole clip is obviously wrong tone vs the natural recording
    bad = oracle.psola_flatten(wav, sr=SR, f0max=600.0)
    return dict(clip_id=clip["clip_id"], text=text, wav_correct=wav, wav_bad=bad,
                tbu_index=None, flip_dir="flatten")


# --------------------------------------------------------------- assemble
def main():
    ap = argparse.ArgumentParser(description="Build the PSOLA-isolated A/B Yorùbá tone listening test.")
    ap.add_argument("--n-clips", type=int, default=24, help="number of native clips -> that many A/B flip PAIRS")
    ap.add_argument("--n-catch", type=int, default=5, help="catch A/B trials (clean vs EXTREME flip)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--per-spk-cap", type=int, default=8, help="max clips per studio speaker (voice control)")
    ap.add_argument("--bitrate", default="32k", help="mp3 bitrate for embedded clips (ffmpeg)")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--dry-run", action="store_true", help="list chosen clips; manipulate / score NOTHING")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    try:
        import parselmouth  # noqa: F401
    except Exception as e:
        sys.exit(f"FATAL: praat-parselmouth is required for PSOLA but is not importable ({e}). "
                 f"pip install praat-parselmouth (run in the audio env, e.g. /home/moses/audio_env).")

    s3 = bp.connect_s3()

    # over-select: many clips are skipped (no H/L spread, no aligned correctly-read H/L TBU, or salience fail)
    need = args.n_clips + args.n_catch
    pool = bp.select_native(s3, max(need * 6, 60), args.per_spk_cap, rng)
    print(f"[select] candidate native clips: {len(pool)} "
          f"(speakers: {len({r['speaker'] for r in pool})})", flush=True)

    if args.dry_run:
        print("\n=== DRY RUN — candidate NATIVE clips (no PSOLA, no scoring, nothing written) ===")
        for r in pool[:need * 3]:
            print(f"  {r['clip_id']:>12} spk {r['speaker']:>6} {r['dur']:.1f}s | {r['text'][:60]}")
        print(f"\nwould build ~{args.n_clips} A/B flip PAIRS (whole-rhyme opposite-band reset) + "
              f"{args.n_catch} catch; shuffle seed {args.seed}.")
        print("DRY RUN complete.")
        return

    work = tempfile.mkdtemp(prefix="psola_")
    scorer = bp.Scorer(s3)                                  # MMS-yor + frozen F0 calibration (heavy; lazy)

    pairs, reserved = [], []
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
        if len(pairs) < args.n_clips:
            try:
                res = process_clip(scorer, clip, rng)
            except Exception as e:
                print(f"  [skip] process {r['clip_id']}: {e}", flush=True)
                continue
            if res is None:
                continue
            pairs.append(res)
            print(f"  [ok] {r['clip_id']}  flip {res['flip_dir']}  tbu#{res['tbu_index']}  "
                  f"realizedΔ={res['realized_st']:.2f}st  "
                  f"i2_frozen c/f={res['correct']['tone_i2_frozen']:.3f}/{res['flipped']['tone_i2_frozen']:.3f}  "
                  f"i2_deployed c/f={res['correct']['tone_i2_deployed']:.3f}/"
                  f"{res['flipped']['tone_i2_deployed']:.3f}", flush=True)
        else:
            reserved.append(clip)

    if not pairs:
        sys.exit("FATAL: no native clip yielded an audible, in-band, class-flipping rhyme reset. "
                 "Try --per-spk-cap higher or widen the manifest.")

    # ---- catch trials (reuse reserved clips; fall back to pair clips' correct twins) ----
    catch_src = reserved + [dict(clip_id=p["clip_id"], text=p["text"], wav=p["correct"]["wav"]) for p in pairs]
    catch = []
    for clip in catch_src:
        if len(catch) >= args.n_catch:
            break
        try:
            c = make_catch(scorer, clip, rng)
        except Exception as e:
            print(f"  [skip] catch {clip['clip_id']}: {e}", flush=True)
            c = None
        if c is not None:
            catch.append(c)

    # ---- build A/B TRIALS with the correct side randomized per trial ----
    def _sides(correct_wav, flipped_wav):
        """Return (wavA, wavB, side_map) where side_map ∈ {'A','B'} = which side holds the CORRECT twin."""
        if rng.random() < 0.5:
            return correct_wav, flipped_wav, "A"
        return flipped_wav, correct_wav, "B"

    trials = []   # each: dict(pair_id, text, wavA, wavB, is_catch, side_map, expect_pick, tone_i2 fields, salience)
    for p in pairs:
        wavA, wavB, side = _sides(p["correct"]["wav"], p["flipped"]["wav"])
        sal = p["salience"]
        trials.append(dict(
            pair_id=f"pair_{p['clip_id']}", text=p["text"], wavA=wavA, wavB=wavB, is_catch=False,
            side_map=side, expect_pick=None, flip_dir=p["flip_dir"], tbu_index=p["tbu_index"],
            realized_st=p["realized_st"], delta_ok=bool(sal["delta_ok"]),
            band_ok=bool(sal["band_ok"]), pred_flip_ok=bool(sal["pred_flip_ok"]),
            tone_i2_frozen_correct=p["correct"]["tone_i2_frozen"],
            tone_i2_frozen_flipped=p["flipped"]["tone_i2_frozen"],
            tone_i2_deployed_correct=p["correct"]["tone_i2_deployed"],
            tone_i2_deployed_flipped=p["flipped"]["tone_i2_deployed"]))
    for n, c in enumerate(catch):
        wavA, wavB, side = _sides(c["wav_correct"], c["wav_bad"])
        trials.append(dict(
            pair_id=f"catch_{n:02d}", text=c["text"], wavA=wavA, wavB=wavB, is_catch=True,
            side_map=side, expect_pick=side, flip_dir=c["flip_dir"], tbu_index=c["tbu_index"],
            realized_st=None, delta_ok=None, band_ok=None, pred_flip_ok=None,
            tone_i2_frozen_correct=None, tone_i2_frozen_flipped=None,
            tone_i2_deployed_correct=None, tone_i2_deployed_flipped=None))

    random.Random(args.seed).shuffle(trials)

    # ---- encode the two sides of each trial + write the two files ----
    items_for_js, keymap = [], {}
    total_bytes = 0
    for n, t in enumerate(trials, 1):
        item_id = f"item{n:02d}"
        uriA, ba = bp.encode_data_uri(t["wavA"], work, item_id + "A", bitrate=args.bitrate,
                                      max_seconds=PSOLA_DUR[1] + 0.5)
        uriB, bb = bp.encode_data_uri(t["wavB"], work, item_id + "B", bitrate=args.bitrate,
                                      max_seconds=PSOLA_DUR[1] + 0.5)
        total_bytes += ba + bb
        items_for_js.append(dict(item_id=item_id, text=t["text"], audioA=uriA, audioB=uriB))

        def _r(x):
            return None if (x is None or x != x) else round(float(x), 4)
        keymap[item_id] = dict(
            pair_id=t["pair_id"], intended_text=t["text"], side_map=t["side_map"],
            tone_i2_frozen_correct=_r(t["tone_i2_frozen_correct"]),
            tone_i2_frozen_flipped=_r(t["tone_i2_frozen_flipped"]),
            tone_i2_deployed_correct=_r(t["tone_i2_deployed_correct"]),
            tone_i2_deployed_flipped=_r(t["tone_i2_deployed_flipped"]),
            condition=("catch" if t["is_catch"] else "pair"), is_catch=bool(t["is_catch"]),
            expect_pick=t["expect_pick"], flip_dir=t["flip_dir"], tbu_index=t["tbu_index"],
            realized_st=_r(t["realized_st"]), delta_ok=t["delta_ok"], band_ok=t["band_ok"],
            pred_flip_ok=t["pred_flip_ok"])

    os.makedirs(args.out_dir, exist_ok=True)
    html_path = os.path.join(args.out_dir, "psola_form.html")
    key_path = os.path.join(args.out_dir, "keymap_psola.json")
    open(html_path, "w", encoding="utf-8").write(bp.render_ab_html(items_for_js, args.seed))
    json.dump(keymap, open(key_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- report ----
    cond_counts = Counter(v["condition"] for v in keymap.values())
    html_mb = os.path.getsize(html_path) / 1e6
    print("\n" + "=" * 70)
    print(f"  wrote {html_path}  ({html_mb:.2f} MB, audio ~{total_bytes/1e6:.2f} MB @ {args.bitrate})")
    print(f"  wrote {key_path}  ({len(keymap)} trials: {dict(cond_counts)})")
    print(f"  flip pairs: {len(pairs)}; catch: {len(catch)}; every shipped flip PASSED the salience self-check.")
    print("=" * 70)

    # KEYMAP SCHEMA (per item_id / TRIAL):
    #   {pair_id, intended_text, side_map['A'|'B' = which side is the CORRECT twin],
    #    tone_i2_frozen_correct, tone_i2_frozen_flipped, tone_i2_deployed_correct, tone_i2_deployed_flipped,
    #    condition['pair'|'catch'], is_catch, expect_pick['A'|'B' for catch else None], flip_dir, tbu_index,
    #    realized_st, delta_ok, band_ok, pred_flip_ok   (salience audit, pairs only; None on catch)}
    #   WITHHELD — never sent to the reviewer.


if __name__ == "__main__":
    main()
