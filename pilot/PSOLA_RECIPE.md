# Yorùbá Tone-Metric Human-Validation Kit — Build-Ready Spec (one-shot, must work first try)

> Derived from research workflow wu1wzq780 (tone-manipulation methods + Yorùbá tone phonetics
> + human-eval design + repo root-cause). This is the authoritative spec for the A/B rebuild.

## 1. Root cause (proven)

The last flips were inaudible because **the manipulation window is a single-grapheme MMS-CTC `merge_tokens` spike (~20–60 ms), not the voiced syllable rhyme.** `tone_eval_v2.forced_tbu_windows` maps each TBU to ONE flat-char position; `AF.merge_tokens` returns the CTC firing span for that one token — a sliver of the ~80–150 ms vowel. `oracle.psola_shift_window` multiplies F0 only inside `[t0,t1)`, touching a handful of pitch pulses while the surrounding original contour dominates the percept. **Proof:** 6 of 14 flip pairs have bit-identical deployed `tone_i2` between correct and flipped twin even at large shifts (e.g. pair_bible_07190 0.7667/0.7667 at −9.89 st). Downward H→L flips are equally unmoved → `f0max=400` clipping exonerated; 4–6.5 st magnitudes are past the +2/−3 st category threshold → magnitude fine. **The fix: expand the flip window from the CTC spike to the whole voiced rhyme, and re-target the entire rhyme's contour.**

## 2. Stimulus recipe (exact)

**Segment = whole voiced rhyme, derived from F0 not the CTC span.** Use `wins[i]` only as a seed center, then grow the real window from `pre["f0"]`/`pre["times"]`:
1. Seed = midpoint of `wins[i]=(t0,t1)`.
2. Grow left/right over contiguous voiced frames (`pre["f0"]` not-NaN) until F0 goes unvoiced/silent each side.
3. Clamp to inter-onset interval — never cross into `wins[i±1]` (no bleed into neighbours).
4. Require ≥ **60 ms** voiced; else reject the TBU. Bounded by unvoiced/silence → no audible edge step.

**Re-target the contour (do NOT blind-multiply a sliver).**
- From the clean clip: `_blind_residuals(v2._tbu_semitones(pre))` + orthographic `tones`. `medH=median(res|H)`, `medL=median(res|L)`, `mid_ref=median(all res)`.
- L→H: target residual `r* = medH`. H→L: `r* = medL`. Add a gentle natural slope (H: slight fall ~−1 st; L: near-flat) in the speaker's own sign.
- Per-frame absolute: `st_abs(t)=r*+slope_clip·t`; `f0_hz(t)=100·2^(st_abs/12)` (100 Hz ref matches `_tbu_semitones`). Lay a DENSE pitch-tier (point ~every 10 ms across the rhyme), PSOLA-resynthesize. Lands the syllable squarely in the opposite tone band — strictly better than relative multiply.

**Magnitude:** realized move ≈ the clip's own H–L span (`measure_delta_HL`, ~4–8 st) at the correct register. **Raise `f0max` to 600 Hz on every path.**

**Artifact control — CORRECT twin processed IDENTICALLY.** The correct twin must NOT be `psola_roundtrip` while flipped is a dense-pitch-tier reset (that asymmetry is a confound). Re-impose the clip's OWN measured contour over the SAME rhyme using the SAME dense-pitch-tier-replace + overlap-add. Both twins carry an identical artifact; only the residual values inside the rhyme differ.

**HARD salience self-check the BUILD MUST enforce.** After building `flipped_wav`, re-extract F0 over the rhyme and assert ALL:
1. `|medianF0(flipped) − medianF0(correct)|` over rhyme voiced frames ≥ **3.0 st realized**;
2. flipped residual lands in the opposite band under frozen-clean mid_ref: L→H ⇒ `r ≥ mid_ref+theta_h`; H→L ⇒ `r ≤ mid_ref−theta_l`;
3. frozen-mid_ref oracle `pred[i]` flips to the opposite class.
Any failure → **reject the clip, try the next.** No failing clip enters keymap or HTML.

**PSOLA vs WORLD:** parselmouth PSOLA is adequate once window=full rhyme, target=contour reset, ceiling=600, and F0 re-extraction validates. Keep parselmouth. Only if the self-check keeps failing → fall back to `pyworld` (Harvest→CheapTrick→D4C, replace f0 on voiced frames).

## 3. Human task — A/B forced choice

Trial = one sentence, two clips of the SAME utterance/voice/artifact: correct twin + flipped twin.
- **UI:** players A and B; **randomize which side is correct per trial** (store in keymap `side_map`). Question: **"Which one sounds like correct Yorùbá — A or B?"**; buttons **A / B / — not sure**; unlimited replays.
- **Trials:** **≥24 real A/B pairs** (`--n-clips` 14→24) + **4–5 catch** (~15–20%). ~15–20 min.
- **Catch:** one side natural clean recording, other side a huge-flip or `psola_flatten` of the same sentence (obvious). Keep rater only if **catch ≥ 80%**.
- **Keystone statistic:** **% of real trials picking the correct twin**, Wilson 95% CI, vs **50%** binomial baseline. Tie to metric: metric "picks" correct iff `tone_i2_frozen(correct) > tone_i2_frozen(flipped)` → metric paired-win-rate; plus **AUROC/point-biserial of margin `Δ=tone_i2_frozen(correct)−tone_i2_frozen(flipped)` vs human-correct (0/1)**, bootstrap CIs.
- **Floor:** item is the unit; 1–2 raters suffice. `--min-scoreable 24`. Repeat ~15% of pairs for intra-rater reliability.

## 4. Metric scoring fix (frozen mid_ref + deployed caveat)

Score twins TWICE, store both per item:
- **`tone_i2_frozen`** (mid_ref FROZEN from clean clip's I2 `mid_ref` field): the metric CORE sensitivity — **the column that answers "does the metric agree with humans."** All paired-win / AUROC / point-biserial use this.
- **`tone_i2_deployed`** (`mid_ref=None`, as shipped nb07/nb14/poster): reported as the honesty caveat (per-utterance anchor's localized blindness). Reported, never used for the agreement question.

Print: "Metric–human agreement is on `tone_i2_frozen`; `tone_i2_deployed` documents the deployment-time localized-blindness gap."

## 5. Mandatory ear-check gate (nb16 blocks export)

1. Cell 11 augmented: per shipped pair, print the automated salience table (realized ΔF0 st, opposite-band PASS/FAIL, pred[i]-flip PASS/FAIL) beside inline `▶ correct`/`▶ flipped` players. Set `ALL_SALIENT = all passed`.
2. New gate cell before §6: require `CONFIRM_FLIPS_AUDIBLE=True` after listening; `assert ALL_SALIENT and CONFIRM_FLIPS_AUDIBLE`.
3. §6 download cell prepends the same assert → `psola_form.html` cannot be saved/sent unless automated self-check passed for every pair AND user confirmed by ear.

## 6. Fallback (if manipulation still not unmistakable)

Real tonal minimal-pair identification via `tone_metric/minimal_pairs_draft.json`: native flips `verified:false→true` + expands to ~30–40 sets (add Carter-Ényì attested contrasts: ara, aro, bata, ishe, joko, mimọ, ogun, ori, pipa, sisun; carrier "Sọ ___ sọke"); natural recordings of both tone variants; closed-set 2-AFC "which word did you hear?"; freeze to S3 `tts_data/yoruba/eval/minimal_pairs.v1.json` holdout. No PSOLA dependency.

## 7. Build checklist

**`tone_metric/tone_oracle.py`** — add `voiced_rhyme_window(pre,i,min_ms=60)`, `psola_set_contour(wav,t0,t1,pts,sr,f0min=65,f0max=600)` (dense pitch-tier reset, used for BOTH twins), `tone_level_residuals(pre)→(medH,medL,mid_ref,slope)`, `build_flip_contour(...)`, `reimpose_contour(...)` (original values); raise `F0MAX=600`.

**`pilot/build_psola_form.py`** — `process_clip`: `t0,t1 = oracle.voiced_rhyme_window(pre,i)` (skip if None); `flipped_wav=psola_set_contour(reset to opposite band)`, `correct_wav=reimpose_contour(same rhyme, original)`. Freeze `mid_ref_clean`; score both twins twice (`tone_i2_frozen`, `tone_i2_deployed`); `score_full` gains a `mid_ref` arg. Enforce the §2 hard self-check, `return None` on any failure. `--n-clips` 14→24. Emit A/B paired HTML; store `side_map`, both tone_i2 in keymap.

**`pilot/build_pilot_form.py`** — add `render_ab_html` (two players A/B per trial, A/B/— buttons, randomized side; copy/CSV/localStorage unchanged).

**`pilot/score_psola.py`** — parse A/B picks, map `side_map`→correct/incorrect; keystone human %-correct vs 50% Wilson; metric paired-win + AUROC/point-biserial on `tone_i2_frozen`; print deployed paired-win as caveat; catch ≥80% gate; `--min-scoreable 24`.

**`notebooks/16_psola_tone_form.ipynb`** — Cell 11 salience table + players + `ALL_SALIENT`; new gate cell (`CONFIRM_FLIPS_AUDIBLE` + assert); §6 download prepend assert; §4 `--n-clips 24`. Update `3_REVIEWER_instructions_psola.md` to A/B wording.
