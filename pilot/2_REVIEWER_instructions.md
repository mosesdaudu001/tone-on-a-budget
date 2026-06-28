# Reviewer instructions — send WITH the listening form (after they say yes)

> Goal: everything the reviewer needs, in plain language. The precise "tap this button"
> steps also live INSIDE the form itself, so this message is just friendly framing +
> how to send answers back. Paste this as the message body when you share the form.

---

Thank you so much for helping! 🙏 Here's everything you need — it takes about **20 minutes**.

**What this is:** a short listening check. You'll hear ~25 short Yorùbá clips. For each clip, the **written word or sentence is shown on screen**. Your only job is to tell me whether the **tone (ohùn)** of the voice matches the written word.

**How to do it:**
1. Open the file/link I sent — it works in any phone browser. **Headphones help but a phone speaker is fine.**
2. For each clip, tap **▶** to listen. You can replay as many times as you like.
3. Tap **one** button:
   - **✓ Tone is correct** — the melody/tone matches the written word
   - **✗ Tone is wrong** — the tone sounds off for that word
   - **— Not sure**
4. Some clips will sound clearly fine, some may sound off, and a few may sound almost identical — **that's all expected.** Just give your honest best judgment.
5. At the very end, tap **"Copy my answers"** and **paste them back to me here in this chat.**

That's the whole thing. There are genuinely **no wrong answers** — I'm testing the AI voices and a measurement tool, not you. If anything is confusing, just tell me. Ẹ ṣé púpọ̀! 🙏

---

### Notes for YOU (Moses) — not for the reviewer

- The form is a **single self-contained `.html` file** (audio embedded) — it works offline once opened, so WhatsApp/email both work. No links can break.
- If two reviewers are available, send the **same file to both separately** — a second rater gives you an inter-rater agreement number too.
- When they paste answers back, save the text and run `score_pilot.py` (or send the pasted block to me) → it prints **`AUROC(tone_i2 vs human)`** (primary, threshold-free), human accuracy on the native anchor clips, the catch screen, Wilson CIs, and a conservative κ (secondary).
- **Pre-registered read (corrected for a spec error — the original used a faulty `OR`):** anchor ✓-acc ≥ 75% **AND** catch ≥ 5/6 are a **rater-validity GATE** (is the listener engaged?), *not* evidence about the metric. Metric validity = **AUROC ≥ 0.70 alone**, *after* the gate passes. Below that → honest inconclusive/negative. (AUROC, not κ, is the headline.)
- **Known limitation of this design:** the ✓/✗-on-TTS task conflates *tone* with *synthetic naturalness* — a native may mark a tonally-correct-but-robotic clip "wrong." A clean metric-validity test isolates tone (PSOLA H↔L flip on real native clips, naturalness held constant) with ≥2 raters. Treat a model-clip result here as confounded.
- If a reviewer marks the real **native** anchor clips as ✗, or fails the catch trials (< 5 of 6), discard that rater as not-engaged rather than counting noise.
