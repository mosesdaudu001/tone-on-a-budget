# coding=utf-8
# Tier-0 WIRING TEST for L2 language-ID conditioning — pure CPU, seconds, NO model download.
#
# The #1 risk in L2 is the +1 preamble shift being wrong (this project has shipped off-by-one
# preamble / double-shift bugs before). This test builds dataset_b's collate output for both Auto
# and L2 and asserts the EXACT token layout against the golden derived by reading modeling
# generate() (lines ~2135-2147 of modeling_qwen3_tts.py):
#
#   Auto preamble (codec):  [nothink, think_bos, think_eos, SPEAKER, pad, (bos later)]
#   L2   preamble (codec):  [think,   think_bos, LANG_ID,    think_eos, SPEAKER, pad, (bos later)]
#
# Run in the training env (needs qwen_tts importable for dataset_b):  python test_l2_wiring.py
import types
import torch
from dataset_b import TTSDatasetB

# real Qwen3-TTS-12Hz-1.7B-Base ids (from config.json — see lang_id.py)
TC = types.SimpleNamespace(
    codec_nothink_id=2155, codec_think_id=2154, codec_think_bos_id=2156, codec_think_eos_id=2157,
    codec_pad_id=2148, codec_bos_id=2149, codec_eos_token_id=2150,
)
CFG = types.SimpleNamespace(
    tts_pad_token_id=151671, tts_bos_token_id=151672, tts_eos_token_id=151673, talker_config=TC,
)
LANG_ID = 2072  # Yoruba

T, C = 8, 10                       # text_ids length (role 3 + content 5), codec frames
ROLE = [900, 901, 902]
CONTENT = [10, 11, 12, 13, 14]     # text_ids[3:] = the 5 content tokens


def make_batch():
    text_ids = torch.tensor([ROLE + CONTENT], dtype=torch.long)         # [1, T]
    audio = (torch.arange(C * 16).reshape(C, 16) % 2000) + 1            # [C, 16], all valid audio ids
    return [{"text_ids": text_ids, "audio_codes": audio, "ref_mel": torch.zeros(1, 5, 128),
             "instruct_ids": None}]


def collate(language_id):
    ds = TTSDatasetB([], processor=None, config=CFG, language_id=language_id)
    return ds.collate_fn(make_batch())


def check(mode, language_id, PRE, spk_pos, base, preamble_codec):
    out = collate(language_id)
    text = out["input_ids"][0, :, 0]
    codec = out["input_ids"][0, :, 1]
    cem = out["codec_embedding_mask"][0, :, 0]
    bos_t = base - 1
    errs = []

    def eq(name, got, exp):
        if list(map(int, got)) != list(map(int, exp)):
            errs.append(f"{name}: got {list(map(int,got))} != expected {list(map(int,exp))}")

    # 1) spk_pos reported by collate (now a per-row [B] tensor)
    if int(out["spk_pos"][0]) != spk_pos:
        errs.append(f"spk_pos: got {int(out['spk_pos'][0])} != {spk_pos}")
    # 2) codec preamble block (positions 3 .. 3+PRE-1) — the SPEAKER slot is written 0
    eq("codec preamble [3:3+PRE]", codec[3:3 + PRE], preamble_codec)
    # 3) speaker slot is masked OUT of codec embedding (filled by speaker_encoder in build_inputs_embeds)
    if bool(cem[spk_pos]) is not False:
        errs.append(f"codec_embedding_mask[{spk_pos}] should be False (speaker slot)")
    if not all(bool(cem[p]) for p in range(3, 3 + PRE) if p != spk_pos):
        errs.append("non-speaker preamble positions should be codec-embedded (mask True)")
    # 4) text channel: role(0:3), pads(3:bos_t), tts_bos@bos_t, content@base.., tts_eos, then pad
    eq("text role [0:3]", text[0:3], ROLE)
    eq("text pads [3:bos_t]", text[3:bos_t], [CFG.tts_pad_token_id] * (bos_t - 3))
    eq("tts_bos @ bos_t", [text[bos_t]], [CFG.tts_bos_token_id])
    eq("text content @ base", text[base:base + T - 3], CONTENT)
    eq("tts_eos", [text[base + T - 3]], [CFG.tts_eos_token_id])
    # 5) codec: pad during text block, codec_bos right before audio, audio cb0, codec_eos
    eq("codec_bos before audio", [codec[base + T - 2]], [TC.codec_bos_id])
    eq("codec audio cb0", codec[base + T - 1:base + T - 1 + C], (make_batch()[0]["audio_codes"][:, 0]))
    eq("codec_eos", [codec[base + T - 1 + C]], [TC.codec_eos_token_id])
    # 6) labels align with cb0 audio + eos; everything else -100
    lab = out["codec_0_labels"][0]
    eq("label audio", lab[base + T - 1:base + T - 1 + C], (make_batch()[0]["audio_codes"][:, 0]))
    eq("label eos", [lab[base + T - 1 + C]], [TC.codec_eos_token_id])
    if int((lab[:base + T - 1] != -100).sum()) != 0:
        errs.append("labels before audio should all be -100 (preamble/text masked)")

    status = "PASS" if not errs else "FAIL"
    print(f"[{status}] {mode}: PRE={PRE} spk_pos={spk_pos} base={base} "
          f"preamble={list(map(int, codec[3:3+PRE]))}")
    for e in errs:
        print("        -", e)
    return not errs


def main():
    ok = True
    # AUTO — must be byte-identical to the original hardcoded layout (regression guard)
    ok &= check("AUTO", None, PRE=5, spk_pos=6, base=8,
                preamble_codec=[TC.codec_nothink_id, TC.codec_think_bos_id, TC.codec_think_eos_id,
                                0, TC.codec_pad_id])
    # L2 — registered-language preamble; speaker shifts 6->7, base 8->9 (clean +1)
    ok &= check("L2(yoruba)", LANG_ID, PRE=6, spk_pos=7, base=9,
                preamble_codec=[TC.codec_think_id, TC.codec_think_bos_id, LANG_ID,
                                TC.codec_think_eos_id, 0, TC.codec_pad_id])

    # ROW-TAG ROUTING — the language:"Auto" bug class. The GPU train_b assert only catches the EXTREME
    # (0 rows tagged); this catches the routing itself. A row stored as "Auto" (what nb11 wrote) must NOT
    # receive the L2 id when target=yoruba; a "yoruba" row must; single-language mode tags ALL rows.
    rds = TTSDatasetB([], processor=None, config=CFG, language_id=LANG_ID, language="yoruba")
    route_ok = (rds._row_language_id("Auto") is None and rds._row_language_id("yoruba") == LANG_ID
                and rds._row_language_id("english") is None)
    single = TTSDatasetB([], processor=None, config=CFG, language_id=LANG_ID, language=None)
    route_ok &= (single._row_language_id("Auto") == LANG_ID and single._row_language_id("yoruba") == LANG_ID)
    none_ds = TTSDatasetB([], processor=None, config=CFG, language_id=None)
    route_ok &= (none_ds._row_language_id("yoruba") is None)
    print(f"[{'PASS' if route_ok else 'FAIL'}] row-tag routing: Auto->None / yoruba->id (per-row); "
          f"language=None->all rows id; language_id=None->None")
    ok &= route_ok

    # cross-check the insertion: L2 inserts LANG_ID at preamble index 2 (position 5), so
    #   AUTO[3]=nothink -> L2[3]=think (tag differs); AUTO[4]=think_bos == L2[4] (UNshifted);
    #   AUTO[5:8] (think_eos/spk/pad) == L2[6:9] (shifted +1).
    a = collate(None)["input_ids"][0, :, 1]
    l = collate(LANG_ID)["input_ids"][0, :, 1]
    shift_ok = (int(a[4]) == int(l[4])) and all(int(a[k]) == int(l[k + 1]) for k in range(5, 8))
    print(f"[{'PASS' if shift_ok else 'FAIL'}] insertion: AUTO[4]==L2[4] (think_bos) and "
          f"AUTO[5:8]==L2[6:9] (+1 shift of think_eos/spk/pad)")
    ok &= shift_ok

    # L3 MIXED batch: one yoruba row (-> L2 preamble) + one english row (-> Auto), SAME batch.
    dsm = TTSDatasetB([], processor=None, config=CFG, language_id=LANG_ID, language="yoruba")
    def _row(lang, C):
        return {"text_ids": torch.tensor([ROLE + CONTENT], dtype=torch.long),
                "audio_codes": (torch.arange(C * 16).reshape(C, 16) % 2000) + 1,
                "ref_mel": torch.zeros(1, 5, 128), "instruct_ids": None, "language": lang}
    mo = dsm.collate_fn([_row("yoruba", 14), _row("english", 8)])   # different lengths -> shared pad
    mc = mo["input_ids"][:, :, 1]
    mix_ok = (
        list(map(int, mc[0, 3:9])) == [TC.codec_think_id, TC.codec_think_bos_id, LANG_ID,
                                       TC.codec_think_eos_id, 0, TC.codec_pad_id]
        and list(map(int, mc[1, 3:8])) == [TC.codec_nothink_id, TC.codec_think_bos_id,
                                           TC.codec_think_eos_id, 0, TC.codec_pad_id]
        and int(mo["spk_pos"][0]) == 7 and int(mo["spk_pos"][1]) == 6
        and bool(mo["codec_embedding_mask"][0, 7, 0]) is False
        and bool(mo["codec_embedding_mask"][1, 6, 0]) is False)
    print(f"[{'PASS' if mix_ok else 'FAIL'}] MIXED batch: row0 yoruba->L2 (spk7/base9), "
          f"row1 english->Auto (spk6/base8), per-row preambles in one batch")
    ok &= mix_ok

    print("\n" + ("ALL WIRING CHECKS PASSED ✅ — preamble matches generate(); safe to forward-test on GPU."
                  if ok else "WIRING CHECKS FAILED ❌ — fix before any GPU run."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
