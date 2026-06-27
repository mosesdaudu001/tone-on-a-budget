# Unit tests for tone_probe (run: cd approachB && python test_tone_probe.py — CPU, no downloads)
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tone_eval_v2 as v2
import tone_probe as tp

DEV = "cpu"
RNG = np.random.default_rng(0)


def _seg_for(cls, t=10, dim=32):
    """Synthetic latent segment whose class is linearly embedded (dims 0-2) + noise."""
    s = RNG.normal(0, 0.3, (t, dim)).astype("float32")
    s[:, tp.CLASS_TO_ID[cls]] += 2.0
    return torch.from_numpy(s)


def test_weighted_f1():
    f1, per = tp.weighted_f1(["H", "H", "M", "L"], ["H", "M", "M", "L"])
    assert abs(per["M"] - 2 / 3) < 1e-9 and abs(per["H"] - 2 / 3) < 1e-9 and per["L"] == 1.0
    assert abs(f1 - (2 / 3 * 0.5 + 2 / 3 * 0.25 + 1.0 * 0.25)) < 1e-9
    print("PASS weighted_f1")


def test_tbu_segments():
    lat = torch.zeros(100, 8)                       # 100 frames = 2.0 s at 20 ms
    wins = [(0.10, 0.30), None, (0.50, 0.52), (1.0, 1.4)]
    segs = tp.tbu_segments(lat, wins, min_frames=2)
    assert segs[0].shape[0] == 10                   # 0.2 s -> 10 frames
    assert segs[1] is None and segs[2] is None      # abstain: missing window, too short (1 frame)
    assert segs[3].shape[0] == 20
    print("PASS tbu_segments (slicing + abstention)")


def test_train_predict_saveload():
    labels = [tp.CLASSES[i % 3] for i in range(900)]
    segs = [_seg_for(l, t=int(RNG.integers(4, 16))) for l in labels]
    dev_labels = [tp.CLASSES[i % 3] for i in range(150)]
    dev_segs = [_seg_for(l, t=8) for l in dev_labels]
    probe, f1, hist = tp.train_probe(segs, labels, dev_segs, dev_labels, DEV, in_dim=32,
                                     hidden=16, epochs=4, bs=64, log=lambda *_: None)
    assert f1 >= 0.95, f1
    p = "/tmp/_probe_test.pt"
    tp.save_probe(probe, p, dict(layer=9, note="test"))
    probe2, meta = tp.load_probe(p, DEV)
    a = tp.predict(probe, dev_segs, DEV)
    b = tp.predict(probe2, dev_segs, DEV)
    assert a == b and meta["layer"] == 9
    print(f"PASS train/predict/save/load (dev wF1 {f1:.3f})")
    return probe


class _FakeEncoderOut:
    def __init__(self, hs): self.hidden_states = hs


class _FakeEncoder:
    """Returns hidden states where each 20 ms frame's class signal follows a schedule of (t0,t1,cls)."""
    def __init__(self, schedule, dim=32, n_layers=13):
        self.schedule, self.dim, self.n_layers = schedule, dim, n_layers
    def __call__(self, iv, output_hidden_states=True):
        n16 = iv.shape[-1]
        T = n16 // 320
        h = torch.from_numpy(RNG.normal(0, 0.3, (1, T, self.dim)).astype("float32"))
        for t0, t1, cls in self.schedule:
            i0, i1 = int(t0 / tp.FRAME_SEC), int(math.ceil(t1 / tp.FRAME_SEC))
            h[0, i0:i1, tp.CLASS_TO_ID[cls]] += 2.0
        return _FakeEncoderOut([h] * self.n_layers)


class _FakeFE:
    def __call__(self, w, sampling_rate, return_tensors):
        class _R:  pass
        r = _R(); r.input_values = torch.from_numpy(np.asarray(w, dtype="float32"))[None, :]
        return r


def test_probe_score_end_to_end():
    probe = test_train_predict_saveload.__wrapped__() if hasattr(test_train_predict_saveload, "__wrapped__") \
        else _TRAINED
    text = "bá ba bà bá"                            # H M L H
    tones = v2.tbu_seq(text)
    assert tones == ["H", "M", "L", "H"]
    wins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    enc = _FakeEncoder([(w[0], w[1], t) for w, t in zip(wins, tones)])
    fe = _FakeFE()
    orig = v2.forced_tbu_windows
    v2.forced_tbu_windows = lambda *a, **k: list(wins)
    try:
        wav = np.zeros(int(0.8 * 16000), dtype="float32")
        r = tp.probe_score(wav, 16000, text, probe, enc, fe, device=DEV, layer=9)
        assert r["method"] == "probe" and r["coverage"] == 1.0
        assert r["accuracy"] == 1.0, (r["accuracy"], r["pred"], r["target"])
        # abstention path: kill one window
        v2.forced_tbu_windows = lambda *a, **k: [wins[0], None, wins[2], wins[3]]
        r2 = tp.probe_score(wav, 16000, text, probe, enc, fe, device=DEV, layer=9)
        assert r2["coverage"] == 0.75 and r2["pred"][1] is None and r2["accuracy"] == 1.0
        # alignment impossible -> probe-abstain, NaN accuracy
        v2.forced_tbu_windows = lambda *a, **k: None
        r3 = tp.probe_score(wav, 16000, text, probe, enc, fe, device=DEV, layer=9)
        assert r3["method"] == "probe-abstain" and math.isnan(r3["accuracy"])
    finally:
        v2.forced_tbu_windows = orig
    print("PASS probe_score end-to-end (accuracy, abstention, align-fail)")


if __name__ == "__main__":
    test_weighted_f1()
    test_tbu_segments()
    _TRAINED = test_train_predict_saveload()
    test_probe_score_end_to_end()
    print("\nALL TESTS PASS")
