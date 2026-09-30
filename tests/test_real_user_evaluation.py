"""The offline real-user evaluation harness: protocol, anonymity rule and output columns (stub embedders)."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from scipy.io import wavfile
from tests.voice_signals import speech_envelope


class _Stub:
    """Deterministic embedder: the embedding depends on the sample's content, so a participant's captures (drawn
    from their own base pattern + small noise) are close and different participants are unrelated."""

    is_mock = False

    def __init__(self, dim):
        self.dim = dim

    def _vec(self, x):
        x = np.asarray(x[0] if isinstance(x, tuple) else x, dtype=np.float64).ravel()[:4096]
        proj = np.random.default_rng(len(x)).standard_normal((len(x), self.dim))
        v = (x - x.mean()) @ proj
        return v / np.linalg.norm(v)

    def embed(self, raw):
        return self._vec(raw)

    def embed_poses(self, captures):
        return [self._vec(img) for _, img in captures], [{"pose": p, "status": "VALID"} for p, _ in captures]


def _write_participant(root, pid, seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (64, 64, 3)).astype(np.float64)
    tone = rng.uniform(150, 900)

    def img(path):
        cv2.imwrite(str(path), np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8))

    def wav(path):
        t = np.linspace(0, 2, 32000, endpoint=False)
        w = 0.3 * speech_envelope(t) * np.sin(2 * np.pi * tone * t) + rng.normal(0, 0.003, t.size)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, (w * 32767).astype(np.int16))
        path.write_bytes(buf.getvalue())

    e = root / pid / "enroll"
    e.mkdir(parents=True)
    for pose in ("front", "left", "right", "up", "down"):
        img(e / f"face_{pose}.png")
    wav(e / "voice_1.wav")
    wav(e / "voice_2.wav")
    for s in (1, 2):
        d = root / pid / f"session_{s}"
        d.mkdir()
        img(d / "face.png")
        wav(d / "voice.wav")


def _consent(root, answers, withdrawn=()):
    lines = ["participant_id,consent,withdrawn"] + [f"{p},{c},{'yes' if p in withdrawn else 'no'}" for p, c in answers.items()]
    (root / "consent.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_harness_builds_genuine_and_target_key_impostor_comparisons(tmp_path):
    from evaluation.real_user_evaluation import run

    root = tmp_path / "participants"
    for k, pid in enumerate(("P001", "P002", "P003")):
        _write_participant(root, pid, k)
    _consent(root, {"P001": "yes", "P002": "yes", "P003": "yes"})
    pipes = {"face": _Stub(512), "voice": _Stub(192)}
    rows, metrics = run(root, pipelines=pipes, accept_fair_voice=True, out_dir=tmp_path)
    valid = [r for r in rows if r.get("genuine") != ""]
    # 3 claimants x 2 sessions x 2 modalities x 3 targets
    assert len(valid) == 3 * 2 * 2 * 3
    assert sum(r["genuine"] for r in valid) == 3 * 2 * 2
    assert {"participant_id", "modality", "session", "genuine", "hamming_similarity", "estimated_cosine",
            "estimated_distance", "accepted", "template_set", "key_version"} <= set(valid[0])
    genuine_h = [r["hamming_similarity"] for r in valid if r["genuine"]]
    impostor_h = [r["hamming_similarity"] for r in valid if not r["genuine"]]
    assert min(genuine_h) > max(impostor_h)  # the stub separates perfectly
    assert {m["scope"] for m in metrics} >= {"face", "voice", "fusion_ALL_REQUIRED", "fusion_WEIGHTED"}
    assert (tmp_path / "real_user_scores.csv").exists() and (tmp_path / "real_user_metrics.csv").exists()


def test_harness_refuses_non_anonymous_participant_folders(tmp_path):
    from evaluation.real_user_evaluation import discover

    (tmp_path / "Firstname Lastname").mkdir()
    with pytest.raises(ValueError, match="anonymous"):
        discover(tmp_path)


def test_harness_requires_a_consent_record(tmp_path):
    from evaluation.real_user_evaluation import discover

    _write_participant(tmp_path, "P001", 0)
    with pytest.raises(ValueError, match="consent"):
        discover(tmp_path)


def test_participants_without_consent_or_withdrawn_are_never_read(tmp_path):
    from evaluation.real_user_evaluation import discover

    for k, pid in enumerate(("P001", "P002", "P003", "P004")):
        _write_participant(tmp_path, pid, k)
    _consent(tmp_path, {"P001": "yes", "P002": "no", "P003": "yes"}, withdrawn=("P003",))
    assert sorted(discover(tmp_path)) == ["P001"]  # P002 refused, P003 withdrawn, P004 has no record


def test_outputs_hold_scores_only_and_withdrawal_removes_a_participant(tmp_path):
    import csv

    from evaluation.real_user_evaluation import run, withdraw

    root = tmp_path / "participants"
    for k, pid in enumerate(("P001", "P002", "P003")):
        _write_participant(root, pid, k)
    _consent(root, {"P001": "yes", "P002": "yes", "P003": "yes"})
    run(root, pipelines={"face": _Stub(512), "voice": _Stub(192)}, accept_fair_voice=True, out_dir=tmp_path)
    header = (tmp_path / "real_user_scores.csv").read_text(encoding="utf-8").splitlines()[0].lower()
    for forbidden in ("embedding", "image", "audio", "path", "template_bits", "name"):
        assert forbidden not in header
    check = list(csv.DictReader((tmp_path / "real_user_protocol_check.csv").open(encoding="utf-8")))
    assert {r["meets_protocol"] for r in check} == {"False"}  # 2 sessions x 1 attempt < 10 genuine attempts

    removed = withdraw("P002", root, out_dir=tmp_path, delete_captures=True)
    assert removed["real_user_scores.csv"] > 0 and removed["capture_folder_deleted"] == 1
    assert not (root / "P002").exists()
    text = (tmp_path / "real_user_scores.csv").read_text(encoding="utf-8")
    assert "P002" not in text and "P001" in text
