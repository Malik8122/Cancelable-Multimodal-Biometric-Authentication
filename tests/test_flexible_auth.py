"""Flexible multimodal authentication (V3): the USER chooses what to enroll and what to present.

Buildings are authentication context only - no biometric policy. Any non-empty subset of face / voice can
be enrolled, and any subset of the enrolled modalities can be presented; one fusion engine handles every combination,
whatever the building. A submitted modality that is not enrolled is ENROLLMENT_REQUIRED (HTTP 409), not a denial.

Face detection (MTCNN) cannot run on synthetic images, so - for this file only - the modality embedders are
replaced by deterministic stubs (same sample -> same embedding, different sample -> unrelated embedding). Everything
else is the real code: services, template sets, fusion, database, audit, HTTP layer.
"""

from __future__ import annotations

import functools
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from tests.voice_signals import speech_envelope

APPLICATION_ID = "capstone-demo"

# Buildings are only a label for the session; every outcome below is identical whichever one is named.
BUILDING = "defence_research_lab"
OTHER_BUILDING = "administration_block"


# ----------------------------------------------------------------------------- samples


def _png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return buffer.tobytes()


def _noise_png(seed: int) -> bytes:
    return _png(np.random.default_rng(seed).integers(0, 256, size=(300, 300, 3), dtype=np.uint8))


def _tone_wav(frequency: float) -> bytes:
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    buffer = io.BytesIO()
    wavfile.write(buffer, 16000, (0.3 * speech_envelope(t) * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16))
    return buffer.getvalue()


FACE, OTHER_FACE = _noise_png(1), _noise_png(2)
VOICE, OTHER_VOICE = _tone_wav(220), _tone_wav(880)


# ----------------------------------------------------------------------------- stubbed embedders


class _StubPipeline:
    is_mock = False

    def __init__(self, dim: int):
        self.dim = dim

    def check_capture(self, raw):
        """Stand-in for FacePipeline.check_capture: no contrast = no face, low contrast = blurry, else valid."""
        spread = float(np.asarray(raw, dtype=np.float64).std())
        return "NO_FACE" if spread <= 1.0 else "BLURRY" if spread < 12.0 else "VALID"

    def embed_poses(self, captures):
        """Stand-in for FacePipeline.embed_poses: one embedding per valid capture + the per-pose report."""
        embeddings, report = [], []
        for pose, raw in captures:
            verdict = self.check_capture(raw)
            report.append({"pose": pose, "status": verdict})
            if verdict == "VALID":
                embeddings.append(self.embed(raw))
        return embeddings, report

    def embed(self, raw) -> np.ndarray:
        if isinstance(raw, tuple):  # voice: (waveform, sample_rate)
            spectrum = np.log1p(np.abs(np.fft.rfft(np.asarray(raw[0], dtype=np.float64)[:8192], n=8192)))[:512]
            features = spectrum - spectrum.mean()
        else:  # face image
            features = np.asarray(raw, dtype=np.float64)[::37, ::37].ravel() - 127.5
        projection = np.random.default_rng(len(features)).standard_normal((len(features), self.dim))
        vector = features @ projection
        return vector / np.linalg.norm(vector)


def _stub_getter(modality: str, dim: int):
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    @functools.lru_cache
    def getter():
        return ModalityService(modality, _StubPipeline(dim), get_settings())

    return getter


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    import backend.services.face_service as face_service
    import backend.services.voice_service as voice_service

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'bp.db'}")
    _reset_caches()
    monkeypatch.setattr(face_service, "get_face_service", _stub_getter("face", 512))
    monkeypatch.setattr(voice_service, "get_voice_service", _stub_getter("voice", 192))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
    _reset_caches()


def _db():
    from backend.database.session import get_session_factory

    return get_session_factory()()


# ----------------------------------------------------------------------------- helpers


def _enroll(client, user, modality, sample):
    filename = "s.wav" if modality == "voice" else "s.png"
    content_type = "audio/wav" if modality == "voice" else "image/png"
    response = client.post(
        "/enroll",
        data={"user_id": user, "modality": modality, "application_id": APPLICATION_ID},
        files={"image": (filename, sample, content_type)},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _enroll_user(client, user, **samples):
    """_enroll_user(client, "u", face=FACE, voice=VOICE)"""
    for modality, sample in samples.items():
        _enroll(client, user, modality, sample)


def _fusion_files(face=None, voice=None):
    files = {}
    if face is not None:
        files["face_image"] = ("f.png", face, "image/png")
    if voice is not None:
        files["voice_audio"] = ("v.wav", voice, "audio/wav")
    return files


def _authenticate(client, user, building, face=None, voice=None, **extra):
    return client.post(
        "/authenticate/fusion",
        data={"user_id": user, "building_id": building, "application_id": APPLICATION_ID, **extra},
        files=_fusion_files(face, voice),
    )


def _audit(client, user):
    return client.get(f"/audit/{user}").json()["entries"]




# ----------------------------------------------------------------------------- the three combinations

COMBOS = [
    ("face",),
    ("voice",),
    ("face", "voice"),
]
SAMPLE = {"face": FACE, "voice": VOICE}
WRONG = {"face": OTHER_FACE, "voice": OTHER_VOICE}
IDS = ["+".join(c) for c in COMBOS]


def _submit(client, user, modalities, wrong=(), building=BUILDING, **extra):
    samples = {m: (WRONG[m] if m in wrong else SAMPLE[m]) for m in modalities}
    return _authenticate(client, user, building, **samples, **extra)


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_each_combination_enrolls_and_authenticates_independently(client, combo):
    """Enroll exactly `combo`, present exactly `combo`: granted by the one shared fusion engine."""
    _enroll_user(client, "u", **{m: SAMPLE[m] for m in combo})
    body = _submit(client, "u", combo).json()
    assert body["authentication_state"] == body["status"] == "ACCESS_GRANTED" and body["authenticated"] is True
    assert body["modalities_used"] == sorted(combo) and body["matched_modalities"] == sorted(combo)
    assert body["fusion_policy"] == "ALL_REQUIRED"
    assert body["fusion_similarity"] == pytest.approx(1.0) and body["fusion_distance"] == pytest.approx(0.0)
    assert body["active_template_set"] == body["template_set_version"] == 1 and body["key_version"] == 1
    assert set(body["results"]) == set(combo)  # (debug view) exactly the presented modalities were evaluated
    # the fused value is the mean over exactly those modalities
    assert body["fusion_similarity"] == pytest.approx(sum(r["score"] for r in body["results"].values()) / len(combo))


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_each_combination_denies_when_one_presented_modality_is_wrong(client, combo):
    _enroll_user(client, "u", **{m: SAMPLE[m] for m in combo})
    for wrong in combo:
        body = _submit(client, "u", combo, wrong=(wrong,)).json()
        assert body["authentication_state"] == "ACCESS_DENIED" and body["authenticated"] is False, (combo, wrong)
        assert body["matched_modalities"] == sorted(set(combo) - {wrong})
        assert body["modalities_used"] == sorted(combo)  # a real verification happened and failed


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_a_fully_enrolled_user_may_present_any_subset(client, combo):
    """Both enrolled; the user picks `combo` for this session: only those are evaluated and fused."""
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    body = _submit(client, "u", combo).json()
    assert body["authentication_state"] == "ACCESS_GRANTED"
    assert set(body["results"]) == set(combo) and body["modalities_used"] == sorted(combo)
    (entry,) = _audit(client, "u")
    assert entry["modality_list"] == sorted(combo) and entry["submitted_modalities"] == sorted(combo)
    assert entry["enrolled_modalities"] == ["face", "voice"]
    assert entry["authenticated_modalities"] == sorted(combo) and entry["authentication_state"] == "ACCESS_GRANTED"


@pytest.mark.parametrize("combo", COMBOS[:-1], ids=IDS[:-1])
def test_presenting_a_modality_that_is_not_enrolled_is_enrollment_required(client, combo):
    """Enrolled = `combo`; the user also presents one modality they never enrolled -> 409, nothing evaluated."""
    _enroll_user(client, "u", **{m: SAMPLE[m] for m in combo})
    absent = [m for m in ("face", "voice") if m not in combo]
    response = _submit(client, "u", (*combo, *absent))
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == body["authentication_state"] == "ENROLLMENT_REQUIRED"
    assert body["missing_modalities"] == absent and body["enrolled_modalities"] == sorted(combo)
    assert body["submitted_modalities"] == sorted((*combo, *absent))
    assert "authenticated" not in body and "fusion_similarity" not in body and "results" not in body
    (entry,) = _audit(client, "u")
    assert entry["authentication_state"] == "ENROLLMENT_REQUIRED" and entry["modality_list"] == []
    assert entry["submitted_modalities"] == sorted((*combo, *absent)) and entry["template_set_version"] is None
    # the user then simply presents what they DO have
    assert _submit(client, "u", combo).json()["authentication_state"] == "ACCESS_GRANTED"


def test_a_user_with_nothing_enrolled_is_enrollment_required_for_whatever_they_present(client):
    for combo in COMBOS:
        response = _submit(client, "nobody", combo)
        assert response.status_code == 409 and response.json()["missing_modalities"] == sorted(combo)


def test_at_least_one_factor_must_be_selected(client):
    response = _authenticate(client, "u", BUILDING)
    assert response.status_code == 422 and "Select at least one biometric factor" in response.json()["detail"]


def test_single_modality_endpoints_share_the_same_rules(client):
    _enroll_user(client, "u", face=FACE)
    ok = client.post("/verify/face", data={"user_id": "u", "application_id": APPLICATION_ID}, files={"image": ("f.png", FACE, "image/png")})
    assert ok.status_code == 200 and ok.json()["authentication_state"] == "ACCESS_GRANTED"
    missing = client.post("/verify/voice", data={"user_id": "u", "application_id": APPLICATION_ID}, files={"image": ("v.wav", VOICE, "audio/wav")})
    assert missing.status_code == 409 and missing.json()["missing_modalities"] == ["voice"]


# ----------------------------------------------------------------------------- buildings: context only


def test_no_building_specific_modality_rules_remain(client):
    """Same enrollment + same presented modalities -> the same decision for every building, and with none."""
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    buildings = [b["id"] for b in client.get("/buildings").json()]
    assert len(buildings) >= 2
    decisions = set()
    for building in buildings:
        body = _submit(client, "u", ("face", "voice"), building=building).json()
        decisions.add((body["authentication_state"], round(body["fusion_similarity"], 6), tuple(body["modalities_used"])))
        assert body["building_id"] == building  # a label, nothing more
        assert _submit(client, "u", ("face",), building=building).json()["modalities_used"] == ["face"]  # any subset, any building
    assert len(decisions) == 1
    no_building = client.post(
        "/authenticate/fusion", data={"user_id": "u", "application_id": APPLICATION_ID}, files=_fusion_files(FACE, VOICE)
    ).json()
    assert no_building["authentication_state"] == "ACCESS_GRANTED" and "building_id" not in no_building


def test_buildings_carry_only_context(client):
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "buildings.json").read_text(encoding="utf-8"))
    assert all(set(b) <= {"id", "name", "description", "clearance_level"} for b in config)
    listed = client.get("/buildings").json()
    assert [b["id"] for b in listed] == [b["id"] for b in config]
    assert all(set(b) == {"id", "name", "description", "clearance_level"} for b in listed)
    one = client.get(f"/building/{BUILDING}").json()
    assert set(one) == {"id", "name", "description", "clearance_level"} and one["clearance_level"]
    assert client.get("/building/does_not_exist").status_code == 404
    assert client.get(f"/building/{BUILDING}/readiness", params={"user_id": "u"}).status_code == 404  # readiness policy is gone


def test_a_building_config_that_defines_a_biometric_policy_is_rejected():
    from backend.buildings import parse_buildings

    assert parse_buildings([{"id": "a", "name": "A", "clearance_level": "III", "description": ""}])[0].id == "a"
    for bad in (
        [],
        [{"name": "no id"}],
        [{"id": "a"}, {"id": "a"}],
        [{"id": "a", "required_modalities": ["face"]}],  # buildings must NOT carry a modality policy
    ):
        with pytest.raises(ValueError):
            parse_buildings(bad)


def test_an_unknown_building_label_is_a_404_and_a_custom_config_can_be_configured(tmp_path, monkeypatch, client):
    _enroll_user(client, "u", face=FACE)
    assert _authenticate(client, "u", "no_such_building", face=FACE).status_code == 404
    path = tmp_path / "custom.json"
    path.write_text(json.dumps([{"id": "vault", "name": "Vault", "clearance_level": "V", "description": "x"}]), encoding="utf-8")
    monkeypatch.setenv("BUILDINGS_CONFIG_PATH", str(path))
    from backend.config import get_settings

    get_settings.cache_clear()
    assert [b["id"] for b in client.get("/buildings").json()] == ["vault"]
    assert _authenticate(client, "u", "vault", face=FACE).json()["authentication_state"] == "ACCESS_GRANTED"


# ----------------------------------------------------------------------------- enrollment status


def _statuses(client, user):
    return client.get(f"/user/{user}/enrollment-status", params={"application_id": APPLICATION_ID}).json()


def test_enrollment_status_not_registered_registered_updated(client):
    assert _statuses(client, "u")["statuses"] == {"face": "NOT_REGISTERED", "voice": "NOT_REGISTERED", "hand": "NOT_REGISTERED"}
    _enroll(client, "u", "face", FACE)
    profile = _statuses(client, "u")
    assert profile["statuses"]["face"] == "REGISTERED" and profile["modalities"] == {"face": True, "voice": False, "hand": False}
    _enroll(client, "u", "face", OTHER_FACE)  # re-enroll
    assert _statuses(client, "u")["statuses"]["face"] == "UPDATED"
    assert _statuses(client, "u")["statuses"]["voice"] == "NOT_REGISTERED"  # independent
    body = _authenticate(client, "u", BUILDING, face=OTHER_FACE).json()
    assert body["authentication_state"] == "ACCESS_GRANTED"  # the updated enrollment is the one in use


def test_voice_retry_required_after_an_inconsistent_enrollment_and_registered_after_a_good_one(client):
    def enroll_voice(first, second):
        return client.post(
            "/enroll",
            data={"user_id": "u", "modality": "voice", "application_id": APPLICATION_ID},
            files={"image": ("v.wav", first, "audio/wav"), "confirm_image": ("v2.wav", second, "audio/wav")},
        )

    bad = enroll_voice(VOICE, OTHER_VOICE)
    assert bad.status_code == 422 and bad.json()["status"] == "ENROLLMENT_INCONSISTENT"
    assert bad.json()["enrollment_status"] == "RETRY_REQUIRED"
    profile = _statuses(client, "u")
    assert profile["statuses"]["voice"] == "RETRY_REQUIRED" and profile["modalities"]["voice"] is False  # nothing stored
    assert profile["statuses"]["face"] == "NOT_REGISTERED"  # RETRY_REQUIRED is voice-only
    assert _submit(client, "u", ("voice",)).status_code == 409  # cannot authenticate with it

    assert enroll_voice(VOICE, VOICE).status_code == 200
    assert _statuses(client, "u")["statuses"]["voice"] == "REGISTERED"


# ----------------------------------------------------------------------------- public response + template pool


def test_production_response_exposes_only_the_fusion_values(client, monkeypatch):
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    body = _submit(client, "u", ("face", "voice")).json()
    assert set(body) == {
        "user_id", "authentication_state", "status", "authenticated", "fusion_similarity", "fusion_distance",
        "fusion_threshold", "fusion_policy", "matched_modalities", "modalities_used", "active_template_set",
        "template_set_version", "key_version", "authentication_time_ms", "building_id",
        "display_name",  # granted: the user's name (never on a denial)
    }
    assert {"results", "score", "threshold", "distance", "fused_score", "face_similarity", "voice_similarity"}.isdisjoint(body)
    entry = _audit(client, "u")[0]
    assert entry["similarity_scores"] is None and entry["authentication_state"] == "ACCESS_GRANTED"


def test_template_pool_is_four_sets_for_every_enrolled_modality_and_revocation_promotes_the_next(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    pool = client.get("/templates/u", params={"application_id": APPLICATION_ID}).json()
    assert [(s["template_set_version"], s["status"]) for s in pool["sets"]] == [(1, "ACTIVE"), (2, "STANDBY"), (3, "STANDBY"), (4, "STANDBY")]
    assert all(s["modalities"] == ["face", "voice"] for s in pool["sets"])
    revoked = client.post(
        "/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID}, files=_fusion_files(FACE, VOICE)
    )
    assert revoked.status_code == 200 and revoked.json()["new_active_template_set_version"] == 2
    for combo in COMBOS:  # every combination keeps working on the promoted set
        body = _submit(client, "u", combo).json()
        assert body["authentication_state"] == "ACCESS_GRANTED" and body["active_template_set"] == 2


def test_a_face_only_user_can_manage_templates_with_just_their_face(client):
    _enroll_user(client, "u", face=FACE)
    ok = client.post("/revoke-template", data={"user_id": "u", "application_id": APPLICATION_ID}, files=_fusion_files(face=FACE))
    assert ok.status_code == 200
    assert _submit(client, "u", ("face",)).json()["active_template_set"] == 2


# ----------------------------------------------------------------------------- template sets stay consistent as modalities are added


def test_modalities_enrolled_one_at_a_time_share_the_same_sets(client):
    from backend.database import crud

    _enroll(client, "u", "face", FACE)
    before = {(r.modality, r.template_set_version): bytes(r.protected_template) for r in crud.get_set_rows(_db(), "u", APPLICATION_ID)}
    _enroll(client, "u", "voice", VOICE)
    rows = crud.get_set_rows(_db(), "u", APPLICATION_ID)
    after = {(r.modality, r.template_set_version): bytes(r.protected_template) for r in rows}
    assert all(after[k] == v for k, v in before.items())  # earlier templates untouched
    assert len(rows) == 8 and {r.template_set_version for r in rows if r.is_active} == {1}
    for combo in COMBOS:
        assert _submit(client, "u", combo).json()["authentication_state"] == "ACCESS_GRANTED"
