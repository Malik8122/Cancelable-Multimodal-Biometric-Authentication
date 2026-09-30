"""Face enrollment: one-time, five guided poses -> centroid embedding -> T1-T4 templates.

Per pose: MTCNN detection + the existing alignment + one 512-d embedding; only blurry or faceless poses are rejected. The valid
embeddings are averaged and L2-normalized into a centroid, the temporary embeddings are discarded at once, and the template sets
are generated from the centroid ALONE. Authentication is unchanged: one live image, one embedding, compared with the ACTIVE
template. The FaceNet model, thresholds, BioHash and HKDF are untouched.

Stub tests use the deterministic embedders of tests/test_flexible_auth.py (real services, DB, HTTP); the last tests run the REAL
MTCNN + FaceNet on a real portrait.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from embeddings.centroid import centroid_embedding
from tests.test_flexible_auth import (  # noqa: F401  (client is a fixture)
    APPLICATION_ID,
    BUILDING,
    FACE,
    OTHER_FACE,
    _authenticate,
    _db,
    _enroll,
    _noise_png,
    _png,
    client,
)

POSES = ("front", "left", "right", "up", "down")


def _flat():
    return _png(np.full((300, 300, 3), 128, dtype=np.uint8))  # no contrast: 'no face'


def _blurry():
    rng = np.random.default_rng(0)
    return _png(np.clip(120 + rng.normal(0, 3, (300, 300, 3)), 0, 255).astype(np.uint8))  # low contrast: 'blurry'


def _files(**poses):
    return {f"pose_{name}": (f"{name}.png", data, "image/png") for name, data in poses.items()}


def _enroll_poses(client, user="u", **poses):
    return client.post("/enroll", data={"user_id": user, "modality": "face", "application_id": APPLICATION_ID}, files=_files(**poses))


FIVE = {name: _noise_png(10 + i) for i, name in enumerate(POSES)}  # five different valid captures


# ------------------------------------------------------------------ the centroid


def test_centroid_is_the_normalized_mean_of_unit_embeddings():
    rng = np.random.default_rng(0)
    embs = [rng.standard_normal(512) for _ in range(5)]
    centroid = centroid_embedding(embs)
    assert centroid.shape == (512,) and np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-6)
    expected = np.mean([e / np.linalg.norm(e) for e in embs], axis=0)
    assert np.allclose(centroid, expected / np.linalg.norm(expected), atol=1e-6)
    # it sits closer to every pose than the poses sit to each other on average
    unit = [e / np.linalg.norm(e) for e in embs]
    assert all(float(u @ centroid) > 0.3 for u in unit)


def test_centroid_edge_cases():
    single = np.array([3.0, 4.0, 0.0])
    assert np.allclose(centroid_embedding([single]), single / 5.0, atol=1e-6)  # one pose -> itself, normalized
    assert np.allclose(centroid_embedding([single, single * 2]), single / 5.0, atol=1e-6)  # magnitude does not matter
    with pytest.raises(ValueError):
        centroid_embedding([])
    with pytest.raises(ValueError):
        centroid_embedding([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])  # cancels out: no direction
    with pytest.raises(ValueError):
        centroid_embedding([np.zeros(4)])


# ------------------------------------------------------------------ enrollment from five poses


def test_five_poses_enroll_once_and_report_each_pose(client):
    response = _enroll_poses(client, **FIVE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["templates_created"] == 4 and body["active_template_set_version"] == 1 and body["standby_template_set_versions"] == [2, 3, 4]
    assert body["poses_valid"] == 5
    assert [(r["pose"], r["status"]) for r in body["pose_results"]] == [(p, "VALID") for p in POSES]
    assert "protected_template" not in body and "embedding" not in str(body)


def test_templates_are_generated_from_the_centroid_alone(client):
    from backend.config import get_settings
    from backend.database import crud
    from backend.services import get_service_for_modality
    from backend.services.base_service import build_entry

    assert _enroll_poses(client, **FIVE).status_code == 200
    service = get_service_for_modality("face")
    from backend.utils import decode_image

    embeddings = [service.embed(decode_image(FIVE[p])) for p in POSES]
    centroid = centroid_embedding(embeddings)
    stored = {r.template_set_version: bytes(r.protected_template) for r in crud.get_set_rows(_db(), "u", APPLICATION_ID)}
    assert len(stored) == 4
    for version in (1, 2, 3, 4):  # every set = BioHash of the CENTROID under its own HKDF key version
        expected = build_entry(get_settings(), centroid, user_id="u", application_id=APPLICATION_ID, modality="face", key_version=version)
        assert stored[version] == expected.protected_template
    # ...and no single pose's template equals it
    for embedding in embeddings:
        single = build_entry(get_settings(), embedding, user_id="u", application_id=APPLICATION_ID, modality="face", key_version=1)
        assert single.protected_template != stored[1]


def test_only_protected_templates_are_stored_and_the_temporary_embeddings_are_discarded(client, monkeypatch):
    from backend.database import crud
    from backend.database.models import ProtectedTemplate
    from backend.services import get_service_for_modality

    service = get_service_for_modality("face")
    kept = []
    original = service.pipeline.embed_poses

    def spy(captures):
        embeddings, report = original(captures)
        kept.append(embeddings)  # a reference to the very list the service receives
        return embeddings, report

    monkeypatch.setattr(service.pipeline, "embed_poses", spy)
    assert _enroll_poses(client, **FIVE).status_code == 200
    assert len(kept) == 1 and kept[0] == []  # the five temporary embeddings were cleared right after the centroid was taken
    rows = crud.get_set_rows(_db(), "u", APPLICATION_ID)
    assert len(rows) == 4 and all(len(r.protected_template) == 32 for r in rows)  # 256-bit templates only
    columns = {c.name for c in ProtectedTemplate.__table__.columns}
    assert not any("embedding" in name or "image" in name for name in columns)  # nowhere to keep one


def test_only_blurry_or_faceless_poses_are_rejected_and_the_rest_are_averaged(client):
    from backend.config import get_settings
    from backend.database import crud
    from backend.services import get_service_for_modality
    from backend.services.base_service import build_entry
    from backend.utils import decode_image

    poses = {**FIVE, "left": _flat(), "up": _blurry()}  # two bad captures
    response = _enroll_poses(client, **poses)
    assert response.status_code == 200 and response.json()["poses_valid"] == 3
    verdicts = {r["pose"]: r["status"] for r in response.json()["pose_results"]}
    assert verdicts == {"front": "VALID", "left": "NO_FACE", "right": "VALID", "up": "BLURRY", "down": "VALID"}
    service = get_service_for_modality("face")
    centroid = centroid_embedding([service.embed(decode_image(FIVE[p])) for p in ("front", "right", "down")])  # valid ones only
    expected = build_entry(get_settings(), centroid, user_id="u", application_id=APPLICATION_ID, modality="face", key_version=1)
    assert bytes(crud.get_active_template(_db(), "u", "face", APPLICATION_ID).protected_template) == expected.protected_template


def test_too_few_usable_poses_enroll_nothing(client):
    from backend.database import crud

    response = _enroll_poses(client, front=FIVE["front"], left=_flat(), right=_blurry(), up=_flat(), down=FIVE["down"])
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "FACE_CAPTURE_REJECTED" and "Nothing was enrolled" in body["detail"]
    assert {r["pose"]: r["status"] for r in body["pose_results"]} == {
        "front": "VALID", "left": "NO_FACE", "right": "BLURRY", "up": "NO_FACE", "down": "VALID",
    }
    assert crud.get_set_rows(_db(), "u", APPLICATION_ID) == []
    assert client.get("/user/u/enrollment-status", params={"application_id": APPLICATION_ID}).json()["statuses"]["face"] == "NOT_REGISTERED"


def test_pose_uploads_are_face_only_and_an_input_is_required(client):
    response = client.post(
        "/enroll", data={"user_id": "u", "modality": "voice", "application_id": APPLICATION_ID}, files=_files(front=FACE)
    )
    assert response.status_code == 422 and "only used for face" in response.json()["detail"]
    empty = client.post("/enroll", data={"user_id": "u", "modality": "face", "application_id": APPLICATION_ID})
    assert empty.status_code == 422


def test_a_single_face_image_still_enrolls(client):
    body = _enroll(client, "u", "face", FACE)
    assert body["templates_created"] == 4 and "pose_results" not in body


def test_re_enrolling_with_five_new_poses_updates_the_face(client):
    assert _enroll_poses(client, **FIVE).status_code == 200
    again = _enroll_poses(client, **{p: _noise_png(50 + i) for i, p in enumerate(POSES)})
    assert again.status_code == 200
    profile = client.get("/user/u/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert profile["statuses"]["face"] == "UPDATED"


# ------------------------------------------------------------------ per-pose check (immediate retake)


def test_check_pose_reports_a_verdict_and_stores_nothing(client):
    from backend.database import crud

    def check(image, pose="front"):
        return client.post("/enroll/face/check-pose", data={"pose": pose}, files={"image": ("p.png", image, "image/png")})

    ok = check(FACE)
    assert ok.status_code == 200 and ok.json() == {"pose": "front", "status": "VALID", "valid": True}
    no_face = check(_flat(), "left").json()
    assert no_face["status"] == "NO_FACE" and no_face["valid"] is False and "No face was detected" in no_face["detail"]
    blurry = check(_blurry(), "up").json()
    assert blurry["status"] == "BLURRY" and "too blurry" in blurry["detail"]
    assert check(FACE, "sideways").status_code == 422
    assert crud.get_set_rows(_db(), "u", APPLICATION_ID) == []


# ------------------------------------------------------------------ authentication is unchanged


def test_authentication_uses_one_live_image_and_the_active_centroid_template(client, monkeypatch):
    from backend.services import get_service_for_modality

    assert _enroll_poses(client, **FIVE).status_code == 200
    service = get_service_for_modality("face")
    calls = {"embed": 0, "embed_poses": 0}
    embed, embed_poses = service.embed, service.pipeline.embed_poses
    monkeypatch.setattr(service, "embed", lambda raw: (calls.__setitem__("embed", calls["embed"] + 1), embed(raw))[1])
    monkeypatch.setattr(service.pipeline, "embed_poses", lambda c: (calls.__setitem__("embed_poses", calls["embed_poses"] + 1), embed_poses(c))[1])

    body = _authenticate(client, "u", BUILDING, face=FIVE["front"]).json()
    assert calls == {"embed": 1, "embed_poses": 0}  # one live image, one embedding, no pose logic
    assert body["modalities_used"] == ["face"] and body["active_template_set"] == 1
    # compared with the CENTROID template, not with a single pose: even a pose the user enrolled with is not a perfect match
    assert body["results"]["face"]["score"] < 1.0
    # a wrong face is still denied
    assert _authenticate(client, "u", BUILDING, face=OTHER_FACE).json()["authentication_state"] == "ACCESS_DENIED"


# ------------------------------------------------------------------ the REAL MTCNN + FaceNet


def _portrait():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("facenet_pytorch")
    import cv2

    path = pathlib.Path(matplotlib.__file__).parent / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    if not path.exists():
        pytest.skip("no real face image available")
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _jpg(image):
    import cv2

    return cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()


def _variants(image):
    """Stand-ins for the five poses (rotations / shifts) and for a fresh live capture (real head turns are not available offline)."""
    import cv2

    def rot(angle):
        m = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
        return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def shift(dx, dy):
        return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    poses = {"front": image, "left": rot(14), "right": rot(-14), "up": shift(0, -22), "down": shift(0, 22)}
    # Fresh captures with mild appearance change. (Exposure changes are left out on purpose: they land within 0.01 of the 0.90
    # threshold - see the notes in docs/MULTI_TEMPLATE_ARCHITECTURE.md - and would make this test assert on noise.)
    live = [shift(-5, 3), rot(5), rot(8), rot(-10)]
    return poses, live


@pytest.fixture
def real_client(tmp_path, monkeypatch):
    pytest.importorskip("facenet_pytorch")
    from fastapi.testclient import TestClient

    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'real_face.db'}")
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()


def test_real_models_five_poses_enroll_and_a_fresh_capture_authenticates(real_client):
    poses, live = _variants(_portrait())
    response = _enroll_poses(real_client, **{name: _jpg(img) for name, img in poses.items()})
    assert response.status_code == 200, response.text
    assert response.json()["poses_valid"] == 5 and response.json()["templates_created"] == 4
    for capture in live:  # one live image each; mild appearance changes still authenticate
        body = _authenticate(real_client, "u", BUILDING, face=_jpg(capture)).json()
        assert body["authentication_state"] == "ACCESS_GRANTED", body["results"]


def test_real_mtcnn_rejects_only_blurry_and_faceless_captures(real_client):
    import cv2

    portrait = _portrait()

    def check(image):
        return real_client.post("/enroll/face/check-pose", data={"pose": "front"}, files={"image": ("p.jpg", _jpg(image), "image/jpeg")}).json()

    assert check(portrait)["status"] == "VALID"
    assert check(cv2.convertScaleAbs(portrait, alpha=0.5, beta=0))["status"] == "VALID"  # dim is NOT rejected
    assert check(cv2.GaussianBlur(portrait, (0, 0), 4))["status"] == "BLURRY"
    assert check(np.random.default_rng(0).integers(0, 256, portrait.shape, dtype=np.uint8))["status"] == "NO_FACE"

    poses, _ = _variants(portrait)
    mixed = {**{n: _jpg(i) for n, i in poses.items()}, "up": _jpg(cv2.GaussianBlur(portrait, (0, 0), 4))}
    response = _enroll_poses(real_client, **mixed)
    assert response.status_code == 200 and response.json()["poses_valid"] == 4  # the blurry pose was skipped, the rest averaged
    assert {r["pose"]: r["status"] for r in response.json()["pose_results"]}["up"] == "BLURRY"
