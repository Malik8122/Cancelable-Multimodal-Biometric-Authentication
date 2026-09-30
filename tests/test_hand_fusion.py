"""Three-modality fusion: Face + Voice + Hand, every pair, ALL_REQUIRED / WEIGHTED / AT_LEAST_TWO, failure states.
Face/voice: deterministic stub embedders (tests/test_flexible_auth.py). Hand: the real service on synthetic landmarks."""

from __future__ import annotations

import json

import pytest

from tests.hand_signals import Person, no_hand_capture
from tests.test_flexible_auth import (  # noqa: F401
    APPLICATION_ID,
    FACE,
    OTHER_FACE,
    OTHER_VOICE,
    VOICE,
    _enroll_user,
    client,
)
from tests.test_hand_api import _fresh_hand_service, enroll_hand  # noqa: F401

ALICE, MALLORY = Person(1), Person(2)
GENUINE = {"face": FACE, "voice": VOICE, "hand": ALICE.capture(99)}
WRONG = {"face": OTHER_FACE, "voice": OTHER_VOICE, "hand": MALLORY.capture(98)}


def _files(samples: dict) -> dict:
    files = {}
    if "face" in samples:
        files["face_image"] = ("f.png", samples["face"], "image/png")
    if "voice" in samples:
        files["voice_audio"] = ("v.wav", samples["voice"], "audio/wav")
    if "hand" in samples:
        hand = samples["hand"]
        files["hand_gesture"] = ("g.json", hand if isinstance(hand, str) else json.dumps(hand), "application/json")
    return files


def submit(client, combo, wrong=(), policy=None, overrides=None):
    samples = {m: (WRONG[m] if m in wrong else GENUINE[m]) for m in combo}
    samples.update(overrides or {})
    data = {"user_id": "u", "application_id": APPLICATION_ID}
    if policy:
        data["fusion_policy"] = policy
    return client.post("/authenticate/fusion", data=data, files=_files(samples))


@pytest.fixture
def enrolled(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    assert enroll_hand(client).status_code == 200
    return client


COMBOS = [("face", "hand", "voice"), ("face", "voice"), ("face", "hand"), ("hand", "voice"), ("hand",)]


@pytest.mark.parametrize("combo", COMBOS, ids=["+".join(c) for c in COMBOS])
def test_every_combination_is_granted_for_the_genuine_user(enrolled, combo):
    body = submit(enrolled, combo).json()
    assert body["authentication_state"] == "ACCESS_GRANTED", body
    assert body["modalities_used"] == sorted(combo) and body["matched_modalities"] == sorted(combo)
    # the fused similarity is the mean of the fusion-scale scores of exactly the submitted modalities
    assert body["fusion_similarity"] == pytest.approx(sum(r["score"] for r in body["results"].values()) / len(combo))
    if "hand" in combo:
        assert body["results"]["hand"]["metric"] == "dtw_distance" and body["results"]["hand"]["threshold"] == 0.5


def test_face_and_voice_behaviour_is_unchanged(enrolled):
    body = submit(enrolled, ("face", "voice")).json()
    assert body["results"]["face"]["metric"] == "cosine_estimate" and body["results"]["voice"]["metric"] == "euclidean_estimate"
    assert "hand_gesture" not in body
    assert submit(enrolled, ("face", "voice"), wrong=("voice",)).json()["authentication_state"] == "ACCESS_DENIED"


@pytest.mark.parametrize("wrong", ["face", "voice", "hand"])
def test_all_required_denies_when_any_one_of_three_fails(enrolled, wrong):
    body = submit(enrolled, ("face", "hand", "voice"), wrong=(wrong,)).json()
    assert body["authentication_state"] == "ACCESS_DENIED" and body["fusion_policy"] == "ALL_REQUIRED"
    assert body["matched_modalities"] == sorted({"face", "hand", "voice"} - {wrong})


@pytest.mark.parametrize("wrong", ["face", "voice", "hand"])
def test_at_least_two_grants_two_of_three(enrolled, wrong):
    body = submit(enrolled, ("face", "hand", "voice"), wrong=(wrong,), policy="AT_LEAST_TWO").json()
    assert body["authentication_state"] == "ACCESS_GRANTED" and body["fusion_policy"] == "AT_LEAST_TWO"


def test_at_least_two_denies_one_of_three(enrolled):
    body = submit(enrolled, ("face", "hand", "voice"), wrong=("face", "voice"), policy="AT_LEAST_TWO").json()
    assert body["authentication_state"] == "ACCESS_DENIED" and body["matched_modalities"] == ["hand"]


@pytest.mark.parametrize("combo", [("face", "voice"), ("face", "hand"), ("hand", "voice")])
def test_at_least_two_with_two_submitted_needs_both(enrolled, combo):
    assert submit(enrolled, combo, policy="AT_LEAST_TWO").json()["authentication_state"] == "ACCESS_GRANTED"
    assert submit(enrolled, combo, wrong=(combo[0],), policy="AT_LEAST_TWO").json()["authentication_state"] == "ACCESS_DENIED"


def test_at_least_two_with_one_submitted_is_a_request_error(enrolled):
    response = submit(enrolled, ("hand",), policy="AT_LEAST_TWO")
    assert response.status_code == 422 and "at least two" in response.json()["detail"]


def test_a_hand_capture_error_fails_closed_under_every_policy(enrolled):
    """Face and voice both pass, the hand capture is unusable: never granted on a sample that was never compared."""
    for policy in ("ALL_REQUIRED", "AT_LEAST_TWO", "WEIGHTED"):
        body = submit(enrolled, ("face", "hand", "voice"), policy=policy, overrides={"hand": no_hand_capture()}).json()
        assert body["authentication_state"] == "ACCESS_DENIED", policy
        assert body["hand_gesture"]["decision"] == "NO_HAND_DETECTED"
        assert body["attempt_assessment"]["modality_statuses"]["hand"] == "CAPTURE_ERROR"
        assert body["attempt_assessment"]["modality_statuses"]["face"] == "VERIFIED"


def test_weighted_policy_with_hand(enrolled):
    assert submit(enrolled, ("face", "hand", "voice"), policy="WEIGHTED").json()["authentication_state"] == "ACCESS_GRANTED"


def test_hand_not_enrolled_is_enrollment_required(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    response = submit(client, ("face", "hand", "voice"))
    assert response.status_code == 409 and response.json()["missing_modalities"] == ["hand"]
