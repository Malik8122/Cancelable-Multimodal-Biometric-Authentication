"""Hand gesture protocol: ENROLLMENT = three independent Z samples (separate performances, kept as three separate
sequences, never overwritten or duplicated); AUTHENTICATION = one sample. Synthetic landmark captures
(tests/hand_signals.py) - a software test, not evidence of accuracy."""

from __future__ import annotations

import itertools
import json

import numpy as np

from tests.hand_signals import Person
from tests.test_flexible_auth import APPLICATION_ID, FACE, _db, _enroll, client  # noqa: F401
from tests.test_hand_api import _fresh_hand_service, _file, enroll_hand, verify_hand  # noqa: F401

ALICE = Person(1)


def _hand_status(client, user="u"):
    return client.get(f"/user/{user}/enrollment-status", params={"application_id": APPLICATION_ID}).json()["modalities"]["hand"]


def _active_hand_sequences(user="u"):
    """The ACTIVE hand template's stored sequences, un-transformed with the set's own key (test-only inspection)."""
    from backend.config import get_settings
    from backend.database import crud
    from template_protection.hkdf_keys import derive_key
    from template_protection.sequence_transform import deserialize, unprotect

    row = next(r for r in crud.get_set_rows(_db(), user, APPLICATION_ID) if r.modality == "hand" and r.template_set_status == "ACTIVE")
    key = derive_key(get_settings().master_secret, application_id=APPLICATION_ID, user_id=user, modality="hand", key_version=row.key_version)
    return unprotect(deserialize(bytes(row.protected_template))[1], key)


def _expected_sequence(capture):
    from backend.services.hand_service import get_hand_service
    from preprocessing.hand_gesture import parse_capture, weighted

    return weighted(get_hand_service().process(parse_capture(json.dumps(capture).encode())).features)


def test_three_independent_samples_are_each_kept_in_order(client):
    captures = [ALICE.capture(seed) for seed in (11, 12, 13)]
    assert enroll_hand(client, captures=captures).status_code == 200
    stored = _active_hand_sequences()
    assert len(stored) == 3  # one template holding three sequences - not three templates, not one averaged sequence
    for n, (sequence, capture) in enumerate(zip(stored, captures), start=1):
        assert np.allclose(sequence, _expected_sequence(capture), atol=1e-4), f"sample {n} was not stored as captured"


def test_samples_are_not_overwritten_or_duplicated(client):
    assert enroll_hand(client, seeds=range(3)).status_code == 200
    stored = _active_hand_sequences()
    for a, b in itertools.combinations(stored, 2):
        assert not np.allclose(a, b), "two stored samples are identical - a sample was duplicated or overwritten"


def test_one_sequence_submitted_three_times_is_refused_and_nothing_is_stored(client):
    one = ALICE.capture(7)
    response = enroll_hand(client, captures=[one] * 3)
    assert response.status_code == 422 and "same captured sequence" in response.json()["detail"]
    assert _hand_status(client) is False


def test_a_single_duplicate_among_distinct_samples_is_refused(client):
    captures = [ALICE.capture(seed) for seed in range(3)]
    captures[2] = captures[0]
    response = enroll_hand(client, captures=captures)
    assert response.status_code == 422 and "samples 1 and 3" in response.json()["detail"]
    assert _hand_status(client) is False


def test_authentication_needs_only_one_gesture_sample(client):
    assert enroll_hand(client).status_code == 200
    body = verify_hand(client, ALICE.capture(99)).json()
    assert body["status"] == "ACCESS_GRANTED" and body["hand_gesture"]["decision"] == "MATCH"
    assert body["modalities_used"] == ["hand"]


def test_fusion_receives_one_outcome_per_submitted_modality(client):
    _enroll(client, "u", "face", FACE)
    assert enroll_hand(client).status_code == 200
    body = client.post("/authenticate/fusion", data={"user_id": "u", "application_id": APPLICATION_ID},
                       files={"face_image": ("f.png", FACE, "image/png"), "hand_gesture": _file(ALICE.capture(42))}).json()
    assert body["status"] == "ACCESS_GRANTED" and sorted(body["matched_modalities"]) == ["face", "hand"]
    assert sorted(body["modalities_used"]) == ["face", "hand"]
