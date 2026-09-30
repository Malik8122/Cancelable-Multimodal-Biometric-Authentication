"""Human-readable display names: a label only, separate from the internal user id (the real identifier)."""

from __future__ import annotations

import pytest

from backend.display_names import MAX_DISPLAY_NAME_LENGTH, InvalidDisplayName, fallback_display_name, normalize_display_name
from tests.test_flexible_auth import (  # noqa: F401 - `client` is a fixture
    APPLICATION_ID,
    FACE,
    OTHER_FACE,
    VOICE,
    _authenticate,
    _enroll_user,
    client,
)


@pytest.mark.parametrize(
    "raw, stored",
    [
        ("Sanya Malik", "Sanya Malik"),
        ("  Sanya   Malik  ", "Sanya Malik"),  # trimmed, inner runs collapsed to one normal space
        ("Mary-Jane O'Neil", "Mary-Jane O'Neil"),
        ("J. R. R. Tolkien", "J. R. R. Tolkien"),
        ("José Álvarez", "José Álvarez"),
    ],
)
def test_valid_names_are_normalized(raw, stored):
    assert normalize_display_name(raw) == stored


@pytest.mark.parametrize("raw", ["", "   ", "123", "Sanya2", "<script>", "Sanya\nMalik\x00", "-Sanya", "a" * (MAX_DISPLAY_NAME_LENGTH + 1)])
def test_invalid_names_are_rejected(raw):
    with pytest.raises(InvalidDisplayName):
        normalize_display_name(raw)


def test_register_returns_a_generated_id_and_the_name(client):
    response = client.post("/users", json={"display_name": "  Sanya Malik "})
    assert response.status_code == 201
    body = response.json()
    assert body["display_name"] == "Sanya Malik"
    assert body["user_id"].startswith("USER-") and "Sanya" not in body["user_id"]  # the name is never the id


@pytest.mark.parametrize("name", ["", "   ", "R2-D2"])
def test_register_rejects_an_invalid_name(client, name):
    assert client.post("/users", json={"display_name": name}).status_code == 422


def test_register_requires_a_name_field(client):
    assert client.post("/users", json={}).status_code == 422


def test_duplicate_names_get_distinct_ids(client):
    first = client.post("/users", json={"display_name": "Sanya Malik"}).json()
    second = client.post("/users", json={"display_name": "Sanya Malik"}).json()
    assert first["user_id"] != second["user_id"]


def test_the_name_is_returned_only_when_access_is_granted(client):
    user_id = client.post("/users", json={"display_name": "Sanya Malik"}).json()["user_id"]
    _enroll_user(client, user_id, face=FACE, voice=VOICE)

    granted = _authenticate(client, user_id, "defence_research_lab", face=FACE, voice=VOICE).json()
    assert granted["authenticated"] is True and granted["display_name"] == "Sanya Malik"

    denied = _authenticate(client, user_id, "defence_research_lab", face=OTHER_FACE, voice=VOICE).json()
    assert denied["authenticated"] is False and "display_name" not in denied


def test_two_users_with_the_same_name_keep_separate_credentials(client):
    a = client.post("/users", json={"display_name": "Alex Kim"}).json()["user_id"]
    b = client.post("/users", json={"display_name": "Alex Kim"}).json()["user_id"]
    _enroll_user(client, a, face=FACE)
    _enroll_user(client, b, face=OTHER_FACE)
    assert _authenticate(client, a, "defence_research_lab", face=FACE).json()["authenticated"] is True
    assert _authenticate(client, b, "defence_research_lab", face=FACE).json()["authenticated"] is False


def test_enrollment_status_reports_the_name(client):
    user_id = client.post("/users", json={"display_name": "Sanya Malik"}).json()["user_id"]
    body = client.get(f"/user/{user_id}/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert body["display_name"] == "Sanya Malik" and body["has_display_name"] is True


def test_a_user_enrolled_before_names_existed_keeps_working_with_a_fallback_label(client):
    _enroll_user(client, "OPERATOR-P5KG0T", face=FACE)  # the old flow: no name anywhere
    status = client.get("/user/OPERATOR-P5KG0T/enrollment-status", params={"application_id": APPLICATION_ID}).json()
    assert status["has_display_name"] is False and status["display_name"] == fallback_display_name("OPERATOR-P5KG0T") == "User P5KG0T"
    granted = _authenticate(client, "OPERATOR-P5KG0T", "defence_research_lab", face=FACE).json()
    assert granted["authenticated"] is True and granted["display_name"] == "User P5KG0T"


def test_an_existing_user_can_be_named_without_touching_templates(client):
    _enroll_user(client, "OPERATOR-P5KG0T", face=FACE)
    response = client.post("/user/OPERATOR-P5KG0T/display-name", json={"display_name": "Sanya Malik"})
    assert response.status_code == 200 and response.json() == {"user_id": "OPERATOR-P5KG0T", "display_name": "Sanya Malik"}
    assert _authenticate(client, "OPERATOR-P5KG0T", "defence_research_lab", face=FACE).json()["authenticated"] is True
    assert client.post("/user/NOBODY/display-name", json={"display_name": "Sanya Malik"}).status_code == 404
