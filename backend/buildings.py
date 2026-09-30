"""Buildings: authentication CONTEXT only - an id, a name, a clearance level and a description.

A building carries no biometric policy. Which modalities to enrol and which to present in a session is the user's
decision (see `backend/services/enrollment.py` and `backend/services/authentication.py`); the same fusion engine
handles every combination, whatever the building. `building_id` on an authentication request is a label that is
recorded in the audit log.

The single source of truth is `config/buildings.json`; the backend serves it (`GET /buildings`) and the frontend
renders from that endpoint. `Settings.buildings_config_path` / `BUILDINGS_CONFIG_PATH` can point at another file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "buildings.json"


@dataclass(frozen=True)
class Building:
    id: str
    name: str
    description: str
    clearance_level: str


def parse_buildings(raw: object) -> list[Building]:
    """Validate the config structure; raise `ValueError` naming the first problem."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("buildings config must be a non-empty JSON list")
    buildings: list[Building] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"buildings[{index}] must be an object")
        building_id = item.get("id")
        if not isinstance(building_id, str) or not building_id:
            raise ValueError(f"buildings[{index}] needs a non-empty string 'id'")
        if building_id in seen:
            raise ValueError(f"duplicate building id {building_id!r}")
        seen.add(building_id)
        if "required_modalities" in item:
            raise ValueError(
                f"building {building_id!r} defines 'required_modalities': buildings are authentication context only "
                "and must not carry a biometric policy"
            )
        buildings.append(
            Building(
                id=building_id,
                name=str(item.get("name", building_id)),
                description=str(item.get("description", "")),
                clearance_level=str(item.get("clearance_level", "")),
            )
        )
    return buildings


@lru_cache
def load_buildings(path: str | None = None) -> tuple[Building, ...]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as config_file:
        return tuple(parse_buildings(json.load(config_file)))


def get_building(building_id: str, path: str | None = None) -> Building | None:
    return next((b for b in load_buildings(path) if b.id == building_id), None)
