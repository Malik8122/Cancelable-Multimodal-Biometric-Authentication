"""GET /buildings, GET /building/{id}: authentication context (id, name, clearance level, description).

Buildings define NO biometric requirements - the user chooses which enrolled modalities to present in a session.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.buildings import Building, get_building, load_buildings
from backend.config import Settings, get_settings
from backend.database.schema import BuildingResponse

router = APIRouter()


def _to_response(building: Building) -> BuildingResponse:
    return BuildingResponse(
        id=building.id,
        name=building.name,
        description=building.description,
        clearance_level=building.clearance_level,
    )


@router.get("/buildings", response_model=list[BuildingResponse])
def list_buildings(settings: Settings = Depends(get_settings)) -> list[BuildingResponse]:
    return [_to_response(b) for b in load_buildings(settings.buildings_config_path)]


@router.get("/building/{building_id}", response_model=BuildingResponse)
def get_building_info(building_id: str, settings: Settings = Depends(get_settings)) -> BuildingResponse:
    building = get_building(building_id, settings.buildings_config_path)
    if building is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown building {building_id!r}.")
    return _to_response(building)
