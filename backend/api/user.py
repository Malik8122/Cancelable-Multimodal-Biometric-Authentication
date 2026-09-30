"""POST /users: register a named user. GET /user/{id}: enrolled modalities. POST /user/{id}/display-name: (re)name a
user. DELETE /user/{id}: delete a user's protected templates."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.schema import (
    CreateUserRequest,
    DeleteUserResponse,
    EnrolledModality,
    EnrollmentStatusResponse,
    UserModalitiesResponse,
    UserProfileResponse,
)
from backend.database.session import get_db
from backend.display_names import InvalidDisplayName, normalize_display_name
from backend.services import enrollment

logger = logging.getLogger("backend.api.user")

router = APIRouter()


def _validated_name(body: CreateUserRequest) -> str:
    try:
        return normalize_display_name(body.display_name)
    except InvalidDisplayName as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@router.post("/users", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, db: Session = Depends(get_db)) -> UserProfileResponse:
    """Start a registration: a new user with a server-generated internal id and the given display name.

    Nothing biometric happens here - the face / voice samples are enrolled afterwards under the returned `user_id`
    (POST /enroll, /enroll/face). Duplicate names are allowed: every registration gets its own id.
    """
    user = crud.create_named_user(db, _validated_name(body))
    logger.info("Registered user_id=%s", user.id)  # the name is not logged: it is not needed to trace a request
    return UserProfileResponse(user_id=user.id, display_name=user.username)


@router.post("/user/{user_id}/display-name", response_model=UserProfileResponse)
def update_display_name(
    user_id: str, body: CreateUserRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
) -> UserProfileResponse:
    """Name or rename an existing user (Edit Profile).

    The display name is a label only: templates, key versions and template-set statuses are untouched, because keys are
    derived from the immutable `user_id`, never from the name. Editing a name that is already set is a testing aid
    (`Settings.profile_editing_enabled`, off in production); naming a user who has no name yet is always allowed.
    """
    name = _validated_name(body)
    existing = crud.get_user(db, user_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No such user: {user_id!r}")
    if existing.username and not settings.profile_editing_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Profile editing is disabled in this deployment.")
    user = crud.set_display_name(db, user_id, name)
    logger.info("Display name updated user_id=%s (templates untouched)", user_id)
    return UserProfileResponse(user_id=user.id, display_name=user.username)


@router.get("/user/{user_id}", response_model=UserModalitiesResponse)
def get_user_modalities(user_id: str, db: Session = Depends(get_db)) -> UserModalitiesResponse:
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No such user: {user_id!r}")

    templates = crud.get_templates_for_user(db, user_id, active_only=True)
    return UserModalitiesResponse(
        user_id=user_id,
        enrolled_modalities=[
            EnrolledModality(
                modality=template.modality,
                application_id=template.application_id,
                template_version=template.template_version,
                key_version=template.key_version,
                created_at=template.created_at,
            )
            for template in templates
        ],
    )


@router.get("/user/{user_id}/enrollment-status", response_model=EnrollmentStatusResponse)
def get_enrollment_status(
    user_id: str,
    application_id: str | None = Query(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> EnrollmentStatusResponse:
    """User Enrollment Profile: which modalities are enrolled. Unknown users simply have none (200)."""
    status_ = enrollment.get_user_enrollment_status(db, user_id, application_id or settings.application_id)
    user = crud.get_user(db, user_id)
    return EnrollmentStatusResponse(
        user_id=user_id,
        application_id=status_.application_id,
        modalities=status_.modalities,
        statuses=status_.statuses,
        display_name=crud.display_name_for(db, user_id),
        has_display_name=bool(user is not None and user.username),
        profile_editing_enabled=settings.profile_editing_enabled,
    )


@router.delete("/user/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: str, db: Session = Depends(get_db)) -> DeleteUserResponse:
    templates_deleted = crud.delete_user_templates(db, user_id)
    logger.info("Deleted user_id=%s templates_deleted=%d", user_id, templates_deleted)
    return DeleteUserResponse(success=True, user_id=user_id, templates_deleted=templates_deleted)
