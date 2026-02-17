"""
Cohort CRUD Endpoints
=====================
Create, list, and manage venue cohorts for comparative analytics.
"""

from datetime import datetime

import sqlalchemy
from fastapi import APIRouter, Depends, HTTPException
from app.auth import require_api_key
from app.responses import success_response
from app.schemas import CohortCreate
from app.database import database, cohorts, cohort_members, venues

router = APIRouter()


@router.post("/cohorts")
async def create_cohort(cohort: CohortCreate, _api_key: str = Depends(require_api_key)):
    """Create a new cohort and optionally add venues to it."""
    query = cohorts.insert().values(
        id=cohort.id,
        name=cohort.name,
        color=cohort.color,
        created_at=datetime.utcnow(),
    )
    try:
        await database.execute(query)
    except Exception:
        raise HTTPException(status_code=400, detail="Cohort with this ID already exists")

    # Add venue memberships
    for venue_id in cohort.venue_ids:
        try:
            await database.execute(
                cohort_members.insert().values(cohort_id=cohort.id, venue_id=venue_id)
            )
        except Exception:
            pass  # Skip duplicates

    return success_response({"cohort_id": cohort.id, "venues_added": len(cohort.venue_ids)})


@router.get("/cohorts")
async def list_cohorts(_api_key: str = Depends(require_api_key)):
    """List all cohorts with their venue counts."""
    from sqlalchemy import func

    rows = await database.fetch_all(
        sqlalchemy.select(cohorts.c.id, cohorts.c.name, cohorts.c.color, cohorts.c.created_at)
    )
    result = []
    for r in rows:
        count = await database.fetch_val(
            sqlalchemy.select(func.count()).select_from(cohort_members).where(
                cohort_members.c.cohort_id == r["id"]
            )
        ) or 0
        result.append({
            "id": r["id"],
            "name": r["name"],
            "color": r["color"],
            "venue_count": count,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return success_response(result)


@router.post("/cohorts/{cohort_id}/venues/{venue_id}")
async def add_venue_to_cohort(cohort_id: str, venue_id: str, _api_key: str = Depends(require_api_key)):
    """Add a venue to a cohort."""
    try:
        await database.execute(
            cohort_members.insert().values(cohort_id=cohort_id, venue_id=venue_id)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Already a member or invalid IDs")
    return success_response({"message": f"Venue '{venue_id}' added to cohort '{cohort_id}'"})


@router.delete("/cohorts/{cohort_id}/venues/{venue_id}")
async def remove_venue_from_cohort(cohort_id: str, venue_id: str, _api_key: str = Depends(require_api_key)):
    """Remove a venue from a cohort."""
    await database.execute(
        cohort_members.delete().where(
            (cohort_members.c.cohort_id == cohort_id) & (cohort_members.c.venue_id == venue_id)
        )
    )
    return success_response({"message": f"Venue '{venue_id}' removed from cohort '{cohort_id}'"})


@router.delete("/cohorts/{cohort_id}")
async def delete_cohort(cohort_id: str, _api_key: str = Depends(require_api_key)):
    """Delete a cohort and its memberships."""
    await database.execute(cohort_members.delete().where(cohort_members.c.cohort_id == cohort_id))
    await database.execute(cohorts.delete().where(cohorts.c.id == cohort_id))
    return success_response({"message": f"Cohort '{cohort_id}' deleted"})
