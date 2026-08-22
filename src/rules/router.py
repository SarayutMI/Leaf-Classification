# src/rules/router.py
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.auth.dependencies import require_session
from src.rules import repository, service
from src.rules.schemas import TRAIT_KEYS, VOCAB, RuleGroupIn, RuleTestRequest

# The session check lives on the router, not on each route: a new endpoint
# added here is protected by default instead of only when someone remembers
# the Depends().
router = APIRouter(
    prefix="/api/rules",
    tags=["rules"],
    dependencies=[Depends(require_session)],
)
logger = logging.getLogger(__name__)


# Everything behind require_session is per-user data. Without this a browser
# is free to reuse a 200 it fetched while logged in — served straight from its
# own cache, never reaching the server, so the session check cannot run.
NO_STORE = {"Cache-Control": "no-store, private", "Pragma": "no-cache"}


def _ok(data, code: int = 200):
    return JSONResponse(
        status_code=code,
        content={"code": code, "status": "success", "data": data},
        headers=NO_STORE,
    )


def _combination_text(row: dict) -> str:
    combination = " / ".join(row[key] for key in TRAIT_KEYS)
    return f"Another group already uses the combination {combination}"


def _error(code: int, message: str, details: str | None = None):
    content = {"code": code, "status": "error", "message": message}
    if details:
        content["errors"] = {"type": "VALIDATION_ERROR", "details": details}
    return JSONResponse(status_code=code, content=content, headers=NO_STORE)


# Static paths are declared before /{group_id} so they are never swallowed by
# the parameterised route.
@router.get("/vocab")
async def get_vocab():
    """The class list per trait, so the page builds its dropdowns from the
    same source the models were configured with."""
    return _ok({"vocab": VOCAB})


@router.post("/test")
async def test_rules(body: RuleTestRequest):
    try:
        results = service.match_group(
            traits=body.traits,
            probs=body.probs,
            conf_th=body.conf_th,
            top_n=body.top_n,
        )
    except Exception:
        logger.exception("Rule test failed")
        return _error(500, "Database error")
    return _ok({"results": results})


@router.get("")
async def list_rules():
    try:
        groups = repository.list_groups()
    except Exception:
        logger.exception("Failed to list rule groups")
        return _error(500, "Database error")
    return _ok({"groups": groups})


@router.post("")
async def create_rule(body: RuleGroupIn):
    try:
        row = body.to_row()
        if repository.code_exists(body.code):
            return _error(409, "Duplicate code", f"Rule code '{body.code}' already exists")
        if repository.combination_exists(row):
            return _error(409, "Duplicate rule", _combination_text(row))
        group_id = repository.create_group(row)
        service.invalidate_cache()
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to create rule group")
        return _error(500, "Database error")
    return _ok({"group": group}, code=201)


@router.get("/{group_id}")
async def get_rule(group_id: int):
    try:
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to read rule group %s", group_id)
        return _error(500, "Database error")
    if group is None:
        return _error(404, "Rule group not found")
    return _ok({"group": group})


@router.put("/{group_id}")
async def update_rule(
    group_id: int,
    body: RuleGroupIn,
):
    try:
        row = body.to_row()
        if repository.code_exists(body.code, exclude_id=group_id):
            return _error(409, "Duplicate code", f"Rule code '{body.code}' already exists")
        if repository.combination_exists(row, exclude_id=group_id):
            return _error(409, "Duplicate rule", _combination_text(row))
        if not repository.update_group(group_id, row):
            return _error(404, "Rule group not found")
        service.invalidate_cache()
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to update rule group %s", group_id)
        return _error(500, "Database error")
    return _ok({"group": group})


@router.delete("/{group_id}")
async def delete_rule(group_id: int):
    try:
        deleted = repository.delete_group(group_id)
    except Exception:
        logger.exception("Failed to delete rule group %s", group_id)
        return _error(500, "Database error")
    if not deleted:
        return _error(404, "Rule group not found")
    service.invalidate_cache()
    return _ok({"deleted": group_id})
